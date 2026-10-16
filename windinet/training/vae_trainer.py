# Based on finetune_decoder.py from the WinDiNet development codebase.

"""Fine-tuning the LTX-Video VAE on 4-channel shockwave CFD fields.

Two ways to feed the CFD fields through the 3-channel (RGB) pretrained VAE
(``adapter.mode``):

* ``inflate``: widen ``encoder.conv_in`` / ``decoder.conv_out`` to 4 channels
  (see ``windinet.vae_adapter.inflate_vae_io_channels``). The decoder always
  trains; the encoder trains as configured by ``freeze_conv_in`` /
  ``unfreeze_encoder_trunk`` (decoder only, decoder + conv_in, or everything).
* ``adapter``: keep the VAE frozen behind small learned 4<->3 channel adapters.

The objective is a fixed-weight sum of reconstruction terms (RMSE, H1, SSIM,
...) plus optional latent regularisers (KL, SDS, sliced-Wasserstein); see
``windinet.losses`` and ``VaeTrainerConfig``.
"""

import csv
import math
import os
import shutil
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import rich
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from pydantic import BaseModel
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Subset

from windinet.config import VaeTrainerConfig
from windinet.inference.model_loader import load_vae
from windinet.vae_adapter import (
    AdaptedVAE,
    inflate_vae_io_channels,
    load_adapted_vae,
    load_inflated_vae_checkpoint,
)

from windinet.losses import (
    reconstruction_losses,
    SlicedWassersteinPrior,
    SSIMLoss,
    vrms_loss,
    vrms_per_channel,
)

from windinet.training.shockwave_data import ShockWaveDataset, build_shockwave_video, parse_gamma
from windinet.training.sds_loss import SdsDistillationLoss
from windinet.training.vae_visualization import (
    denormalize_fields,
    save_metrics_history,
    save_reconstruction_panels,
    save_reconstruction_video,
)
from windinet.utils import logger

IS_MAIN_PROCESS = os.environ.get("LOCAL_RANK", "0") == "0"


@torch.no_grad()
def vrmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Variance-normalized RMSE averaged over the batch, as a float (the val_vrmse metric)."""
    return float(vrms_loss(pred.float(), target.float(), eps=eps).item())


@torch.no_grad()
def vrmse_per_channel(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-channel VRMSE as a [C] tensor, in cfg.data.channel_order."""
    return vrms_per_channel(pred.float(), target.float(), eps=eps)


def warmup_cosine_factor(
    step: int, *, warmup_steps: int, total_steps: int, warmup_start_factor: float, floor: float
) -> float:
    """LR multiplier: linear warmup from warmup_start_factor to 1, then cosine decay to `floor`."""
    if step < warmup_steps:
        return warmup_start_factor + (1.0 - warmup_start_factor) * (step / max(1, warmup_steps))
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


# With adapter.unfreeze_encoder_trunk (inflate mode), the encoder beyond
# conv_in also trains: the spatially-downsampling stages down_blocks[0:3] plus
# the tail (down_blocks[-1], mid_block, norm_out, conv_out), i.e. the whole
# encoder. A head-vs-tail unfreeze sweep favoured unfreezing all of it.
_UNFROZEN_DOWN_BLOCKS = [0, 1, 2]


class VaeTrainer:
    def __init__(self, config: VaeTrainerConfig) -> None:
        self._config = config
        # Resume state is read here, before train() may wipe output_dir, so a run
        # can resume into its own output_dir.
        self._start_epoch = 1
        self._resume_global_step = 0
        self._resume_state: Optional[dict] = None
        # Best-checkpoint tracking, and per-epoch checkpoints (oldest first) for
        # keep_last_n pruning.
        self._best_metric_value = math.inf
        self._best_epoch: int | None = None
        self._best_ckpt_path: Path | None = None
        self._checkpoint_paths: list[Path] = []
        self._print_config(config)
        self._setup_accelerator()
        self._load_vae()
        self._collect_trainable_params()
        self._load_resume_state()
        self._init_wandb()

        self.ssim_loss = SSIMLoss(
            channels=len(config.data.channel_order),
            window_size=11,
            sigma=1.5,
        ).to(self._accelerator.device)


        self._loss_weights = dict(config.loss_weighting.weights)

        # SDS runs a frozen 2B-parameter transformer, so load it only when enabled.
        self._sds_loss = (
            SdsDistillationLoss(config.sds, self._accelerator.device) if config.sds.enabled else None
        )
        # Cheap, so always computed and logged; part of the loss only when weighted.
        self._swd_prior = SlicedWassersteinPrior(
            num_projections=config.swd.num_projections,
            bank_size=config.swd.bank_size,
            device=self._accelerator.device,
            seed=config.seed + self._accelerator.process_index,
        )

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def _load_vae(self) -> None:
        """Load the VAE, adapt it to the CFD channels and set which parts train."""
        self._vae = load_vae(self._config.model.model_source, dtype=torch.float32)

        adapter_cfg = self._config.adapter
        self._adapter_meta = None
        self._inflated = False
        if adapter_cfg.enabled and adapter_cfg.mode == "adapter":
            # On resume, the checkpoint supplies the fine-tuned decoder + adapters.
            self._vae, meta = load_adapted_vae(
                self._vae,
                ckpt_path=self._config.resume_from or adapter_cfg.checkpoint,
                device="cpu",
                dtype=torch.float32,
                channels=adapter_cfg.channels,
                k=adapter_cfg.hidden_channels,
                activation=adapter_cfg.activation,
                identity_init=adapter_cfg.identity_init,
                default_temb=adapter_cfg.default_temb,
            )
            logger.info(
                "VAE adapters enabled: "
                f"channels={meta['channels']}, hidden_channels={meta['k']}, "
                f"activation={meta['activation']}"
            )
            self._adapter_meta = meta
        elif adapter_cfg.enabled and adapter_cfg.mode == "inflate":
            n = len(adapter_cfg.channels)
            inflate_vae_io_channels(self._vae, n=n)
            self._inflated = True
            logger.info(f"VAE inflated to {n} native channels (new channel zero-initialised)")
            # On resume, load the fine-tuned weights over the freshly inflated VAE.
            if self._config.resume_from is not None:
                meta = load_inflated_vae_checkpoint(
                    self._vae,
                    ckpt_path=self._config.resume_from,
                    device="cpu",
                    dtype=torch.float32,
                )
                logger.info(
                    f"Resumed inflated VAE weights from {self._config.resume_from} "
                    f"(checkpoint epoch={meta.get('epoch', '?')})"
                )

        # Freeze everything, then unfreeze what this configuration trains.
        for p in self._vae.parameters():
            p.requires_grad_(False)

        decoder = self._get_decoder()
        for p in decoder.parameters():
            p.requires_grad_(True)

        if isinstance(self._vae, AdaptedVAE):
            self._vae.in_adapter.requires_grad_(True)
            self._vae.out_adapter.requires_grad_(True)

        # Inflate mode: conv_in must train for the new (zero-initialised) channel
        # to reach the encoder at all. freeze_conv_in=true trains the decoder
        # alone, which keeps the latent space (and any DiT trained on it) valid.
        if self._inflated and not adapter_cfg.freeze_conv_in:
            for p in self._get_encoder_conv_in().parameters():
                p.requires_grad_(True)
        elif self._inflated:
            logger.info(
                "encoder.conv_in FROZEN (freeze_conv_in=true): decoder-only refinement. "
                "The latent space is unchanged, so existing latents and any DiT trained on "
                "them remain valid and this VAE can be swapped in at inference."
            )

        # The rest of the encoder (see _UNFROZEN_DOWN_BLOCKS): inflate mode only.
        self._encoder_extra_unfrozen = self._inflated and adapter_cfg.unfreeze_encoder_trunk
        if self._encoder_extra_unfrozen:
            extra_modules = self._get_encoder_extra_modules()
            for module in extra_modules:
                for p in module.parameters():
                    p.requires_grad_(True)
            logger.info(
                f"Encoder extra modules UNFROZEN: down_blocks{_UNFROZEN_DOWN_BLOCKS} + "
                "tail (down_blocks[-1]+mid_block+norm_out+conv_out)"
            )
        else:
            reason = "mode='adapter'" if not self._inflated else "unfreeze_encoder_trunk=false"
            logger.info(f"Encoder extra modules (down_blocks/tail) FROZEN ({reason})")

        if self._config.optimization.enable_gradient_checkpointing:
            base_vae = self._vae.vae if isinstance(self._vae, AdaptedVAE) else self._vae
            base_vae.enable_gradient_checkpointing()
            logger.info("VAE gradient checkpointing enabled")

        # Frozen modules stay in eval mode; _set_trainable_modules_mode switches
        # only the trainable ones.
        self._vae.eval()

        logger.info(f"VAE loaded. Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

    def _unwrap_vae(self) -> nn.Module:
        return self._accelerator.unwrap_model(self._vae)

    def _get_decoder(self) -> nn.Module:
        vae = self._unwrap_vae()
        if isinstance(vae, AdaptedVAE):
            return vae.vae.decoder
        for name in ("decoder", "vae_decoder"):
            if hasattr(vae, name) and isinstance(getattr(vae, name), nn.Module):
                return getattr(vae, name)
        return vae

    def _get_encoder(self) -> nn.Module:
        vae = self._unwrap_vae()
        base_vae = vae.vae if isinstance(vae, AdaptedVAE) else vae
        return base_vae.encoder

    def _get_encoder_conv_in(self) -> nn.Module:
        return self._get_encoder().conv_in

    def _get_encoder_tail_modules(self) -> list[nn.Module]:
        encoder = self._get_encoder()
        return [encoder.down_blocks[-1], encoder.mid_block, encoder.norm_out, encoder.conv_out]

    def _get_encoder_downblock_modules(self) -> list[nn.Module]:
        """The spatial downsampling stages in _UNFROZEN_DOWN_BLOCKS."""
        encoder = self._get_encoder()
        return [encoder.down_blocks[i] for i in _UNFROZEN_DOWN_BLOCKS]

    def _get_encoder_extra_modules(self) -> list[nn.Module]:
        """Every encoder module beyond conv_in that unfreeze_encoder_trunk trains."""
        return self._get_encoder_downblock_modules() + self._get_encoder_tail_modules()

    def _collect_trainable_params(self) -> None:
        self._trainable_params = [p for p in self._vae.parameters() if p.requires_grad]
        vae = self._unwrap_vae()
        base_vae = vae.vae if isinstance(vae, AdaptedVAE) else vae
        # Sanity check: the only trainable encoder parameters are conv_in and the
        # unfreeze_encoder_trunk modules.
        conv_in_trainable = (
            sum(p.numel() for p in self._get_encoder_conv_in().parameters() if p.requires_grad)
            if self._inflated else 0
        )
        extra_trainable = sum(
            p.numel()
            for module in self._get_encoder_extra_modules()
            for p in module.parameters()
            if p.requires_grad
        )
        encoder_trainable = sum(p.numel() for p in base_vae.encoder.parameters() if p.requires_grad)
        expected_encoder_trainable = conv_in_trainable + extra_trainable
        if encoder_trainable != expected_encoder_trainable:
            raise RuntimeError(
                f"Only encoder.conv_in and the whole-structure-unfreeze modules "
                f"(_UNFROZEN_DOWN_BLOCKS + tail) may be trainable, but "
                f"{encoder_trainable:,} encoder parameters are trainable "
                f"({expected_encoder_trainable:,} expected)"
            )

        decoder_trainable = sum(p.numel() for p in self._get_decoder().parameters() if p.requires_grad)
        in_adapter_trainable = (
            sum(p.numel() for p in vae.in_adapter.parameters() if p.requires_grad)
            if isinstance(vae, AdaptedVAE) else 0
        )
        out_adapter_trainable = (
            sum(p.numel() for p in vae.out_adapter.parameters() if p.requires_grad)
            if isinstance(vae, AdaptedVAE) else 0
        )
        logger.info(
            "Trainable parameters: "
            f"encoder_conv_in={conv_in_trainable:,}, encoder_extra={extra_trainable:,}, "
            f"decoder={decoder_trainable:,}, in_adapter={in_adapter_trainable:,}, "
            f"out_adapter={out_adapter_trainable:,}"
        )

    # ------------------------------------------------------------------
    # Resume (optimizer / scheduler / RNG state)
    # ------------------------------------------------------------------

    @staticmethod
    def _state_file_for(ckpt_path: str | Path) -> Path:
        p = Path(ckpt_path)
        # vae_shockwave_epoch005.safetensors -> vae_shockwave_epoch005.state.pt
        return p.parent / (p.stem + ".state.pt")

    def _load_resume_state(self) -> None:
        """Read resume_from's sibling .state.pt; train() applies it once the optimizer exists."""
        if self._config.resume_from is None:
            return
        if self._config.resume_weights_only:
            logger.info(
                "resume_weights_only=true: loaded the checkpoint's weights but ignoring its "
                "training state. Optimizer, LR schedule and epoch counter start fresh, so the "
                "configured learning_rate/epochs take effect (warm restart)."
            )
            return
        state_path = self._state_file_for(self._config.resume_from)
        if not state_path.exists():
            logger.warning(
                f"resume_from set but no training-state file at {state_path}; "
                "model weights will be warm-started, but optimizer/scheduler/epoch "
                "will start fresh (epoch 1)."
            )
            return
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        self._resume_state = state
        saved_epoch = int(state.get("epoch", 0))
        self._start_epoch = saved_epoch + 1
        self._resume_global_step = int(state.get("global_opt_step", 0))
        # Restore the best metric too, or the first resumed epoch would always
        # overwrite vae_shockwave_best.
        if "best_metric_value" in state:
            self._best_metric_value = float(state["best_metric_value"])
            self._best_epoch = state.get("best_epoch")
            best_path = Path(self._config.output_dir) / "checkpoints" / "vae_shockwave_best.safetensors"
            if best_path.exists():
                self._best_ckpt_path = best_path
        else:
            logger.warning(
                f"{state_path} predates best-metric tracking in the training state; "
                "the first resumed epoch will overwrite the best checkpoint."
            )
        logger.info(
            f"Loaded training state from {state_path}: resuming at epoch "
            f"{self._start_epoch} (saved epoch {saved_epoch}, "
            f"global_opt_step {self._resume_global_step})"
        )

    def _apply_resume_state(self, optimizer, scheduler) -> None:
        """Restore optimizer/scheduler/RNG from the stashed state."""
        state = self._resume_state
        if state is None:
            return
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        rng = state.get("rng")
        if rng is not None:
            import random as _random

            import numpy as _np

            if rng.get("python") is not None:
                _random.setstate(rng["python"])
            if rng.get("numpy") is not None:
                _np.random.set_state(rng["numpy"])
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
            if rng.get("xpu") is not None and hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.set_rng_state_all(rng["xpu"])
        logger.info("Restored optimizer, scheduler and RNG state.")

    def _set_trainable_modules_mode(self, training: bool) -> None:
        self._get_decoder().train(training)
        vae = self._unwrap_vae()
        if isinstance(vae, AdaptedVAE):
            vae.in_adapter.train(training)
            vae.out_adapter.train(training)
        # A frozen conv_in/trunk stays in eval mode with the rest of the frozen encoder.
        if self._inflated and not self._config.adapter.freeze_conv_in:
            self._get_encoder_conv_in().train(training)
        if self._encoder_extra_unfrozen:
            for module in self._get_encoder_extra_modules():
                module.train(training)

    # ------------------------------------------------------------------
    # VAE encode / decode
    # ------------------------------------------------------------------

    def _encode(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode video [B, C, F, H, W].

        Returns (latents, posterior_mean, posterior_logvar): ``latents`` is the
        normalised posterior mean (what the decoder and the DiT see); the raw
        posterior mean/log-variance are returned for the KL term.
        """
        out = self._vae.encode(video)
        posterior_mean = out.latent_dist.mean
        posterior_logvar = out.latent_dist.logvar
        norm_mean = self._vae.latents_mean.view(1, -1, 1, 1, 1).to(posterior_mean.device, posterior_mean.dtype)
        norm_std = self._vae.latents_std.view(1, -1, 1, 1, 1).to(posterior_mean.device, posterior_mean.dtype)
        sf = float(getattr(self._vae.config, "scaling_factor", 1.0))
        latents = (posterior_mean - norm_mean) * sf / norm_std
        return latents, posterior_mean, posterior_logvar

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode normalised latents to video."""
        mean = self._vae.latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        std = self._vae.latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        sf = float(getattr(self._vae.config, "scaling_factor", 1.0))
        z = latents * std / sf + mean
        temb = torch.full(
            (z.shape[0],),
            self._config.adapter.default_temb,
            device=z.device,
            dtype=z.dtype,
        )
        return self._vae.decode(z, temb=temb, return_dict=True).sample

    def _append_grad_norm(self, step: int, epoch: int, norm: float) -> None:
        """Append one optimizer step's pre-clip gradient norm to metrics/grad_norms.csv."""
        path = Path(self._config.output_dir) / "metrics" / "grad_norms.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        with path.open("a") as handle:
            if new_file:
                handle.write("step,epoch,grad_norm\n")
            handle.write(f"{step},{epoch},{norm:.6g}\n")

    @property
    def _sample_posterior(self) -> bool:
        """Whether training decodes a reparameterized posterior sample (KL weighted, kl.decode="sample").

        KL(q(z|x) || N(0, I)) only regularises the posterior variance if the
        decoder actually sees samples z ~ q(z|x); decoding the mean would leave
        sigma free and KL would simply push logvar to 0. Without KL, training
        decodes the posterior mean.
        """
        return self._loss_weights.get("kl", 0.0) != 0.0 and self._config.kl.decode == "sample"

    def _rescale_latents(self, z: torch.Tensor) -> torch.Tensor:
        """Raw encoder-space latents -> the normalised latents _decode expects."""
        norm_mean = self._vae.latents_mean.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
        norm_std = self._vae.latents_std.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
        sf = float(getattr(self._vae.config, "scaling_factor", 1.0))
        return (z - norm_mean) * sf / norm_std

    def _forward_pass(
        self, x: torch.Tensor, sample_posterior: bool = False
    ) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode -> decode.

        sample_posterior=True decodes a reparameterized sample instead of the
        posterior mean (training with KL only, see _sample_posterior).
        Validation, visualization and DiT preprocessing always use the mean.

        Returns (reconstruction, original_frames, posterior_mean,
        posterior_logvar, latents); the posterior is for the KL term, the
        normalised mean latents for the SDS / SWD terms.
        """
        orig_F = x.shape[2]
        latents, posterior_mean, posterior_logvar = self._encode(x)
        if sample_posterior:
            eps = torch.randn_like(posterior_mean)
            z = posterior_mean + torch.exp(0.5 * posterior_logvar) * eps
            recon = self._decode(self._rescale_latents(z))
        else:
            recon = self._decode(latents)
        return recon[:, :, :orig_F], orig_F, posterior_mean, posterior_logvar, latents

    def _sync_grads(self) -> None:
        """Average trainable-parameter gradients across processes, once per optimizer step.

        The VAE is not wrapped in DDP because training calls vae.encode/decode
        directly, which DDP does not hook. Averaging the accumulated gradients
        here is equivalent to DDP's per-microbatch averaging.
        """
        world_size = self._accelerator.num_processes
        for p in self._trainable_params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(world_size)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> Path:
        """Run the fine-tuning loop; returns the path of the checkpoint to use downstream."""
        cfg = self._config
        device = self._accelerator.device
        set_seed(cfg.seed)

        # Data
        data_root = Path(cfg.data.data_root)
        full_dataset = ShockWaveDataset(
            data_root,
            num_sim_frames=cfg.data.num_sim_frames,
        )
        n_eval = min(cfg.data.eval_sims, len(full_dataset))
        n_train = len(full_dataset) - n_eval
        if n_train < 1 or n_eval < 1:
            raise ValueError(
                f"Need at least one training and one evaluation sample; got {n_train} train, {n_eval} eval"
            )
        # Seeded random split. The HDF5 groups are sorted by gamma, so a
        # contiguous slice would hold out whole gamma regimes.
        split_generator = torch.Generator().manual_seed(cfg.seed)
        perm = torch.randperm(len(full_dataset), generator=split_generator).tolist()
        train_set = Subset(full_dataset, perm[:n_train])
        eval_set = Subset(full_dataset, perm[n_train:n_train + n_eval])

        train_loader = DataLoader(
            train_set,
            batch_size=cfg.optimization.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_dataloader_workers,
            drop_last=True,
        )
        # Eval is sharded by hand (strided) rather than with accelerator.prepare(),
        # whose even_batches padding would repeat samples and bias the averages.
        world_size = self._accelerator.num_processes
        rank = self._accelerator.process_index
        eval_shard = Subset(eval_set, list(range(rank, len(eval_set), world_size)))
        eval_loader = DataLoader(
            eval_shard,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.data.num_dataloader_workers,
        )

        # Visualization samples: fixed sims spread over the gamma range (min, ...,
        # max), chosen from the full eval set so the choice does not depend on
        # the number of processes.
        vis_gammas = [parse_gamma(full_dataset.ids[i]) for i in eval_set.indices]
        gamma_order = sorted(range(len(vis_gammas)), key=lambda k: vis_gammas[k])
        n_vis = min(cfg.visualization.num_samples, len(gamma_order))
        if n_vis == 1:
            vis_picks = [gamma_order[len(gamma_order) // 2]]
        else:
            vis_picks = [
                gamma_order[round(i * (len(gamma_order) - 1) / (n_vis - 1))]
                for i in range(n_vis)
            ]
        vis_loader = DataLoader(Subset(eval_set, vis_picks), batch_size=1, shuffle=False)

        logger.info(f"Dataset: {n_train} train, {n_eval} eval samples")

        # Optimizer: one param group per trainable part (decoder, channel
        # interface, encoder trunk). They currently share one LR, but separate
        # groups keep per-part learning rates a one-line change.
        base_lr = cfg.optimization.learning_rate
        fast_lr = base_lr  # adapter/conv_in LR == decoder LR, no separate multiplier
        vae = self._unwrap_vae()
        decoder_params = [p for p in self._get_decoder().parameters() if p.requires_grad]
        fast_params = []
        fast_name = "adapter"
        if isinstance(vae, AdaptedVAE):
            fast_params = (
                [p for p in vae.in_adapter.parameters() if p.requires_grad]
                + [p for p in vae.out_adapter.parameters() if p.requires_grad]
            )
        elif self._inflated:
            fast_params = [p for p in self._get_encoder_conv_in().parameters() if p.requires_grad]
            fast_name = "encoder.conv_in"
        param_groups = [{"params": decoder_params, "lr": base_lr}]
        if fast_params:
            param_groups.append({"params": fast_params, "lr": fast_lr})
            logger.info(f"{fast_name} LR = {fast_lr:.2e} (== decoder LR)")
        if self._encoder_extra_unfrozen:
            extra_lr = base_lr  # encoder extra LR == decoder LR, no separate multiplier
            extra_params = [
                p
                for module in self._get_encoder_extra_modules()
                for p in module.parameters()
                if p.requires_grad
            ]
            param_groups.append({"params": extra_params, "lr": extra_lr})
            extra_desc = [f"down_blocks{_UNFROZEN_DOWN_BLOCKS}", "tail"]
            logger.info(f"encoder extra ({'+'.join(extra_desc)}) LR = {extra_lr:.2e} (== decoder LR)")
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=cfg.optimization.weight_decay,
        )

        # The VAE is not wrapped in DDP (see _sync_grads); Accelerate handles
        # the optimizer, the train loader and mixed precision.
        self._vae.to(self._accelerator.device)
        optimizer = self._accelerator.prepare(optimizer)
        train_loader = self._accelerator.prepare(train_loader)

        # The LR schedule is sized from the per-process (sharded) loader.
        steps_per_epoch = math.ceil(len(train_loader) / cfg.optimization.gradient_accumulation_steps)
        total_opt_steps = max(1, cfg.optimization.epochs * steps_per_epoch)
        warmup_steps = min(cfg.optimization.warmup_steps, total_opt_steps)

        # One multiplicative LambdaLR for every param group, so each group keeps
        # its own base LR through warmup and decay.
        floor = cfg.optimization.min_learning_rate / base_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: warmup_cosine_factor(
                step,
                warmup_steps=warmup_steps,
                total_steps=total_opt_steps,
                warmup_start_factor=cfg.optimization.warmup_start_factor,
                floor=floor,
            ),
        )
        logger.info(
            f"LR schedule: {total_opt_steps} steps ({warmup_steps} warmup, then cosine), "
            f"peak {base_lr:.2e} -> floor {cfg.optimization.min_learning_rate:.2e}"
        )

        # Resume after both exist and after set_seed, so the restored RNG wins.
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._apply_resume_state(optimizer, scheduler)

        self._prepare_output_dir()
        self._save_config()

        # Progress bar
        if IS_MAIN_PROCESS:
            train_progress = Progress(
                TextColumn("Epoch {task.fields[epoch]}"),
                BarColumn(bar_width=40, style="blue"),
                MofNCompleteColumn(),
                TextColumn("Loss: {task.fields[loss]:.4f}"),
                TextColumn("LR: {task.fields[lr]:.2e}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(compact=True),
            )
            live = Live(Panel(train_progress), refresh_per_second=2)
        else:
            train_progress = MagicMock()
            live = nullcontext()

        global_opt_step = self._resume_global_step
        saved_path = None
        metrics_history: list[dict[str, float]] = []
        if self._start_epoch > 1:
            # metrics.csv is rewritten from this list, so keep the finished epochs.
            prev_csv = Path(cfg.output_dir) / "metrics" / "metrics.csv"
            if prev_csv.exists():
                with prev_csv.open(newline="") as handle:
                    metrics_history = [
                        {k: float(v) for k, v in row.items()}
                        for row in csv.DictReader(handle)
                        if float(row["epoch"]) < self._start_epoch
                    ]
                logger.info(f"Loaded {len(metrics_history)} previous epoch rows from {prev_csv}")

        # epochs=0 on a fresh run is allowed: it saves the untrained (inflated)
        # VAE, e.g. for the "unfinetuned" control.
        if self._start_epoch > 1 and self._start_epoch > cfg.optimization.epochs:
            raise ValueError(
                f"resume epoch {self._start_epoch} exceeds configured epochs "
                f"{cfg.optimization.epochs}; increase optimization.epochs to train further."
            )
        if self._start_epoch > 1:
            logger.info(
                f"Resuming VAE finetuning at epoch {self._start_epoch} / "
                f"{cfg.optimization.epochs}"
            )
        else:
            logger.info("Starting VAE decoder finetuning...")

        sample_posterior = self._sample_posterior
        if IS_MAIN_PROCESS:
            logger.info(
                "Training decodes "
                + ("reparameterized posterior samples (KL active)" if sample_posterior else "the posterior mean")
            )

        with live:
            for epoch in range(self._start_epoch, cfg.optimization.epochs + 1):
                epoch_t0 = time.time()
                self._set_trainable_modules_mode(True)
                running_loss = 0.0
                count = 0

                loss_sum = defaultdict(float)

                # Pre-clip gradient norm per optimizer step (spikes precede divergence).
                clip_norms: list[float] = []

                task = train_progress.add_task(
                    f"Epoch {epoch}", total=len(train_loader),
                    epoch=epoch, loss=0.0, lr=cfg.optimization.learning_rate,
                )

                optimizer.zero_grad(set_to_none=True)

                for i, batch in enumerate(train_loader):
                    orig_F = batch["density"].shape[1]
                    x = build_shockwave_video(
                        batch,
                        device=device,
                        channel_mean=cfg.data.channel_mean,
                        channel_std=cfg.data.channel_std,
                        normalization_clip=cfg.data.normalization_clip,
                        channel_order=cfg.data.channel_order,
                    )
                    with self._accelerator.autocast():
                        recon, _, posterior_mean, posterior_logvar, latents = self._forward_pass(
                            x, sample_posterior=sample_posterior
                        )
                    # Losses run in fp32.
                    recon = recon.float()
                    x_target = x[:, :, :orig_F]
                    recon = recon[:, :, :orig_F]

                    losses = reconstruction_losses(
                        pred=recon,
                        target=x_target,
                        ssim_module=self.ssim_loss,
                        wavelet=cfg.loss.wavelet,
                        spatial_level=cfg.loss.spatial_level,
                        temporal_level=cfg.loss.temporal_level,
                        mlw_beta=cfg.loss.mlw_beta,
                        mlw_eps=cfg.loss.mlw_eps,
                        latent_mean=posterior_mean.float(),
                        latent_logvar=posterior_logvar.float(),
                        compute_mlw=cfg.loss_weighting.weights.get("mlw", 0.0) != 0.0,
                        kl_reduction=cfg.kl.reduction,
                    )

                    # Expensive terms (MLW, SDS) are only computed when weighted.
                    losses["sds"] = (
                        self._sds_loss.compute(latents.float(), batch["meta"]["gamma"])
                        if self._sds_loss is not None and cfg.loss_weighting.weights.get("sds", 0.0) != 0.0
                        else recon.new_zeros(())
                    )
                    # Unweighted SWD is only logged, so keep it out of the graph.
                    with torch.set_grad_enabled(cfg.loss_weighting.weights.get("swd", 0.0) != 0.0):
                        losses["swd"] = self._swd_prior(latents)

                    for name, value in losses.items():
                        loss_sum[name] += value.item()
                    weights = self._loss_weights

                    # Terms without a configured weight are logged but not optimised.
                    total_loss = sum(
                        weights.get(name, 0.0) * value
                        for name, value in losses.items()
                    )

                    # Accelerator.backward divides by gradient_accumulation_steps;
                    # rescale the last, possibly shorter, accumulation group.
                    grad_acc = cfg.optimization.gradient_accumulation_steps
                    group_start = (i // grad_acc) * grad_acc
                    group_size = min(grad_acc, len(train_loader) - group_start)
                    backward_loss = total_loss * (grad_acc / group_size)
                    self._accelerator.backward(backward_loss)

                    running_loss += total_loss.item()
                    count += 1

                    do_step = ((i + 1) % cfg.optimization.gradient_accumulation_steps == 0) or (i == len(train_loader) - 1)
                    if do_step:
                        if self._accelerator.num_processes > 1:
                            self._sync_grads()
                        if cfg.optimization.max_grad_norm > 0:
                            total_norm = self._accelerator.clip_grad_norm_(self._trainable_params, cfg.optimization.max_grad_norm)
                            if total_norm is not None:
                                clip_norms.append(float(total_norm))
                                if IS_MAIN_PROCESS:
                                    self._append_grad_norm(global_opt_step + 1, epoch, clip_norms[-1])

                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_opt_step += 1

                        if IS_MAIN_PROCESS and global_opt_step % 10 == 0:
                            lr = optimizer.param_groups[0]["lr"]
                            avg = running_loss / max(count, 1)
                            self._accelerator.print(
                                f"epoch {epoch} step {global_opt_step}  "
                                f"loss={avg:.6f}  lr={lr:.2e}"
                                f"  [rmse={losses['rmse'].item():.4f} "
                                f"h1={losses['h1'].item():.4f} "
                                f"ssim={losses['ssim'].item():.4f} "
                                f"mlw={losses['mlw'].item():.4f}]"
                            )

                    if IS_MAIN_PROCESS:
                        train_progress.update(
                            task, advance=1,
                            loss=running_loss / max(count, 1),
                            lr=optimizer.param_groups[0]["lr"],
                        )

                epoch_losses = {name: value / count for name, value in loss_sum.items()}

                # End of epoch: eval + checkpoint

                train_progress.remove_task(task)
                train_elapsed = time.time() - epoch_t0

                # Every rank evaluates its shard; _evaluate all-reduces, so this
                # must run on all ranks.
                eval_t0 = time.time()
                val_metrics = self._evaluate(eval_loader, device)
                eval_elapsed = time.time() - eval_t0

                monitored = (
                    val_metrics["vrmse"]
                    if cfg.checkpoints.best_metric == "val_vrmse"
                    else val_metrics["total_loss"]
                )

                if IS_MAIN_PROCESS:
                    avg_loss = running_loss / max(count, 1)
                    lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"Epoch {epoch}: train_loss={avg_loss:.6f}  "
                        f"val_loss={val_metrics['total_loss']:.6f}  "
                        f"val_VRMSE={val_metrics['vrmse']:.6f}  "
                        f"val_VRMSE_chmean={val_metrics['vrmse_chmean']:.6f}  lr={lr:.2e}"
                    )

                    metrics_row = {
                        "epoch": epoch,
                        "learning_rate": lr,
                        "train_total_loss": avg_loss,
                        "val_total_loss": val_metrics["total_loss"],
                        "val_vrmse": val_metrics["vrmse"],
                        "grad_norm_mean": sum(clip_norms) / len(clip_norms) if clip_norms else float("nan"),
                        "grad_norm_max": max(clip_norms) if clip_norms else float("nan"),
                        "grad_norm_clipped": sum(n > cfg.optimization.max_grad_norm for n in clip_norms),
                        **{f"train_{name}": value for name, value in epoch_losses.items()},
                        **{
                            f"val_{name}": value
                            for name, value in val_metrics.items()
                            if name not in ("total_loss", "vrmse")
                        },
                    }
                    metrics_history.append(metrics_row)
                    csv_path, curve_path = save_metrics_history(metrics_history, cfg.output_dir)
                    logger.info(
                        f"Metrics updated: {csv_path.relative_to(cfg.output_dir)}, "
                        f"{curve_path.relative_to(cfg.output_dir)}"
                    )

                    self._log_metrics({
                        "epoch": epoch,
                        "epoch/train_loss": avg_loss,
                        "epoch/val_loss": val_metrics["total_loss"],
                        "epoch/eval_vrmse": val_metrics["vrmse"],
                        "epoch/eval_vrmse_chmean": val_metrics["vrmse_chmean"],
                        "epoch/learning_rate": lr,
                    })

                    vis_t0 = time.time()
                    vis_cfg = cfg.visualization
                    is_last_epoch = epoch == cfg.optimization.epochs
                    if vis_cfg.epochs is not None:
                        vis_due = epoch in vis_cfg.epochs
                    else:
                        vis_due = epoch % vis_cfg.interval_epochs == 0 or is_last_epoch
                    if vis_cfg.enabled and vis_due:
                        self._save_visualization(vis_loader, device, epoch)
                    vis_elapsed = time.time() - vis_t0

                    improved = monitored < self._best_metric_value
                    if improved:
                        self._best_metric_value = monitored
                        self._best_epoch = epoch

                    ckpt_t0 = time.time()
                    if cfg.checkpoints.interval and epoch % cfg.checkpoints.interval == 0:
                        saved_path = self._save_checkpoint(
                            epoch, global_opt_step, improved=improved
                        )
                    ckpt_elapsed = time.time() - ckpt_t0

                    logger.info(
                        f"Epoch {epoch} timing: train={train_elapsed:.1f}s "
                        f"eval={eval_elapsed:.1f}s viz={vis_elapsed:.1f}s "
                        f"ckpt={ckpt_elapsed:.1f}s total={time.time() - epoch_t0:.1f}s"
                    )

                self._accelerator.wait_for_everyone()

        # Final save. With save_best_only, improved=False so a worse last epoch
        # cannot overwrite the best weights.
        if IS_MAIN_PROCESS:
            saved_path = self._save_checkpoint(
                cfg.optimization.epochs,
                global_opt_step,
                improved=not cfg.checkpoints.save_best_only,
            )
            if cfg.checkpoints.save_best_only and self._best_epoch is not None:
                logger.info(
                    f"Best epoch: {self._best_epoch} "
                    f"({cfg.checkpoints.best_metric}={self._best_metric_value:.6f})"
                )

            # The run finished, so the rolling "last" checkpoint (kept only for
            # resuming) and its optimizer state are no longer needed.
            if cfg.checkpoints.save_best_only and self._best_ckpt_path is not None:
                last_path = Path(cfg.output_dir) / "checkpoints" / "vae_shockwave_last.safetensors"
                last_state_path = self._state_file_for(last_path)
                removed = [f.name for f in (last_path, last_state_path) if f.exists()]
                for f in (last_path, last_state_path):
                    f.unlink(missing_ok=True)
                if removed:
                    logger.info(f"Run finished: removed {removed} (kept only vae_shockwave_best.safetensors)")

        if self._wandb_run is not None:
            self._wandb_run.finish()

        self._accelerator.end_training()
        return saved_path

    @torch.no_grad()
    def _evaluate(self, eval_loader: DataLoader, device: torch.device) -> dict[str, float]:
        """Training objective and VRMSE over the eval set.

        Runs on every process over its own shard; sums are all-reduced so all
        processes return the same whole-set averages.
        """
        self._set_trainable_modules_mode(False)
        sums = defaultdict(float)
        count = 0
        weights = self._loss_weights
        for batch in eval_loader:
            orig_F = batch["density"].shape[1]
            x = build_shockwave_video(
                batch,
                device=device,
                channel_mean=self._config.data.channel_mean,
                channel_std=self._config.data.channel_std,
                normalization_clip=self._config.data.normalization_clip,
                channel_order=self._config.data.channel_order,
            )
            with self._accelerator.autocast():
                recon, _, posterior_mean, posterior_logvar, latents = self._forward_pass(x)
            recon = recon.float()
            target = x[:, :, :orig_F]
            recon = recon[:, :, :orig_F]
            losses = reconstruction_losses(
                pred=recon,
                target=target,
                ssim_module=self.ssim_loss,
                wavelet=self._config.loss.wavelet,
                spatial_level=self._config.loss.spatial_level,
                temporal_level=self._config.loss.temporal_level,
                mlw_beta=self._config.loss.mlw_beta,
                mlw_eps=self._config.loss.mlw_eps,
                latent_mean=posterior_mean.float(),
                latent_logvar=posterior_logvar.float(),
                compute_mlw=weights.get("mlw", 0.0) != 0.0,
                kl_reduction=self._config.kl.reduction,
            )
            losses["sds"] = (
                self._sds_loss.compute(latents.float(), batch["meta"]["gamma"])
                if self._sds_loss is not None and weights.get("sds", 0.0) != 0.0
                else recon.new_zeros(())
            )
            # Compared against the training bank, which eval does not update.
            losses["swd"] = self._swd_prior(latents, update_bank=False)
            sums["total_loss"] += float(sum(weights.get(name, 0.0) * value for name, value in losses.items()).item())
            sums["vrmse"] += vrmse(recon, target)
            per_channel_vrmse = vrmse_per_channel(recon, target)
            for name, value in zip(self._config.data.channel_order, per_channel_vrmse.tolist()):
                sums[f"vrmse_{name}"] += value
            # Channel-mean VRMSE (The Well's convention, the headline metric).
            # The pooled "vrmse" uses one variance over all channels, which also
            # counts the spread between channel means and so reads lower.
            sums["vrmse_chmean"] += float(per_channel_vrmse.mean().item())
            for name, value in losses.items():
                sums[name] += float(value.item())
            count += 1
        self._set_trainable_modules_mode(True)

        if self._accelerator.num_processes > 1:
            keys = list(sums.keys())
            # float32: fp64 collectives are not reliably supported on XPU/oneCCL.
            totals = torch.tensor([sums[k] for k in keys] + [float(count)], device=device, dtype=torch.float32)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            *summed, total_count = totals.tolist()
            sums = dict(zip(keys, summed))
            count = total_count

        return {name: value / max(1, count) for name, value in sums.items()}

    @torch.no_grad()
    def _save_visualization(self, vis_loader: DataLoader, device: torch.device, epoch: int) -> None:
        """Render GT / prediction / residual panels for the fixed gamma-spread samples."""
        cfg = self._config
        vis_cfg = cfg.visualization
        self._set_trainable_modules_mode(False)
        saved_count = 0
        video_count = 0
        for sample_index, batch in enumerate(vis_loader):
            orig_F = batch["density"].shape[1]
            x = build_shockwave_video(
                batch,
                device=device,
                channel_mean=cfg.data.channel_mean,
                channel_std=cfg.data.channel_std,
                normalization_clip=cfg.data.normalization_clip,
                channel_order=cfg.data.channel_order,
            )
            with self._accelerator.autocast():
                recon, _, _, _, _ = self._forward_pass(x)
            recon = recon.float()
            target = denormalize_fields(
                x[:, :, :orig_F],
                cfg.data.channel_mean,
                cfg.data.channel_std,
                cfg.data.normalization_clip,
            )
            prediction = denormalize_fields(
                recon[:, :, :orig_F],
                cfg.data.channel_mean,
                cfg.data.channel_std,
                cfg.data.normalization_clip,
            )
            sample_id_value = batch.get("id", [f"sample_{sample_index:04d}"])
            sample_id = str(sample_id_value[0] if isinstance(sample_id_value, (list, tuple)) else sample_id_value)
            paths = save_reconstruction_panels(
                prediction=prediction[0],
                target=target[0],
                sample_id=sample_id,
                label=f"epoch_{epoch:04d}",
                frame_numbers=vis_cfg.frame_numbers,
                channel_names=cfg.adapter.channels,
                output_dir=cfg.output_dir,
                dpi=vis_cfg.dpi,
            )
            saved_count += len(paths)
            if vis_cfg.save_video:
                save_reconstruction_video(
                    prediction=prediction[0],
                    target=target[0],
                    sample_id=sample_id,
                    label=f"epoch_{epoch:04d}",
                    channel_names=cfg.adapter.channels,
                    output_dir=cfg.output_dir,
                    fps=vis_cfg.video_fps,
                )
                video_count += 1
        self._set_trainable_modules_mode(True)
        logger.info(
            f"Saved {saved_count} validation reconstruction PNGs and {video_count} videos for epoch {epoch}"
        )

    # ------------------------------------------------------------------
    # Checkpointing (safetensors, compatible with load_adapted_vae)
    # ------------------------------------------------------------------

    def _build_checkpoint_payload(self, epoch: int, step: int) -> tuple[dict, dict]:
        """Trained weights + metadata for a checkpoint (read back by vae_adapter's loaders)."""
        vae = self._unwrap_vae()
        decoder = self._get_decoder()
        tensors = {
            f"decoder.{k}": v.detach().cpu().contiguous()
            for k, v in decoder.state_dict().items()
        }

        adapter_cfg = self._config.adapter
        if isinstance(vae, AdaptedVAE):
            tensors.update({
                f"in_adapter.{k}": v.detach().cpu().contiguous()
                for k, v in vae.in_adapter.state_dict().items()
            })
            tensors.update({
                f"out_adapter.{k}": v.detach().cpu().contiguous()
                for k, v in vae.out_adapter.state_dict().items()
            })
        if self._inflated:
            tensors.update({
                f"encoder_conv_in.{k}": v.detach().cpu().contiguous()
                for k, v in self._get_encoder_conv_in().state_dict().items()
            })
        for idx, module in zip(_UNFROZEN_DOWN_BLOCKS, self._get_encoder_downblock_modules()):
            tensors.update({
                f"encoder_down_block_{idx}.{k}": v.detach().cpu().contiguous()
                for k, v in module.state_dict().items()
            })
        for name, module in zip(
            ("down_blocks_last", "mid_block", "norm_out", "conv_out"),
            self._get_encoder_tail_modules(),
        ):
            tensors.update({
                f"encoder_tail.{name}.{k}": v.detach().cpu().contiguous()
                for k, v in module.state_dict().items()
            })

        metadata = {
            "format": "ltx-inflated-io-v1" if self._inflated else "ltx-decoder-plus-adapters-v1",
            "mode": adapter_cfg.mode,
            "inflate_init": "zeros",
            "channels": str(vae.channels if isinstance(vae, AdaptedVAE) else adapter_cfg.channels),
            "n": str(len(vae.channels) if isinstance(vae, AdaptedVAE) else len(adapter_cfg.channels)),
            "k": str(vae.k if isinstance(vae, AdaptedVAE) else 0),
            "activation": str(
                self._adapter_meta["activation"] if self._adapter_meta else adapter_cfg.activation
            ),
            "identity_init": str(
                self._adapter_meta["identity_init"] if self._adapter_meta else adapter_cfg.identity_init
            ),
            "default_temb": str(vae.default_temb if isinstance(vae, AdaptedVAE) else adapter_cfg.default_temb),
            "normalization": "clipped_zscore",
            "channel_mean": str(self._config.data.channel_mean),
            "channel_std": str(self._config.data.channel_std),
            "normalization_clip": str(self._config.data.normalization_clip),
            "epoch": str(epoch),
            "step": str(step),
        }
        return tensors, metadata

    def _save_checkpoint(self, epoch: int, step: int, *, improved: bool = True) -> Path:
        save_dir = Path(self._config.output_dir) / "checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)
        tensors, metadata = self._build_checkpoint_payload(epoch, step)

        ckpt_cfg = self._config.checkpoints
        if not ckpt_cfg.save_best_only:
            path = save_dir / f"vae_shockwave_epoch{epoch:03d}.safetensors"
            save_file(tensors, path, metadata=metadata)
            logger.info(f"VAE checkpoint saved: {path.relative_to(self._config.output_dir)}")
            self._save_training_state(self._state_file_for(path), epoch, step)
            # The last epoch is saved twice (its own save + the final one).
            if path not in self._checkpoint_paths:
                self._checkpoint_paths.append(path)
            self._prune_checkpoints()
            return path

        # save_best_only: two fixed slots instead of one pair per epoch.
        #   best -- weights only, replaced when the monitored metric improves.
        #   last -- weights + optimizer state, replaced every epoch (for resuming).
        if improved:
            best_path = save_dir / "vae_shockwave_best.safetensors"
            save_file(tensors, best_path, metadata={**metadata, "best_metric": ckpt_cfg.best_metric})
            self._best_ckpt_path = best_path
            logger.info(
                f"New best checkpoint (epoch {epoch}, {ckpt_cfg.best_metric}="
                f"{self._best_metric_value:.6f}): {best_path.relative_to(self._config.output_dir)}"
            )

        if ckpt_cfg.save_last_state:
            last_path = save_dir / "vae_shockwave_last.safetensors"
            save_file(tensors, last_path, metadata=metadata)
            self._save_training_state(self._state_file_for(last_path), epoch, step)

        # Downstream (DiT preprocessing, inference) should use the best weights.
        return self._best_ckpt_path or (save_dir / "vae_shockwave_last.safetensors")

    def _prune_checkpoints(self) -> None:
        """Delete all but the newest keep_last_n per-epoch checkpoints."""
        keep = self._config.checkpoints.keep_last_n
        if not (0 < keep < len(self._checkpoint_paths)):
            return
        for stale in self._checkpoint_paths[:-keep]:
            for f in (stale, self._state_file_for(stale)):
                try:
                    f.unlink(missing_ok=True)
                except OSError as e:  # a missing file is fine; anything else is worth seeing
                    logger.warning(f"Could not remove stale checkpoint {f}: {e}")
            logger.info(f"Pruned old checkpoint: {stale.name} (keep_last_n={keep})")
        self._checkpoint_paths = self._checkpoint_paths[-keep:]

    def _save_training_state(self, state_path: Path, epoch: int, step: int) -> Path:
        """Write optimizer/scheduler/RNG state as <stem>.state.pt next to the weights, for resume_from."""
        import random

        import numpy as np

        rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "xpu": torch.xpu.get_rng_state_all()
            if hasattr(torch, "xpu") and torch.xpu.is_available()
            else None,
        }
        state = {
            "epoch": epoch,
            "global_opt_step": step,
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "rng": rng,
            "best_metric_value": self._best_metric_value,
            "best_epoch": self._best_epoch,
        }
        torch.save(state, state_path)
        logger.info(
            f"Training state saved: {state_path.relative_to(self._config.output_dir)}"
        )
        return state_path

    def _prepare_output_dir(self) -> None:
        """Create the output dir, optionally wiping it first for a clean run."""
        out = Path(self._config.output_dir)
        if IS_MAIN_PROCESS:
            if self._config.clean_output_dir and out.exists():
                shutil.rmtree(out)
                logger.info(f"Cleaned output directory: {out}")
            out.mkdir(parents=True, exist_ok=True)
        self._accelerator.wait_for_everyone()

    # ------------------------------------------------------------------
    # Accelerator, wandb, config printing
    # ------------------------------------------------------------------

    def _setup_accelerator(self) -> None:
        self._accelerator = Accelerator(
            mixed_precision=self._config.acceleration.mixed_precision_mode,
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
            # No cross-rank RNG sync: every rank calls set_seed(cfg.seed) before
            # building the sampler, so the sync is redundant, and on Intel XPU
            # (oneCCL) it can corrupt the generator state in transit.
            rng_types=[],
        )
        if self._accelerator.num_processes > 1:
            logger.info(f"Distributed training: {self._accelerator.num_processes} processes")

    def _init_wandb(self) -> None:
        if not self._config.wandb.enabled or not IS_MAIN_PROCESS:
            self._wandb_run = None
            return
        self._wandb_run = wandb.init(
            project=self._config.wandb.project,
            entity=self._config.wandb.entity,
            name=Path(self._config.output_dir).name,
            tags=self._config.wandb.tags,
            config=self._config.model_dump(),
        )

    def _log_metrics(self, metrics: dict) -> None:
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)

    def _save_config(self) -> None:
        if not IS_MAIN_PROCESS:
            return
        config_path = Path(self._config.output_dir) / "training_config.yaml"
        with open(config_path, "w") as f:
            # mode="json" keeps enums as plain strings, so yaml.safe_load can read it back.
            yaml.dump(self._config.model_dump(mode="json"), f, default_flow_style=False, indent=2)
        logger.info(f"Config saved: {config_path}")

    @staticmethod
    def _print_config(config: BaseModel) -> None:
        if not IS_MAIN_PROCESS:
            return

        from rich.table import Table

        table = Table(title="VAE Finetuning Configuration", show_header=True, header_style="bold green")
        table.add_column("Parameter", style="bold white")
        table.add_column("Value", style="bold cyan")

        def flatten(cfg: BaseModel, prefix: str = "") -> list[tuple[str, str]]:
            rows = []
            for field, value in cfg:
                full = f"{prefix}.{field}" if prefix else field
                if isinstance(value, BaseModel):
                    rows.extend(flatten(value, full))
                else:
                    rows.append((full, str(value)[:70]))
            return rows

        for param, value in flatten(config):
            table.add_row(param, value)
        rich.print(table)

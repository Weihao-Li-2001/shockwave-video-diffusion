"""Score-distillation (SDS) loss from a frozen video DiT, for VAE fine-tuning.

The loss itself is described in windinet.config.SdsLossConfig. This module
loads the frozen transformer + scalar embedding and evaluates the loss;
VaeTrainer weights it into the objective.
"""

from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import Tensor

from windinet.config import SdsLossConfig
from windinet.inference.model_loader import load_transformer
from windinet.latent_utils import get_rope_scale_factors, pack_latents
from windinet.scalar_embeddings import ScalarEmbedding
from windinet.training.timestep_samplers import SAMPLERS
from windinet.utils import logger

# The CFD data has no fps; same fallback as LtxvTrainer._prepare_batch.
DEFAULT_FPS = 24.0


def _find_scalar_checkpoint(dit_checkpoint: Path) -> Path | None:
    """model_weights_<x>.safetensors -> scalar_embedding_<x>.safetensors (step or "best" slot)."""
    if "model_weights_" not in dit_checkpoint.name:
        return None
    scalar_path = dit_checkpoint.parent / dit_checkpoint.name.replace("model_weights_", "scalar_embedding_")
    return scalar_path if scalar_path.exists() else None


class SdsDistillationLoss:
    """Loads a frozen DiT + its scalar embedding once, then scores fresh VAE latents against it."""

    def __init__(self, config: SdsLossConfig, device: torch.device) -> None:
        self._config = config

        transformer = load_transformer(config.model_source, dtype=torch.float32)
        scalar_embedding = ScalarEmbedding(config.scalar_conditioning)

        if config.untrained_dit:
            # Stock transformer: it knows nothing about gamma, so the scalar
            # embedding stays randomly initialised.
            logger.info(
                "SDS distillation loss: untrained_dit=true -- using stock pretrained "
                f"{config.model_source} transformer + a freshly random-initialized "
                "ScalarEmbedding as the frozen critic"
            )
        else:
            dit_checkpoint = Path(config.dit_checkpoint)
            if not dit_checkpoint.is_file():
                raise FileNotFoundError(f"sds.dit_checkpoint not found: {dit_checkpoint}")
            transformer.load_state_dict(load_file(dit_checkpoint))

            scalar_checkpoint = _find_scalar_checkpoint(dit_checkpoint)
            if scalar_checkpoint is None:
                raise FileNotFoundError(
                    f"No sibling scalar_embedding_*.safetensors found next to {dit_checkpoint} -- "
                    "the SDS loss needs the exact ScalarEmbedding the frozen DiT was trained with "
                    "(set sds.untrained_dit=true instead if a freshly-initialized one is intended)."
                )
            scalar_embedding.load_state_dict(load_file(scalar_checkpoint))

            logger.info(
                f"SDS distillation loss: loaded frozen DiT from {dit_checkpoint} "
                f"(+ scalar embedding {scalar_checkpoint})"
            )

        transformer.requires_grad_(False)
        transformer.eval()
        self._transformer = transformer.to(device)

        scalar_embedding.requires_grad_(False)
        scalar_embedding.eval()
        self._scalar_embedding = scalar_embedding.to(device)

        sampler_cls = SAMPLERS[config.timestep_sampling_mode]
        self._timestep_sampler = sampler_cls(**config.timestep_sampling_params)

    @torch.no_grad()
    def _frozen_forward(
        self,
        z_detached: Tensor,
        num_frames: int,
        height: int,
        width: int,
        gamma: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """The gradient-free part: noise draw, conditioning and the frozen DiT forward.

        Returns (model_pred, conditioning_mask, noise); ``compute`` forms the
        target from the gradient-carrying latents.
        """
        batch_size, seq_length, _ = z_detached.shape
        sigmas = self._timestep_sampler.sample_for(z_detached)
        noise = torch.randn_like(z_detached)
        sigmas_bc = sigmas.view(-1, 1, 1)
        noisy = (1 - sigmas_bc) * z_detached + sigmas_bc * noise

        # Force first-frame conditioning (no Bernoulli draw) -- see
        # SdsLossConfig's docstring for why this stays on unconditionally.
        conditioning_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=z_detached.device)
        first_frame_end_idx = height * width
        if first_frame_end_idx < seq_length:
            conditioning_mask[:, :first_frame_end_idx] = True
        noisy = torch.where(conditioning_mask.unsqueeze(-1), z_detached, noisy)

        sampled_timestep_values = torch.round(sigmas * 1000.0).long()
        expanded_timesteps = sampled_timestep_values.unsqueeze(1).expand_as(conditioning_mask)
        timesteps = torch.where(conditioning_mask, 0, expanded_timesteps)

        gamma_in = gamma.to(device=z_detached.device, dtype=torch.float32).view(-1, 1)
        scalar_embeds = self._scalar_embedding(gamma_in)
        prompt_attention_mask = torch.ones(
            batch_size, scalar_embeds.shape[1], dtype=torch.bool, device=z_detached.device
        )
        rope_scale = get_rope_scale_factors(DEFAULT_FPS)

        # bf16, as the DiT was trained; forward only (no_grad).
        with torch.autocast(device_type=noisy.device.type, dtype=torch.bfloat16):
            model_pred = self._transformer(
                hidden_states=noisy,
                encoder_hidden_states=scalar_embeds,
                timestep=timesteps,
                encoder_attention_mask=prompt_attention_mask,
                num_frames=num_frames,
                height=height,
                width=width,
                rope_interpolation_scale=rope_scale,
                return_dict=False,
            )[0]

        return model_pred.float(), conditioning_mask, noise

    def compute(self, latents: Tensor, gamma: Tensor) -> Tensor:
        """SDS loss on this step's freshly-encoded VAE latents.

        Args:
            latents: normalised VAE latents z, [B, C, T, H, W], with gradient.
            gamma: raw physical scalar conditioning values, [B].

        Returns:
            Scalar loss.
        """
        _, _, num_frames, height, width = latents.shape
        z_packed = pack_latents(latents, spatial_patch_size=1, temporal_patch_size=1)

        model_pred, conditioning_mask, noise = self._frozen_forward(
            z_packed.detach(), num_frames, height, width, gamma
        )

        # The gradient reaches the encoder only through z in the target;
        # model_pred is a constant (no backprop through the frozen DiT).
        targets = noise - z_packed
        residual = (model_pred - targets).pow(2)
        loss_mask = (~conditioning_mask.unsqueeze(-1)).float()
        loss = residual.mul(loss_mask).div(loss_mask.mean().clamp_min(1e-8))
        return loss.mean()

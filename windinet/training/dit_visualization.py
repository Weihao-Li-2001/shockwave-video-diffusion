"""Periodic GT-vs-rollout panels and VRMSE during DiT training.

For a fixed set of held-out simulations, runs the full inference rollout
(frame 0 as the condition, as in scripts/inference_shockwave.py), decodes it
and saves static GT / prediction / residual panels. Each pass also appends
latent- and pixel-space VRMSE on those samples to
<output_dir>/dit_vrmse_metrics.csv (and redraws dit_vrmse_curve.png).

The decoder is the fine-tuned VAE that produced the training latents (recorded
in <preprocessed_data_root>/normalization.json), not the trainer's own stock
3-channel VAE. It is loaded once, kept on the CPU, and moved to the device
only while a pass runs.
"""

from __future__ import annotations

import csv
import json
import os
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/windinet-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.amp import autocast

from windinet.inference.model_loader import load_inflated_vae
from windinet.inference.pipeline import LTXConditionPipeline
from windinet.losses import vrms_loss
from windinet.training.shockwave_data import (
    CHANNEL_NAMES,
    ShockWaveDataset,
    normalize_fields,
    pad_frames_8n1,
)
from windinet.training.vae_visualization import (
    denormalize_fields,
    save_reconstruction_panels,
    save_reconstruction_video,
)
from windinet.utils import logger


def _vae_encode(vae, video: torch.Tensor) -> torch.Tensor:
    """Normalized video [B, C, F, H, W] -> normalised latents (as VaeTrainer._encode)."""
    out = vae.encode(video)
    posterior_mean = out.latent_dist.mean
    norm_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(posterior_mean.device, posterior_mean.dtype)
    norm_std = vae.latents_std.view(1, -1, 1, 1, 1).to(posterior_mean.device, posterior_mean.dtype)
    sf = float(getattr(vae.config, "scaling_factor", 1.0))
    return (posterior_mean - norm_mean) * sf / norm_std


def _vae_decode(vae, latents: torch.Tensor, default_temb: float) -> torch.Tensor:
    """Same formula as scripts/eval_dit_vrmse.py's vae_decode() / VaeTrainer._decode."""
    mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    std = vae.latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    sf = float(getattr(vae.config, "scaling_factor", 1.0))
    z = latents * std / sf + mean
    temb = torch.full((z.shape[0],), default_temb, device=z.device, dtype=z.dtype)
    return vae.decode(z, temb=temb, return_dict=True).sample


def _trim_latent_frames(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, C, F, H, W] pair -> both trimmed to the shorter F (same alignment
    issue as pixel-space frame counts: the DiT rollout's latent and the
    ground-truth encode's latent aren't guaranteed to land on the same count)."""
    n = min(a.shape[2], b.shape[2])
    return a[:, :, :n], b[:, :, :n]


def _append_vrmse_metrics(output_dir: str | Path, row: dict[str, float]) -> tuple[Path, Path]:
    """Append one row to <output_dir>/dit_vrmse_metrics.csv and redraw the curve.

    Existing rows are read back first, so a resumed run continues the curve.
    """
    metrics_path = Path(output_dir) / "dit_vrmse_metrics.csv"
    rows: list[dict[str, float]] = []
    if metrics_path.is_file():
        with metrics_path.open(newline="") as handle:
            rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(handle)]
    rows = [r for r in rows if r["step"] != row["step"]] + [row]
    rows.sort(key=lambda r: r["step"])

    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)

    steps = [r["step"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(steps, [r["latent_vrmse_mean"] for r in rows], marker="o", label="latent VRMSE")
    ax.plot(steps, [r["pixel_vrmse_mean"] for r in rows], marker="o", label="pixel VRMSE")
    ax.set(title="DiT rollout VRMSE (fixed visualization samples)", xlabel="Step", ylabel="VRMSE")
    ax.grid(alpha=0.3)
    ax.legend()
    curve_path = Path(output_dir) / "dit_vrmse_curve.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    return metrics_path, curve_path


def pick_fixed_visualization_sample_ids(
    preprocessed_data_root: str | Path, num_samples: int
) -> tuple[list[str], str]:
    """Deterministic, evenly spaced picks from the held-out val ids, plus the raw HDF5 path.

    Shared by DitVisualizer and scripts/inference_shockwave.py, so both render
    the same simulations for the same (preprocessed_data_root, num_samples).
    """
    manifest_path = Path(preprocessed_data_root) / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} not found -- re-run preprocess_dataset.py against this "
            "preprocessed_data_root first."
        )
    manifest = json.loads(manifest_path.read_text())
    val_ids = manifest.get("val_ids") or []
    if not val_ids:
        raise ValueError(f"{manifest_path} has no val_ids -- nothing to pick a fixed sample set from.")

    n = min(num_samples, len(val_ids))
    picks = sorted({round(i * (len(val_ids) - 1) / max(n - 1, 1)) for i in range(n)})
    picked_ids = [val_ids[i] for i in picks]
    return picked_ids, manifest["data_root"]


class DitVisualizer:
    """GT-vs-rollout panel renderer for a fixed sample set.

    Loading (manifest, HDF5, decode VAE) is deferred to the first ``run``.
    """

    def __init__(
        self,
        *,
        preprocessed_data_root: str,
        model_source,
        scalar_names: list[str],
        num_samples: int,
        frame_numbers: list[int],
        num_inference_steps: int,
        dpi: int,
        output_dir: str,
        seed: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        default_temb: float = 0.0,
        save_video: bool = False,
        video_fps: int = 8,
    ) -> None:
        self._preprocessed_data_root = Path(preprocessed_data_root)
        self._model_source = model_source
        self._scalar_names = scalar_names
        self._num_samples = num_samples
        self._frame_numbers = frame_numbers
        self._num_inference_steps = num_inference_steps
        self._dpi = dpi
        self._output_dir = output_dir
        self._seed = seed
        self._device = device
        self._dtype = dtype
        self._default_temb = default_temb
        self._save_video = save_video
        self._video_fps = video_fps

        self._samples: list[dict] | None = None  # lazy: fixed raw ShockWaveDataset rows
        self._stats: dict | None = None
        self._vae = None
        self._pipe: LTXConditionPipeline | None = None

    def _lazy_init(self) -> None:
        if self._samples is not None:
            return

        norm_path = self._preprocessed_data_root / "normalization.json"
        manifest_path = self._preprocessed_data_root / "split_manifest.json"
        if not norm_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"DiT visualization needs both {norm_path} and {manifest_path} "
                "(both written by preprocess_dataset.py) -- re-run preprocessing "
                "if either is missing, or set visualization.enabled: false."
            )

        payload = json.loads(norm_path.read_text())
        vae_checkpoint = payload.get("vae_checkpoint")
        if not vae_checkpoint:
            raise ValueError(f"{norm_path} has no vae_checkpoint recorded -- cannot decode DiT samples.")
        self._stats = {
            # Field order of the latents; files without it are the 4-channel layout.
            "channel_names": payload.get("channel_names") or CHANNEL_NAMES,
            "channel_mean": payload["channel_mean"],
            "channel_std": payload["channel_std"],
            "normalization_clip": payload["normalization_clip"],
        }
        # Latents of truncated sims (--num-sim-frames): roll out that many frames.
        num_sim_frames = payload.get("num_sim_frames")

        # Index-based picks (the manifest has no per-sample gamma), fixed for the run.
        picked_ids, raw_h5 = pick_fixed_visualization_sample_ids(
            self._preprocessed_data_root, self._num_samples
        )
        dataset = ShockWaveDataset(raw_h5, num_sim_frames=num_sim_frames)
        id_to_idx = {sid: i for i, sid in enumerate(dataset.ids)}
        missing = [sid for sid in picked_ids if sid not in id_to_idx]
        if missing:
            raise ValueError(f"Visualization sample ids {missing} not found in {raw_h5}")
        self._samples = [dataset[id_to_idx[sid]] for sid in picked_ids]
        logger.info(
            f"DiT visualization fixed on {len(self._samples)} samples from {raw_h5}: "
            f"{[s['id'] for s in self._samples]}"
        )

        logger.info(f"Loading DiT-visualization VAE from {vae_checkpoint}")
        # Kept on the CPU; run() moves it to the device only for its duration,
        # so it does not take training memory between passes.
        self._vae = load_inflated_vae(
            self._model_source, vae_checkpoint, dtype=self._dtype, device="cpu"
        )
        self._vae.requires_grad_(False)
        self._vae.eval()

    @torch.no_grad()
    def run(self, *, transformer, scalar_embedding, scheduler, step: int) -> None:
        """Roll out, decode and save panels + VRMSE for the fixed samples.

        Call on the main process only, with the unwrapped (non-DDP)
        ``transformer``/``scalar_embedding`` in eval mode.
        """
        self._lazy_init()
        self._vae.to(self._device)

        if self._pipe is None:
            self._pipe = LTXConditionPipeline(
                scheduler=deepcopy(scheduler),
                vae=self._vae,
                text_encoder=None,
                tokenizer=None,
                transformer=transformer,
            ).to(self._device)
            self._pipe.set_progress_bar_config(disable=True)
            # Fields are already in [-1, 1]; the video processor's default
            # normalisation would clip negative momentum (see make_pipe()).
            self._pipe.video_processor.register_to_config(do_normalize=False)
        else:
            self._pipe.transformer = transformer

        try:
            self._run_samples(scalar_embedding=scalar_embedding, step=step)
        finally:
            self._vae.to("cpu")
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
            elif self._device.type == "xpu":
                torch.xpu.empty_cache()

    def _run_samples(self, *, scalar_embedding, step: int) -> None:
        sum_latent_vrmse, sum_pixel_vrmse = 0.0, 0.0

        for i, sample in enumerate(self._samples):
            H, W = sample["density"].shape[-2:]
            channel_names = self._stats["channel_names"]
            gt = torch.stack([sample[name] for name in channel_names]).unsqueeze(0)  # [1, C, F, H, W]
            num_frames_needed = gt.shape[2]
            num_frames_padded = ((num_frames_needed - 1 + 7) // 8) * 8 + 1  # LTX VAE needs 8k+1 (97 -> 97, 101 -> 105)

            # Normalized ground truth: encoded for the latent-space metric and
            # used directly for the pixel-space metric (panels use physical `gt`).
            gt_norm = normalize_fields(
                gt, self._stats["channel_mean"], self._stats["channel_std"], self._stats["normalization_clip"],
            ).to(device=self._device, dtype=self._dtype)  # [1, C, F, H, W]
            # Frame 0 as the condition, in the pipeline's frame-major layout [1, 1, C, H, W].
            cond = gt_norm[:, :, 0].unsqueeze(1)

            scalar_values = [sample["meta"][name] for name in self._scalar_names]
            scalars = torch.tensor([scalar_values], device=self._device, dtype=self._dtype)
            prompt_embeds = scalar_embedding(scalars)
            prompt_mask = torch.ones(1, prompt_embeds.shape[1], device=self._device, dtype=torch.long)

            g = torch.Generator(device=self._device).manual_seed(self._seed + i)
            with autocast(self._device.type, dtype=self._dtype, enabled=(self._device.type in ("cuda", "xpu"))):
                out = self._pipe(
                    prompt=None, negative_prompt=None,
                    video=cond, frame_index=0, strength=1.0,
                    width=W, height=H,
                    num_frames=num_frames_padded,
                    num_inference_steps=self._num_inference_steps,
                    guidance_scale=1.0,
                    image_cond_noise_scale=0.0,
                    generator=g,
                    output_reference_comparison=False,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_mask,
                    negative_prompt_embeds=torch.zeros_like(prompt_embeds),
                    negative_prompt_attention_mask=prompt_mask.clone(),
                    output_type="latent",
                )
            # The pipeline returns latents (same space as _vae_encode); decoding
            # them here gives both the latent- and pixel-space error from one rollout.
            pred_latent = out.frames.float()

            # The VAE needs F = 8n+1; pad a copy for encoding, keep gt_norm unpadded.
            gt_latent = _vae_encode(self._vae, pad_frames_8n1(gt_norm)).float()
            pred_lat_trim, gt_lat_trim = _trim_latent_frames(pred_latent, gt_latent)
            sample_latent_vrmse = float(vrms_loss(pred_lat_trim, gt_lat_trim).item())
            sum_latent_vrmse += sample_latent_vrmse

            pred_norm = _vae_decode(self._vae, pred_lat_trim.to(self._dtype), self._default_temb).float()
            gt_norm_f32 = gt_norm.float()
            n_px = min(pred_norm.shape[2], gt_norm_f32.shape[2])
            sample_pixel_vrmse = float(
                vrms_loss(pred_norm[:, :, :n_px], gt_norm_f32[:, :, :n_px]).item()
            )
            sum_pixel_vrmse += sample_pixel_vrmse

            pred_physical = pred_norm.cpu()[:, :, :num_frames_needed]  # [1, C, F, H, W], trim the padding
            pred_physical = denormalize_fields(
                pred_physical, self._stats["channel_mean"], self._stats["channel_std"],
                self._stats["normalization_clip"],
            )

            save_reconstruction_panels(
                prediction=pred_physical[0],
                target=gt[0],
                sample_id=sample["id"],
                label=f"step_{step:06d}",
                frame_numbers=self._frame_numbers,
                channel_names=channel_names,
                output_dir=self._output_dir,
                dpi=self._dpi,
            )
            if self._save_video:
                save_reconstruction_video(
                    prediction=pred_physical[0],
                    target=gt[0],
                    sample_id=sample["id"],
                    label=f"step_{step:06d}",
                    channel_names=channel_names,
                    output_dir=self._output_dir,
                    fps=self._video_fps,
                )

            if self._device.type == "cuda":
                torch.cuda.empty_cache()
            elif self._device.type == "xpu":
                torch.xpu.empty_cache()

        n = len(self._samples)
        latent_vrmse_mean = sum_latent_vrmse / n
        pixel_vrmse_mean = sum_pixel_vrmse / n
        metrics_path, curve_path = _append_vrmse_metrics(
            self._output_dir,
            {"step": float(step), "latent_vrmse_mean": latent_vrmse_mean, "pixel_vrmse_mean": pixel_vrmse_mean},
        )

        logger.info(
            f"Saved DiT visualization panels for step {step} "
            f"({len(self._samples)} samples x {len(self._frame_numbers)} frames"
            f"{', + videos' if self._save_video else ''}); "
            f"latent_vrmse={latent_vrmse_mean:.5f} pixel_vrmse={pixel_vrmse_mean:.5f} "
            f"-> {metrics_path}, {curve_path}"
        )

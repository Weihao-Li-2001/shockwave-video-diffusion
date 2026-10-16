#!/usr/bin/env python
"""VAE-only reconstruction metrics on the held-out test file (test.h5).

Every VAE in RESULTS.md is compared on the same test file, which was never
part of any train/val split or tuning decision (each run's own val_vrmse was
measured on its own validation slice and also selected its checkpoint).

Reports, computed per simulation and then averaged (so independent of
--batch-size), with the same functions as VaeTrainer's validation:

  * vrmse_chmean   mean of the per-channel VRMSEs (the headline metric)
  * vrmse_<field>  per channel
  * vrmse          pooled over all channels

Per-sim values go into the JSON too, for paired comparisons between runs.
The VAE, its normalization and the encode/decode are rebuilt exactly as in
VaeTrainer from the run's own config (a model trained at 128x128 is applied to
the 256x256 data with its own 128x128 stats, which differ by < 0.2%).

Usage:
    python scripts/eval_vae_test.py configs/vae/baseline.yaml \\
        --checkpoint outputs/vae/baseline/checkpoints/vae_shockwave_best.safetensors \\
        --test-h5 data/euler_mq/256x256_ds/test.h5 \\
        --output outputs/vae/baseline/test_eval_256.json

--save-samples N also saves float16 fields (gt_raw, gt_norm, recon_norm) of N
gamma-spread sims, the same picks as scripts/eval_dit_vrmse.py, to
<output dir>/samples/ (~5 MB per sim). Single process, forward passes only.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import typer
import yaml
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader, Subset

from windinet.config import VaeTrainerConfig
from windinet.inference.model_loader import load_vae
from windinet.training.shockwave_data import ShockWaveDataset, build_shockwave_video, pick_gamma_spread_ids
from windinet.training.vae_trainer import vrmse, vrmse_per_channel
from windinet.utils import get_default_device
from windinet.vae_adapter import inflate_vae_io_channels, load_adapted_vae, load_inflated_vae_checkpoint

console = Console()
app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)

_AUTOCAST_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _build_vae(cfg: VaeTrainerConfig, checkpoint: str | None, device: torch.device):
    """Rebuild the VAE exactly as VaeTrainer._load_vae does, with `checkpoint` as its weights."""
    vae = load_vae(cfg.model.model_source, dtype=torch.float32)
    adapter_cfg = cfg.adapter
    if adapter_cfg.enabled and adapter_cfg.mode == "adapter":
        if checkpoint is None:
            raise typer.BadParameter("mode='adapter' needs --checkpoint (there is no untrained adapter reference)")
        vae, _ = load_adapted_vae(
            vae,
            ckpt_path=checkpoint,
            device="cpu",
            dtype=torch.float32,
            channels=adapter_cfg.channels,
            k=adapter_cfg.hidden_channels,
            activation=adapter_cfg.activation,
            identity_init=adapter_cfg.identity_init,
            default_temb=adapter_cfg.default_temb,
        )
    elif adapter_cfg.enabled and adapter_cfg.mode == "inflate":
        inflate_vae_io_channels(vae, n=len(adapter_cfg.channels))
        if checkpoint is not None:
            meta = load_inflated_vae_checkpoint(vae, ckpt_path=checkpoint, device="cpu", dtype=torch.float32)
            console.print(f"  loaded inflate checkpoint (epoch={meta.get('epoch', '?')}): {checkpoint}")
        else:
            console.print("  no checkpoint: freshly inflated pretrained VAE")
    else:
        raise typer.BadParameter(f"unsupported adapter config (enabled={adapter_cfg.enabled}, mode={adapter_cfg.mode!r})")
    for p in vae.parameters():
        p.requires_grad_(False)
    vae.eval()
    return vae.to(device)


def _encode(vae, video: torch.Tensor) -> torch.Tensor:
    """VaeTrainer._encode, latents only."""
    posterior_mean = vae.encode(video).latent_dist.mean
    norm_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(posterior_mean.device, posterior_mean.dtype)
    norm_std = vae.latents_std.view(1, -1, 1, 1, 1).to(posterior_mean.device, posterior_mean.dtype)
    sf = float(getattr(vae.config, "scaling_factor", 1.0))
    return (posterior_mean - norm_mean) * sf / norm_std


def _decode(vae, latents: torch.Tensor, default_temb: float) -> torch.Tensor:
    """VaeTrainer._decode."""
    mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    std = vae.latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    sf = float(getattr(vae.config, "scaling_factor", 1.0))
    z = latents * std / sf + mean
    temb = torch.full((z.shape[0],), default_temb, device=z.device, dtype=z.dtype)
    return vae.decode(z, temb=temb, return_dict=True).sample


@app.command()
def main(
    config_path: str = typer.Argument(..., help="The run's VaeTrainerConfig YAML (model/adapter/normalization are read from it)"),
    checkpoint: str = typer.Option(None, help="Finetuned VAE checkpoint; omit only for the freshly-inflated reference"),
    test_h5: str = typer.Option(..., help="Standalone test.h5, e.g. data/euler_mq/256x256_ds/test.h5"),
    num_samples: int = typer.Option(0, help="Evaluate only the first N test sims (0 = all)"),
    batch_size: int = typer.Option(4, help="Sims per forward pass (does not affect the metrics)"),
    output: str = typer.Option("vae_test_eval.json", help="Where to write the JSON report"),
    save_samples: int = typer.Option(3, help="Save fields of N gamma-spread test sims for figures (0 = none)"),
    sample_frames: list[int] = typer.Option([0, 25, 50, 75, 96], help="0-based frames saved per sample (repeat the option); 96 = last of 97"),
    metric_frames: int = typer.Option(
        None,
        help="Score only the first N frames. For comparing a 101-frame VAE against a 97-frame "
        "one on the same frames: the VAE still sees its full clip, only the metric is cropped.",
    ),
    samples_dir: str = typer.Option(
        None,
        help="Where the sample .npz files go (default: <output dir>/samples)",
    ),
) -> None:
    with open(config_path) as f:
        cfg = VaeTrainerConfig(**yaml.safe_load(f))
    device = get_default_device()
    channel_order = list(cfg.data.channel_order)
    console.print(f"Device: {device}  config: {config_path}")

    dataset = ShockWaveDataset(Path(test_h5), num_sim_frames=cfg.data.num_sim_frames)
    indices = list(range(len(dataset)))
    if num_samples > 0:
        indices = indices[:num_samples]
    console.print(f"Test set: {test_h5} -- evaluating {len(indices)} of {len(dataset)} sims")
    sample_ids = pick_gamma_spread_ids([dataset.ids[i] for i in indices], save_samples)
    if samples_dir is None:
        samples_dir = str(Path(output).parent / ("samples" + (f"_m{metric_frames}" if metric_frames else "")))
    if sample_ids:
        console.print(f"Saving figure samples {sorted(sample_ids)} (frames {sample_frames}) to {samples_dir}")
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.num_dataloader_workers,
    )

    vae = _build_vae(cfg, checkpoint, device)
    autocast_dtype = _AUTOCAST_DTYPES.get(cfg.acceleration.mixed_precision_mode or "no")

    per_sim: list[dict] = []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            orig_F = batch["density"].shape[1]
            x = build_shockwave_video(
                batch,
                device=device,
                channel_mean=cfg.data.channel_mean,
                channel_std=cfg.data.channel_std,
                normalization_clip=cfg.data.normalization_clip,
                channel_order=channel_order,
            )
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                recon = _decode(vae, _encode(vae, x), cfg.adapter.default_temb)
            score_F = min(orig_F, metric_frames or orig_F)
            recon = recon.float()[:, :, :score_F]
            target = x[:, :, :score_F]
            ids = batch.get("id", [None] * recon.shape[0])
            for i in range(recon.shape[0]):
                pred_i, target_i = recon[i : i + 1], target[i : i + 1]
                per_channel = vrmse_per_channel(pred_i, target_i)
                row = {
                    "id": ids[i],
                    "vrmse": vrmse(pred_i, target_i),
                    "vrmse_chmean": float(per_channel.mean().item()),
                }
                row.update({f"vrmse_{name}": float(v) for name, v in zip(channel_order, per_channel.tolist())})
                per_sim.append(row)
                if ids[i] in sample_ids:
                    frames = [f for f in sample_frames if f < score_F]
                    gt_raw = torch.stack([batch[name][i] for name in channel_order])  # [C, F, H, W]
                    Path(samples_dir).mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        Path(samples_dir) / f"{ids[i]}.npz",
                        frames=np.array(frames),
                        channel_order=np.array(channel_order),
                        gamma=float(batch["meta"]["gamma"][i]),
                        gt_raw=gt_raw[:, frames].numpy().astype(np.float16),
                        gt_norm=target[i][:, frames].cpu().numpy().astype(np.float16),
                        recon_norm=recon[i][:, frames].cpu().numpy().astype(np.float16),
                        channel_mean=np.array(cfg.data.channel_mean),
                        channel_std=np.array(cfg.data.channel_std),
                        normalization_clip=cfg.data.normalization_clip,
                        vrmse_chmean=row["vrmse_chmean"],
                    )
            console.print(f"  {len(per_sim)}/{len(indices)} sims  ({time.time() - t0:.0f}s)")

    keys = ["vrmse", "vrmse_chmean"] + [f"vrmse_{name}" for name in channel_order]
    summary = {}
    for key in keys:
        values = torch.tensor([row[key] for row in per_sim], dtype=torch.float64)
        summary[key] = values.mean().item()
        summary[f"{key}_sem"] = (values.std(unbiased=True) / len(values) ** 0.5).item() if len(values) > 1 else 0.0

    report = {
        "config": config_path,
        "checkpoint": checkpoint,
        "test_h5": test_h5,
        "num_sims": len(per_sim),
        "num_sim_frames": cfg.data.num_sim_frames,
        "metric_frames": metric_frames,
        "mixed_precision": cfg.acceleration.mixed_precision_mode,
        "normalization_stats_file": str(cfg.data.normalization_stats_file),
        "channel_order": channel_order,
        "figure_samples": {"ids": sorted(sample_ids), "frames": sample_frames, "dir": samples_dir},
        "metrics": summary,
        "per_sim": per_sim,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2))

    table = Table(title=f"Test-set reconstruction ({len(per_sim)} sims)")
    table.add_column("metric")
    table.add_column("mean", justify="right")
    table.add_column("sem", justify="right")
    for key in keys:
        table.add_row(key, f"{summary[key]:.4f}", f"{summary[key + '_sem']:.4f}")
    console.print(table)
    console.print(f"Report written to {output}")


if __name__ == "__main__":
    app()

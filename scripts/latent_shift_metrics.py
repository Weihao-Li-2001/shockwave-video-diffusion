#!/usr/bin/env python
"""Paired latent-shift metrics: pretrained vs. fine-tuned encoder on the same inputs.

Complements scripts/latent_stats.py (one encoder vs. LTX-Video's reference
statistics) by pairing z_before (inflated, not fine-tuned encoder; with the
zero-initialised new channel this equals the pretrained encoder) with z_after
(fine-tuned checkpoint) sample by sample on test.h5:

  channel_drift            per-channel mean/std drift
  wasserstein              per-channel Wasserstein-1 distance (shape changes)
  correlation              mean |off-diagonal| correlation between the latent
                           channels, pooled over all samples
  per_sample_channel_corr  the same within each simulation, averaged (pooled
                           correlations can hide per-sample structure), with
                           before/after/diff heatmaps
  displacement_affine      median relative displacement, and how much of it
                           one global affine map explains (low residual: a
                           benign global transform; high: a new nonlinear code)
  cka                      linear CKA between z_before and z_after

Usage:
    python scripts/latent_shift_metrics.py configs/vae/baseline.yaml \\
        --checkpoint outputs/vae/baseline/checkpoints/vae_shockwave_best.safetensors \\
        --test-h5 data/euler_mq/256x256_ds/test.h5 \\
        --num-samples 64 --output outputs/latent_shift_baseline.json

Single process, encode only. The displacement/CKA steps hold every encoded
voxel twice in float64; --max-positions subsamples positions for those two.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/windinet-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import typer
import yaml
from matplotlib.colors import LinearSegmentedColormap
from rich.console import Console
from rich.table import Table
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, Subset

from windinet.config import VaeTrainerConfig
from windinet.inference.model_loader import load_vae
from windinet.training.shockwave_data import ShockWaveDataset, build_shockwave_video
from windinet.utils import get_default_device
from windinet.vae_adapter import inflate_vae_io_channels, load_inflated_vae_checkpoint

console = Console()
app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


def _build_vae(cfg: VaeTrainerConfig, checkpoint: str | None, device: torch.device):
    """Fresh-inflate the pretrained VAE, optionally loading a finetuned checkpoint on top."""
    vae = load_vae(cfg.model.model_source, dtype=torch.float32)
    adapter_cfg = cfg.adapter
    n = len(adapter_cfg.channels)
    inflate_vae_io_channels(vae, n=n)
    if checkpoint is not None:
        meta = load_inflated_vae_checkpoint(vae, ckpt_path=checkpoint, device="cpu", dtype=torch.float32)
        console.print(f"  loaded finetuned checkpoint (epoch={meta.get('epoch', '?')}): {checkpoint}")
    for p in vae.parameters():
        p.requires_grad_(False)
    vae.eval()
    return vae.to(device)


@torch.no_grad()
def _encode_rescaled(vae, loader: DataLoader, cfg: VaeTrainerConfig, device: torch.device) -> torch.Tensor:
    """Normalised posterior-mean latents (as VaeTrainer._encode) of the whole loader, [N, C, T', H', W'] on CPU."""
    sf = float(getattr(vae.config, "scaling_factor", 1.0))
    latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(device, torch.float32)
    latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(device, torch.float32)

    chunks = []
    for batch in loader:
        video = build_shockwave_video(
            batch,
            device=device,
            channel_mean=cfg.data.channel_mean,
            channel_std=cfg.data.channel_std,
            normalization_clip=cfg.data.normalization_clip,
            channel_order=cfg.data.channel_order,
        )
        posterior_mean = vae.encode(video).latent_dist.mean.float()
        rescaled = (posterior_mean - latents_mean) * sf / latents_std
        chunks.append(rescaled.cpu())
    return torch.cat(chunks, dim=0)


def _chan(z: torch.Tensor) -> torch.Tensor:
    """[N, C, ...] -> [C, M], channel axis first, everything else flattened."""
    c = z.shape[1]
    return z.movedim(1, 0).reshape(c, -1).double()


def _pos(z: torch.Tensor) -> torch.Tensor:
    """[N, C, ...] -> [M, C], each sample/spatiotemporal position as a C-d row."""
    c = z.shape[1]
    return z.movedim(1, -1).reshape(-1, c).double()


def _step3_channel_drift(z_before: torch.Tensor, z_after: torch.Tensor) -> dict:
    cb, ca = _chan(z_before), _chan(z_after)
    mb, sb = cb.mean(1), cb.std(1)
    ma, sa = ca.mean(1), ca.std(1)
    mean_drift = (ma - mb).abs()
    std_ratio = sa / sb.clamp_min(1e-8)
    return {
        "before": {"mean": mb.tolist(), "std": sb.tolist()},
        "after": {"mean": ma.tolist(), "std": sa.tolist()},
        "mean_drift_abs_avg": mean_drift.mean().item(),
        "mean_drift_abs_max": mean_drift.max().item(),
        "std_ratio_avg": std_ratio.mean().item(),
        "std_ratio_min": std_ratio.min().item(),
        "std_ratio_max": std_ratio.max().item(),
    }


def _step4_wasserstein(z_before: torch.Tensor, z_after: torch.Tensor, *, subsample: int, seed: int) -> dict:
    cb, ca = _chan(z_before), _chan(z_after)
    m = cb.shape[1]
    if m > subsample:
        idx = torch.randperm(m, generator=torch.Generator().manual_seed(seed))[:subsample]
        cb, ca = cb[:, idx], ca[:, idx]
    w1 = [wasserstein_distance(cb[c].numpy(), ca[c].numpy()) for c in range(cb.shape[0])]
    order = sorted(range(len(w1)), key=lambda i: -w1[i])
    return {
        "per_channel": w1,
        "top5_channels": [[i, round(w1[i], 6)] for i in order[:5]],
        "avg": sum(w1) / len(w1),
        "num_voxels_used": cb.shape[1],
    }


def _offdiag_corr(z: torch.Tensor) -> float:
    zc = _chan(z)
    zc = zc - zc.mean(1, keepdim=True)
    corr = (zc @ zc.T) / zc.shape[1]
    d = torch.sqrt(torch.diag(corr).clamp_min(1e-8))
    corr = corr / (d[:, None] * d[None, :])
    off = corr - torch.diag(torch.diag(corr))
    return off.abs().mean().item()


def _step5_correlation(z_before: torch.Tensor, z_after: torch.Tensor) -> dict:
    return {
        "offdiag_corr_before": _offdiag_corr(z_before),
        "offdiag_corr_after": _offdiag_corr(z_after),
    }


def _mean_pairwise_corr_matrix(z: torch.Tensor) -> torch.Tensor:
    """CxC channel correlation computed within each sample of z [N, C, T, H, W], then averaged."""
    n, c = z.shape[0], z.shape[1]
    mats = torch.empty(n, c, c, dtype=torch.float64)
    for i in range(n):
        zi = z[i].reshape(c, -1).double()
        zi = zi - zi.mean(1, keepdim=True)
        cov = (zi @ zi.T) / zi.shape[1]
        d = torch.sqrt(torch.diag(cov).clamp_min(1e-12))
        mats[i] = cov / (d[:, None] * d[None, :])
    return mats.mean(0)


# dataviz skill's validated diverging pair (blue <-> red, neutral gray midpoint) --
# blue = positive correlation, red = negative, gray = ~0 ("decorrelated" target).
_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "windinet_diverging_corr", ["#e34948", "#f0efec", "#2a78d6"], N=256
)


def _save_corr_heatmap(matrix: torch.Tensor, title: str, path: Path, vlim: float | None = None) -> None:
    """Heatmap of a correlation matrix: fixed [-1, 1] scale (vlim=None) or +/-vlim (for diffs)."""
    v = 1.0 if vlim is None else vlim
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix.numpy(), cmap=_DIVERGING_CMAP, vmin=-v, vmax=v)
    ax.set_title(title)
    ax.set_xlabel("latent channel")
    ax.set_ylabel("latent channel")
    fig.colorbar(im, ax=ax, label="Pearson r" if vlim is None else "Pearson r shift (after - before)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _step6_displacement_and_affine(X: torch.Tensor, Y: torch.Tensor) -> dict:
    rel = (Y - X).norm(dim=1) / (X.norm(dim=1) + 1e-6)
    Xa = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype)], dim=1)
    sol = torch.linalg.lstsq(Xa, Y).solution
    resid = (Y - Xa @ sol).norm() / Y.norm()
    return {
        "median_relative_displacement": rel.median().item(),
        "mean_relative_displacement": rel.mean().item(),
        "affine_fit_residual": resid.item(),
    }


def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    return ((X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm())).item()


@app.command()
def main(
    config_path: str = typer.Argument(..., help="VaeTrainerConfig-shaped YAML (an inflate-mode VAE config)"),
    checkpoint: str = typer.Option(..., help="Finetuned checkpoint (e.g. vae_shockwave_best.safetensors) -- this is E_after / z_after"),
    test_h5: str = typer.Option(..., help="Standalone test.h5 (sibling of train.h5), never touched by any train/eval split or tuning decision"),
    num_samples: int = typer.Option(64, help="How many sims to encode from test_h5"),
    batch_size: int = typer.Option(4, help="Sims per encode() call"),
    max_positions: int = typer.Option(
        200_000,
        help="Subsample this many spatiotemporal positions for steps 6-7 (displacement/affine/CKA), "
        "which scale with voxel count rather than sim count. Steps 3-5 always use every encoded voxel.",
    ),
    w1_subsample: int = typer.Option(
        20_000, help="Subsample this many voxels per channel for the Wasserstein-1 distance (step 4)."
    ),
    seed: int = typer.Option(0, help="Seed for the subsampling above (reproducible across runs)"),
    output: str = typer.Option("latent_shift_metrics.json", help="Where to write the full JSON report"),
    save_corr_matrix: bool = typer.Option(
        True, help="Include the full CxC per-sample-averaged correlation matrices in the JSON report, not just the heatmap PNGs."
    ),
) -> None:
    with open(config_path) as f:
        cfg = VaeTrainerConfig(**yaml.safe_load(f))

    device = get_default_device()
    console.print(f"Device: {device}")

    full_dataset = ShockWaveDataset(Path(test_h5), num_sim_frames=cfg.data.num_sim_frames)
    chosen = list(range(len(full_dataset)))[:num_samples]
    console.print(f"Encoding {len(chosen)} sims from standalone test set '{test_h5}' (of {len(full_dataset)} total)")

    loader = DataLoader(
        Subset(full_dataset, chosen),
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.num_dataloader_workers,
    )

    console.print("\n[bold]z_before (zero-init, freshly-inflated, no finetuning == pretrained latent)[/bold]")
    vae_before = _build_vae(cfg, checkpoint=None, device=device)
    z_before = _encode_rescaled(vae_before, loader, cfg, device)
    del vae_before
    if device.type != "cpu":
        getattr(torch, device.type).empty_cache()

    console.print("\n[bold]z_after (finetuned checkpoint)[/bold]")
    vae_after = _build_vae(cfg, checkpoint=checkpoint, device=device)
    z_after = _encode_rescaled(vae_after, loader, cfg, device)
    del vae_after
    if device.type != "cpu":
        getattr(torch, device.type).empty_cache()

    console.print(f"z_before shape: {tuple(z_before.shape)}  z_after shape: {tuple(z_after.shape)}")

    step3 = _step3_channel_drift(z_before, z_after)
    step4 = _step4_wasserstein(z_before, z_after, subsample=w1_subsample, seed=seed)
    step5 = _step5_correlation(z_before, z_after)

    Xb, Xa = _pos(z_before), _pos(z_after)
    if Xb.shape[0] > max_positions:
        idx = torch.randperm(Xb.shape[0], generator=torch.Generator().manual_seed(seed))[:max_positions]
        Xb, Xa = Xb[idx], Xa[idx]
    step6 = _step6_displacement_and_affine(Xb, Xa)
    step7 = {"linear_cka": _linear_cka(Xb, Xa)}

    console.print("\n[bold]Per-sample channel correlation[/bold]")
    corr_before = _mean_pairwise_corr_matrix(z_before)
    corr_after = _mean_pairwise_corr_matrix(z_after)
    corr_diff = corr_after - corr_before
    c = corr_after.shape[0]
    offdiag_mask = ~torch.eye(c, dtype=torch.bool)
    step5b = {
        "mean_abs_offdiag_before": corr_before[offdiag_mask].abs().mean().item(),
        "mean_abs_offdiag_after": corr_after[offdiag_mask].abs().mean().item(),
        "max_abs_offdiag_after": corr_after[offdiag_mask].abs().max().item(),
    }
    if save_corr_matrix:
        step5b["matrix_before"] = corr_before.tolist()
        step5b["matrix_after"] = corr_after.tolist()

    output_path = Path(output)
    heatmap_paths = {
        "before": output_path.with_name(f"{output_path.stem}_channel_corr_before.png"),
        "after": output_path.with_name(f"{output_path.stem}_channel_corr_after.png"),
        "diff": output_path.with_name(f"{output_path.stem}_channel_corr_diff.png"),
    }
    # Diffs are small, so scale that plot to its own range (floored to avoid vmin == vmax).
    diff_vlim = max(corr_diff[offdiag_mask].abs().max().item(), 1e-3)
    _save_corr_heatmap(corr_before, "Mean per-sample channel correlation -- before (pretrained)", heatmap_paths["before"])
    _save_corr_heatmap(corr_after, "Mean per-sample channel correlation -- after (finetuned)", heatmap_paths["after"])
    _save_corr_heatmap(
        corr_diff, "Channel correlation shift (after - before)", heatmap_paths["diff"], vlim=diff_vlim
    )
    step5b["heatmaps"] = {k: str(v) for k, v in heatmap_paths.items()}
    console.print(f"  heatmaps written: {', '.join(str(p) for p in heatmap_paths.values())}")

    report = {
        "config": config_path,
        "checkpoint": checkpoint,
        "test_h5": test_h5,
        "num_sims": len(chosen),
        "sim_indices": chosen,
        "latent_shape": list(z_before.shape),
        "num_positions_used_steps_6_7": Xb.shape[0],
        "step3_channel_drift": step3,
        "step4_wasserstein": step4,
        "step5_correlation": step5,
        "step5b_per_sample_channel_corr": step5b,
        "step6_displacement_affine": step6,
        "step7_cka": step7,
    }
    Path(output).write_text(json.dumps(report, indent=2))
    console.print(f"\nFull report written to {output}")

    summary = Table(title="Latent shift summary (zero-init z_before vs finetuned z_after, on test.h5)")
    summary.add_column("metric")
    summary.add_column("value", justify="right")
    summary.add_row("mean drift |avg|", f"{step3['mean_drift_abs_avg']:.4f}")
    summary.add_row("mean drift |max|", f"{step3['mean_drift_abs_max']:.4f}")
    summary.add_row("std ratio avg", f"{step3['std_ratio_avg']:.4f}")
    summary.add_row("std ratio range", f"[{step3['std_ratio_min']:.4f}, {step3['std_ratio_max']:.4f}]")
    summary.add_row("Wasserstein-1 avg", f"{step4['avg']:.4f}")
    summary.add_row("off-diag corr before (pooled)", f"{step5['offdiag_corr_before']:.4f}")
    summary.add_row("off-diag corr after (pooled)", f"{step5['offdiag_corr_after']:.4f}")
    summary.add_row("off-diag corr before (per-sample avg)", f"{step5b['mean_abs_offdiag_before']:.4f}")
    summary.add_row("off-diag corr after (per-sample avg)", f"{step5b['mean_abs_offdiag_after']:.4f}")
    summary.add_row("off-diag corr after (per-sample max)", f"{step5b['max_abs_offdiag_after']:.4f}")
    summary.add_row("median relative displacement", f"{step6['median_relative_displacement']:.4f}")
    summary.add_row("affine-fit residual", f"{step6['affine_fit_residual']:.4f}")
    summary.add_row("linear CKA", f"{step7['linear_cka']:.4f}")
    console.print(summary)
    console.print(f"Top-5 channels by Wasserstein-1: {step4['top5_channels']}")


if __name__ == "__main__":
    app()

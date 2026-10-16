#!/usr/bin/env python
"""Field figures (as in assets/) from an eval sample npz file. CPU only.

The input is a per-sim .npz written by scripts/eval_dit_vrmse.py
(--save_npz_samples or --sim_ids): gt_raw, gt_norm, vae_only_norm and
vae_dit_norm at frames 0/25/50/75/100.

Subcommands, one per figure:
  dataset        the four raw fields of one sim at several frames
  vae-dit-frame  one frame: GT | VAE only | VAE+DiT | the two residuals

Styling matches the training-time panels
(windinet.training.vae_visualization.save_reconstruction_panels): viridis for
every field with the row's min/max as limits, signed residuals (Pred - GT) in
coolwarm with symmetric limits, default matplotlib look, image origin top-left.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)

CHANNELS = ["density", "momentum_x", "momentum_y", "pressure"]
CHANNEL_LABEL = {"density": r"$\rho$", "momentum_x": r"$m_x$", "momentum_y": r"$m_y$", "pressure": r"$p$"}


def _denorm(x: np.ndarray, mean: np.ndarray, std: np.ndarray, clip: float) -> np.ndarray:
    """Normalized [-1,1] fields [C, F, H, W] back to physical units (inverse of build_shockwave_video)."""
    return x.astype(np.float32) * (std * clip)[:, None, None, None] + mean[:, None, None, None]


def _show(fig, ax, img, cmap, vmin, vmax, colorbar=True):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)  # 3000 px wide at 20 in: sharp at print width, small PNG
    plt.close(fig)
    print(f"wrote {out}")


@app.command("vae-dit-frame")
def vae_dit_frame(
    npz: Path = typer.Option(..., help="Baseline DiT eval sample, figure_data/dit/<arm>/<sim>.npz"),
    frame: int = typer.Option(100, help="0-indexed frame (must be one of the saved frames)"),
    out: Path = typer.Option(Path("figures/vae_dit_frame100.png")),
) -> None:
    """Rows = channels. Columns = GT, VAE only, VAE+DiT, residual of each (Pred - GT).

    The three field panels of a row share limits, and so do its two residual
    panels, so the VAE's share of the rollout error can be read off directly.
    """
    d = np.load(npz)
    fi = list(d["frames"]).index(frame)
    mean, std, clip = d["channel_mean"], d["channel_std"], float(d["normalization_clip"])
    gt = _denorm(d["gt_norm"], mean, std, clip)[:, fi]
    vae = _denorm(d["vae_only_norm"], mean, std, clip)[:, fi]
    dit = _denorm(d["vae_dit_norm"], mean, std, clip)[:, fi]
    titles = ("GT", "VAE only", "VAE + DiT", "Residual (VAE only - GT)", "Residual (VAE + DiT - GT)")
    fig, axes = plt.subplots(4, 5, figsize=(20, 13), constrained_layout=True)
    for c, name in enumerate(CHANNELS):
        vmin = float(min(gt[c].min(), vae[c].min(), dit[c].min()))
        vmax = float(max(gt[c].max(), vae[c].max(), dit[c].max()))
        res = (vae[c] - gt[c], dit[c] - gt[c])
        lim = max(float(np.abs(res[0]).max()), float(np.abs(res[1]).max()), 1e-12)
        for k, img in enumerate((gt[c], vae[c], dit[c])):
            _show(fig, axes[c, k], img, "viridis", vmin, vmax)
        for k, r in enumerate(res):
            _show(fig, axes[c, 3 + k], r, "coolwarm", -lim, lim)
        axes[c, 0].set_ylabel(CHANNEL_LABEL[name], fontsize=16)
        for k, title in enumerate(titles):
            axes[c, k].set_title(title)
    # no suptitle: sim id and numbers belong in the LaTeX caption
    print(f"caption: sim {npz.stem}, gamma = {float(d['gamma']):.3f}, frame {frame}; "
          f"whole-sim VRMSE (channel mean): VAE only {float(d['vae_only_vrmse_chmean']):.4f}, "
          f"VAE + DiT {float(d['vae_dit_vrmse_chmean']):.4f}")
    _save(fig, out)


@app.command("dataset")
def dataset(
    npz: Path = typer.Option(..., help="Any figure sample npz (uses its raw, unclipped gt_raw)"),
    frames: list[int] = typer.Option([0, 50, 100]),
    out: Path = typer.Option(Path("figures/ch3_euler_mq_sample.png")),
) -> None:
    """Rows = the four raw fields, columns = frames. Physical units, one viridis scale per row."""
    d = np.load(npz)
    saved = list(d["frames"])
    raw = d["gt_raw"].astype(np.float32)
    idx = [saved.index(f) for f in frames]
    n = len(frames)
    fig, axes = plt.subplots(4, n, figsize=(4 * n + 1, 13), constrained_layout=True, squeeze=False)
    for c, name in enumerate(CHANNELS):
        vmin, vmax = float(raw[c, idx].min()), float(raw[c, idx].max())
        for k, fi in enumerate(idx):
            im = _show(fig, axes[c, k], raw[c, fi], "viridis", vmin, vmax, colorbar=False)
            axes[c, k].set_title(f"t = {frames[k]}")
        fig.colorbar(im, ax=list(axes[c, :]), fraction=0.046, pad=0.02)
        axes[c, 0].set_ylabel(CHANNEL_LABEL[name], fontsize=16)
    print(f"caption: sim {npz.stem}, gamma = {float(d['gamma']):.3f}")  # no suptitle, see vae-dit-frame
    _save(fig, out)


if __name__ == "__main__":
    app()

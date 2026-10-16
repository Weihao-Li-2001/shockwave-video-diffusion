"""PNG reconstruction panels, MP4 comparison videos and epoch-level loss curves."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/windinet-matplotlib")
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def denormalize_fields(
    tensor: torch.Tensor,
    channel_mean: list[float],
    channel_std: list[float],
    normalization_clip: float,
) -> torch.Tensor:
    """Invert build_shockwave_video's clipped z-score: normalized [B,C,F,H,W] -> physical units."""
    num_channels = len(channel_mean)
    mean = tensor.new_tensor(channel_mean).view(1, num_channels, 1, 1, 1)
    scale = tensor.new_tensor(channel_std).view(1, num_channels, 1, 1, 1) * normalization_clip
    out = tensor * scale + mean
    return out


def save_reconstruction_panels(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_id: str,
    label: str,
    frame_numbers: list[int],
    channel_names: list[str],
    output_dir: str | Path,
    dpi: int,
) -> list[Path]:
    """Save one four-channel GT/prediction/residual panel per requested frame.

    `label` names this pass in both the save path and the figure title (e.g.
    ``f"epoch_{epoch:04d}"`` for the epoch-based VAE trainer,
    ``f"step_{step:06d}"`` for the step-based DiT trainer) -- shared verbatim
    across both callers rather than assuming either unit.
    """
    prediction = prediction.detach().float().cpu()
    target = target.detach().float().cpu()
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
    save_dir = Path(output_dir) / "visualizations" / label / safe_id
    save_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    num_frames = target.shape[1]
    for frame_number in frame_numbers:
        frame_index = frame_number - 1
        if frame_index >= num_frames:
            continue

        gt = target[:, frame_index].numpy()
        pred = prediction[:, frame_index].numpy()
        residual = pred - gt
        frame_rmse = float(np.sqrt(np.mean(residual**2)))
        channel_rmse = np.sqrt(np.mean(residual**2, axis=(1, 2)))

        fig, axes = plt.subplots(4, 3, figsize=(12, 13), constrained_layout=True)
        for channel, name in enumerate(channel_names):
            value_min = float(min(gt[channel].min(), pred[channel].min()))
            value_max = float(max(gt[channel].max(), pred[channel].max()))
            residual_limit = max(float(np.abs(residual[channel]).max()), 1e-12)

            images = (
                axes[channel, 0].imshow(gt[channel], cmap="viridis", vmin=value_min, vmax=value_max),
                axes[channel, 1].imshow(pred[channel], cmap="viridis", vmin=value_min, vmax=value_max),
                axes[channel, 2].imshow(
                    residual[channel], cmap="coolwarm", vmin=-residual_limit, vmax=residual_limit
                ),
            )
            axes[channel, 0].set_ylabel(f"{name}\nRMSE={channel_rmse[channel]:.4e}")
            for column, title in enumerate(("GT", "Prediction", "Residual (Pred-GT)")):
                axes[channel, column].set_title(title)
                axes[channel, column].set_xticks([])
                axes[channel, column].set_yticks([])
                fig.colorbar(images[column], ax=axes[channel, column], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"{label}  sample={sample_id}  frame={frame_number}  RMSE={frame_rmse:.4e}",
            fontsize=13,
        )
        path = save_dir / f"frame_{frame_number:04d}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        saved.append(path)

    return saved


def write_comparison_video(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    channel_names: list[str],
    out_path: str | Path,
    title: str,
    fps: int = 8,
    dpi: int = 100,
) -> Path:
    """Write one MP4 of [C,F,H,W] fields: a row per channel, GT / Prediction /
    Residual (Pred-GT) columns, one video frame per simulation frame.

    Color limits are fixed per channel over the whole sequence, so brightness
    changes show the field evolving, not a rescaled axis.
    """
    num_frames = min(prediction.shape[1], target.shape[1])
    prediction, target = prediction[:, :num_frames], target[:, :num_frames]
    residual = prediction - target

    num_channels = len(channel_names)
    fig, axes = plt.subplots(num_channels, 3, figsize=(12, 3.25 * num_channels), dpi=dpi, constrained_layout=True)
    images = []
    for channel, name in enumerate(channel_names):
        value_min = float(min(target[channel].min(), prediction[channel].min()))
        value_max = float(max(target[channel].max(), prediction[channel].max()))
        residual_limit = max(float(np.abs(residual[channel]).max()), 1e-12)
        row = []
        for column, column_title in enumerate(("GT", "Prediction", "Residual (Pred-GT)")):
            if column == 2:
                cmap, vmin, vmax = "coolwarm", -residual_limit, residual_limit
            else:
                cmap, vmin, vmax = "viridis", value_min, value_max
            image = axes[channel, column].imshow(target[channel, 0], cmap=cmap, vmin=vmin, vmax=vmax)
            axes[channel, column].set_xticks([])
            axes[channel, column].set_yticks([])
            if channel == 0:
                axes[channel, column].set_title(column_title)
            fig.colorbar(image, ax=axes[channel, column], fraction=0.046, pad=0.04)
            row.append(image)
        axes[channel, 0].set_ylabel(name)
        images.append(row)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path, fps=fps) as writer:
        for t in range(num_frames):
            for channel in range(num_channels):
                images[channel][0].set_data(target[channel, t])
                images[channel][1].set_data(prediction[channel, t])
                images[channel][2].set_data(residual[channel, t])
            frame_rmse = float(np.sqrt(np.mean(residual[:, t] ** 2)))
            fig.suptitle(f"{title}  frame={t + 1}/{num_frames}  RMSE={frame_rmse:.4e}", fontsize=13)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            # Pad (white) to a multiple of 16 px so ffmpeg doesn't rescale the frame.
            pad_h, pad_w = -frame.shape[0] % 16, -frame.shape[1] % 16
            frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=255)
            writer.append_data(frame)
    plt.close(fig)
    return out_path


def save_reconstruction_video(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_id: str,
    label: str,
    channel_names: list[str],
    output_dir: str | Path,
    fps: int = 8,
    dpi: int = 100,
) -> Path:
    """Save a four-channel GT/prediction/residual MP4 of the whole sequence next
    to save_reconstruction_panels' PNGs: visualizations/<label>/<sample>/video.mp4."""
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
    return write_comparison_video(
        prediction=prediction.detach().float().cpu().numpy(),
        target=target.detach().float().cpu().numpy(),
        channel_names=channel_names,
        out_path=Path(output_dir) / "visualizations" / label / safe_id / "video.mp4",
        title=f"{label}  sample={sample_id}",
        fps=fps,
        dpi=dpi,
    )


def save_metrics_history(rows: list[dict[str, float]], output_dir: str | Path) -> tuple[Path, Path]:
    """Write epoch metrics to CSV and update the train/validation loss curve."""
    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "metrics.csv"
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    flat_axes = axes.ravel()
    for axis, metric, title in zip(
        flat_axes[:5],
        ("total_loss", "rmse", "h1", "ssim", "mlw"),
        ("Total reconstruction loss", "RMSE", "H1 semi-norm", "SSIM loss", "Wavelet loss"),
    ):
        axis.plot(epochs, [row[f"train_{metric}"] for row in rows], marker="o", label="train")
        axis.plot(epochs, [row[f"val_{metric}"] for row in rows], marker="o", label="validation")
        axis.set(title=title, xlabel="Epoch", ylabel="Loss")
        axis.grid(alpha=0.3)
        axis.legend()

    flat_axes[5].plot(epochs, [row["val_vrmse"] for row in rows], marker="o", color="tab:red")
    flat_axes[5].set(title="Validation VRMSE", xlabel="Epoch", ylabel="VRMSE")
    flat_axes[5].grid(alpha=0.3)

    curve_path = metrics_dir / "loss_curves.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    return csv_path, curve_path

#!/usr/bin/env python3
"""
Render scripts/inference_shockwave.py's .npz predictions as videos, one per sample.

Loads the matching ground truth from the source .h5 (same sample id) and
animates all four channels with GT | prediction | residual columns.

Usage:
    python scripts/visualize_dit_predictions.py \\
        --pred_dir predictions/ \\
        --h5 data/euler_mq/256x256_ds/train.h5 \\
        --out_dir predictions/videos
"""

import argparse
from pathlib import Path

import numpy as np

from windinet.training.shockwave_data import CHANNEL_NAMES, ShockWaveDataset
from windinet.training.vae_visualization import write_comparison_video


def load_ground_truth(dataset: ShockWaveDataset, sample_id: str, channel_names: list[str]) -> dict[str, np.ndarray]:
    idx = dataset.ids.index(sample_id)
    sample = dataset[idx]
    return {name: sample[name].numpy() for name in channel_names}  # [T,H,W]


def render_video(
    *,
    pred: dict[str, np.ndarray],
    gt: dict[str, np.ndarray],
    sample_id: str,
    gamma: float,
    out_path: Path,
    fps: int,
    dpi: int,
) -> None:
    """One MP4 per sample: a row per channel x 3 columns (GT/Pred/Residual)."""
    channel_names = list(pred)
    write_comparison_video(
        prediction=np.stack([pred[name] for name in channel_names]),
        target=np.stack([gt[name] for name in channel_names]),
        channel_names=channel_names,
        out_path=out_path,
        title=f"{sample_id}  gamma={gamma:.4f}",
        fps=fps,
        dpi=dpi,
    )
    print(f"  {sample_id} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=Path, required=True,
                    help="Directory of .npz files from scripts/inference_shockwave.py")
    ap.add_argument("--h5", type=Path, required=True,
                    help="ShockWave HDF5 file the predictions' initial conditions came from")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Output directory for .mp4 files (default: <pred_dir>/videos)")
    ap.add_argument("--sample_ids", type=str, default=None,
                    help="Comma-separated subset of sample ids (default: every .npz in pred_dir)")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()

    out_dir = args.out_dir or (args.pred_dir / "videos")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_ids:
        npz_paths = [args.pred_dir / f"{sid}.npz" for sid in args.sample_ids.split(",")]
    else:
        npz_paths = sorted(args.pred_dir.glob("*.npz"))

    if not npz_paths:
        raise SystemExit(f"No .npz predictions found in {args.pred_dir}")

    dataset = ShockWaveDataset(args.h5)

    print(f"Rendering {len(npz_paths)} sample(s) -> {out_dir}")
    for npz_path in npz_paths:
        sample_id = npz_path.stem
        data = np.load(npz_path)
        # Whichever fields the run predicted (all four, or e.g. the 3-channel run's).
        pred = {name: data[name] for name in CHANNEL_NAMES if name in data.files}
        gamma = float(data["gamma"])
        gt = load_ground_truth(dataset, sample_id, list(pred))
        render_video(
            pred=pred, gt=gt, sample_id=sample_id, gamma=gamma,
            out_path=out_dir / f"{sample_id}.mp4", fps=args.fps, dpi=args.dpi,
        )

    print(f"\nDone! {len(npz_paths)} video(s) -> {out_dir}")


if __name__ == "__main__":
    main()

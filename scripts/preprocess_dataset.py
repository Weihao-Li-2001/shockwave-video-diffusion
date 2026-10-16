# checked and changed

"""
Encode ShockWave simulations into VAE latents for DiT training.

Normalizes each simulation exactly as the fine-tuned VAE saw it during
training, encodes it with that VAE (posterior mean), and saves the latents and
the scalar conditioning (gamma) as `.pt` files.

Expected input dataset::

    data_root/
        train.h5 (or val.h5 / test.h5)
            <sample_id>/
                density        [T, H, W]
                momentum_x     [T, H, W]
                momentum_y     [T, H, W]
                pressure       [T, H, W]
                metadata such as gamma

Output structure (compatible with PrecomputedDataset)::

    output_dir/
        latents/<sample_id>.pt     # {latents, num_frames, height, width}
        scalars/<sample_id>.pt     # {scalars, scalar_names}
        normalization.json         # the channel stats the latents were built with
        split_manifest.json        # which ids went to train vs val (if --eval-sims)
        val/
            latents/<sample_id>.pt # held-out split, same layout
            scalars/<sample_id>.pt

With ``--eval-sims N`` the held-out simulations are written under ``val/`` instead
of the top-level ``latents``/``scalars``, so ``PrecomputedDataset(output_dir)``
sees only the training split and ``PrecomputedDataset(output_dir/"val")`` sees
only the held-out one. Pass ``val/`` to the DiT trainer as
``validation.data_root``.

Usage:
    python scripts/preprocess_dataset.py data/euler_mq/256x256_ds/train.h5 \
        --output-dir outputs/latents/baseline \
        --inflate-checkpoint outputs/vae/baseline/checkpoints/vae_shockwave_best.safetensors \
        --eval-sims 500
"""

import json
from pathlib import Path

import torch
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from windinet.inference.model_loader import load_inflated_vae
from windinet.latent_utils import encode_video
from windinet.vae_adapter import latent_space_fingerprint

from windinet.training.shockwave_data import (
    CHANNEL_NAMES,
    ShockWaveDataset,
    build_shockwave_video,
    extract_scalars,
    load_channel_normalization,
)

from windinet.utils import get_default_device, logger

console = Console()
app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


def _parse_floats(raw: str | None) -> list[float] | None:
    return None if raw is None else [float(v) for v in raw.split(",")]


def _save_atomic(obj, path: Path) -> None:
    """torch.save via a temp file + rename, so a job killed mid-write (e.g. at
    its walltime) never leaves a truncated file that a resumed run would skip
    as already done."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _resolve_normalization(
    inflate_checkpoint: Path,
    vae_config: str | None,
    channel_mean: str | None,
    channel_std: str | None,
    normalization_clip: float | None,
) -> tuple[list[str], list[float], list[float], float, str]:
    """Channel layout and normalization the latents must be built with.

    The VAE only ever saw fields normalized as clip((x - mean) / (std * clip)),
    so the stats must be the ones it was trained with. By default they come from
    the VAE run's training_config.yaml next to the checkpoint.

    Returns (channel names, mean, std, clip, source description). The channel
    names are the VAE run's data.channel_order (all four fields, or a subset);
    explicit --channel-mean/--channel-std imply the default four-channel layout.
    """
    explicit_mean = _parse_floats(channel_mean)
    explicit_std = _parse_floats(channel_std)
    if (explicit_mean is None) != (explicit_std is None):
        raise typer.BadParameter("--channel-mean and --channel-std must be given together")
    if explicit_mean is not None:
        if len(explicit_mean) != len(CHANNEL_NAMES):
            raise typer.BadParameter(f"--channel-mean/--channel-std need {len(CHANNEL_NAMES)} values ({CHANNEL_NAMES})")
        return list(CHANNEL_NAMES), explicit_mean, explicit_std, normalization_clip or 5.0, "command line"

    # Auto-discover: outputs/<run>/checkpoints/vae_*.safetensors
    #             -> outputs/<run>/training_config.yaml
    config_path = Path(vae_config) if vae_config else inflate_checkpoint.parent.parent / "training_config.yaml"
    if not config_path.is_file():
        raise typer.BadParameter(
            f"Cannot determine channel normalization: no VAE training config at {config_path}. "
            "Pass --vae-config, or give --channel-mean/--channel-std explicitly. Encoding "
            "un-normalized fields would silently produce unusable latents, so there is no default."
        )
    stats = load_channel_normalization(config_path)
    clip = normalization_clip if normalization_clip is not None else stats["normalization_clip"]
    return stats["channel_names"], stats["channel_mean"], stats["channel_std"], float(clip), str(config_path)


@app.command()
def main(
    data_root: str = typer.Argument(..., help="ShockWave HDF5 file (e.g. data/euler_mq/256x256_ds/train.h5)"),
    output_dir: str = typer.Option(..., help="Output directory for preprocessed data"),
    model_source: str = typer.Option(
        default="LTXV_2B_0.9.6_DEV",
        help="LTX-Video model version for VAE encoding",
    ),
    num_sim_frames: int = typer.Option(
        default=None,
        help="Keep only the first N simulation frames (e.g. 97 = 8*12+1, which the LTX VAE "
        "encodes without padding). Must match the VAE run's data.num_sim_frames. Default: all.",
    ),
    device: str = typer.Option(default=str(get_default_device()), help="Device for VAE encoding"),
    max_samples: int = typer.Option(default=0, help="Limit number of samples (0 = all)"),
    inflate_checkpoint: str = typer.Option(
        default=None,
        help="Finetuned inflate-mode VAE checkpoint (.safetensors). REQUIRED to encode "
        "4-channel CFD fields: the base VAE only reads 3 channels. Its grown conv_in and "
        "finetuned decoder are loaded so latents match the VAE that will decode them.",
    ),
    scalar_names: str = typer.Option(
        default="gamma",
        help="Comma-separated scalar conditioning names to extract",
    ),
    vae_config: str = typer.Option(
        default=None,
        help="VAE training_config.yaml to read the channel normalization from. "
        "Defaults to the one next to --inflate-checkpoint.",
    ),
    channel_mean: str = typer.Option(
        default=None, help="Override: comma-separated per-channel means (with --channel-std)"
    ),
    channel_std: str = typer.Option(
        default=None, help="Override: comma-separated per-channel stds (with --channel-mean)"
    ),
    normalization_clip: float = typer.Option(
        default=None, help="Override: normalization clip (defaults to the VAE config's value)"
    ),
    eval_sims: int = typer.Option(
        default=0,
        help="Hold out N simulations as a DiT validation split, written to <output-dir>/val. "
        "0 disables the split and encodes everything into the training set. Use the VAE "
        "run's data.eval_sims (500 in configs/vae/) to hold out the same simulations.",
    ),
    split_seed: int = typer.Option(
        default=42,
        help="Seed for the train/val partition. Must match the VAE run's seed for the two "
        "splits to be identical.",
    ),
) -> None:
    """Preprocess ShockWave HDF5 simulations into VAE latents for DiT training."""
    output_path = Path(output_dir)
    latents_dir = output_path / "latents"
    scalars_dir = output_path / "scalars"
    latents_dir.mkdir(parents=True, exist_ok=True)
    scalars_dir.mkdir(parents=True, exist_ok=True)

    val_latents_dir = output_path / "val" / "latents"
    val_scalars_dir = output_path / "val" / "scalars"

    parsed_scalar_names = [s.strip() for s in scalar_names.split(",")]

    # Load dataset
    dataset = ShockWaveDataset(data_root, num_sim_frames=num_sim_frames)
    if num_sim_frames is not None:
        logger.info(f"Truncating every simulation to its first {num_sim_frames} frames")
    if max_samples > 0:
        if eval_sims > 0:
            raise typer.BadParameter(
                "--max-samples truncates dataset.ids, which shifts every index the split "
                "permutation is drawn against. The resulting val set would not match the "
                "VAE run's. Use one or the other, not both."
            )
        dataset.ids = dataset.ids[:max_samples]
    logger.info(f"Found {len(dataset)} samples in {data_root}")

    # Held-out split, identical to VaeTrainer's (same sorted ids, seed and
    # randperm), so with the same --eval-sims / --split-seed the DiT holds out
    # exactly the simulations the VAE held out.
    eval_indices: set[int] = set()
    if eval_sims > 0:
        if eval_sims >= len(dataset):
            raise typer.BadParameter(
                f"--eval-sims {eval_sims} leaves no training data (dataset has {len(dataset)} sims)"
            )
        split_generator = torch.Generator().manual_seed(split_seed)
        perm = torch.randperm(len(dataset), generator=split_generator).tolist()
        eval_indices = set(perm[len(dataset) - eval_sims:])
        val_latents_dir.mkdir(parents=True, exist_ok=True)
        val_scalars_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Split (seed={split_seed}): {len(dataset) - eval_sims} train, {eval_sims} val "
            f"-> {output_path / 'val'}"
        )

    # The fine-tuned (inflated) VAE; the same one must later decode these latents.
    if not inflate_checkpoint:
        raise typer.BadParameter(
            "--inflate-checkpoint is required: the base VAE reads only 3 channels and "
            "cannot encode the 4-channel CFD fields. Pass the finetuned inflate VAE, e.g. "
            "outputs/vae/baseline/checkpoints/vae_shockwave_best.safetensors"
        )
    logger.info(f"Loading inflated VAE ({model_source}) + checkpoint {inflate_checkpoint}...")
    vae = load_inflated_vae(model_source, inflate_checkpoint, dtype=torch.bfloat16, device=device)
    vae = vae.to(device).eval()
    vae.requires_grad_(False)

    # The fields must be normalized exactly as during VAE finetuning before they
    # are encoded, otherwise the latents are off-distribution and unusable.
    channel_names, norm_mean, norm_std, norm_clip, norm_source = _resolve_normalization(
        Path(inflate_checkpoint), vae_config, channel_mean, channel_std, normalization_clip
    )
    logger.info(
        f"Channels {channel_names}, normalization from {norm_source}: "
        f"mean={norm_mean}, std={norm_std}, clip={norm_clip}"
    )
    # Record the normalization (inference inverts it) and the latent-space
    # fingerprint that DiT training and inference check against.
    fingerprint = latent_space_fingerprint(
        Path(inflate_checkpoint), norm_mean, norm_std, norm_clip
    )
    logger.info(f"Latent space fingerprint: {fingerprint}")
    (output_path / "normalization.json").write_text(
        json.dumps(
            {
                "channel_names": channel_names,
                "channel_mean": norm_mean,
                "channel_std": norm_std,
                "normalization_clip": norm_clip,
                "vae_checkpoint": str(Path(inflate_checkpoint).resolve()),
                "latent_fingerprint": fingerprint,
                "source": norm_source,
                "num_sim_frames": num_sim_frames,
            },
            indent=2,
        )
    )

    processed = 0
    skipped = 0
    train_ids: list[str] = []
    val_ids: list[str] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Encoding", total=len(dataset))

        for idx in range(len(dataset)):
            sample = dataset[idx]
            sample_id = sample["id"]

            is_eval = idx in eval_indices
            (val_ids if is_eval else train_ids).append(sample_id)
            latent_path = (val_latents_dir if is_eval else latents_dir) / f"{sample_id}.pt"
            scalar_path = (val_scalars_dir if is_eval else scalars_dir) / f"{sample_id}.pt"

            if latent_path.exists() and scalar_path.exists():
                skipped += 1
                progress.advance(task)
                continue

            try:
                # Build conditioning video and encode
                video = build_shockwave_video(
                    sample,
                    device=torch.device(device),
                    channel_mean=norm_mean,
                    channel_std=norm_std,
                    normalization_clip=norm_clip,
                    channel_order=channel_names,
                )  # [1, C, F, H, W], normalized to [-1, 1]
                video = video.permute(0, 2, 1, 3, 4)  # [1, F, C, H, W] for encode_video

                with torch.no_grad():
                    latent_dict = encode_video(vae, video, device=torch.device(device), dtype=torch.bfloat16)

                # Save latents (squeeze batch dim so DataLoader collation gives [B, seq, C])
                _save_atomic({
                    "latents": latent_dict["latents"].squeeze(0).cpu(),
                    "num_frames": latent_dict["num_frames"],
                    "height": latent_dict["height"],
                    "width": latent_dict["width"],
                }, latent_path)

                # Save scalars
                scalars = extract_scalars(sample["meta"], parsed_scalar_names)
                scalar_tensor = torch.tensor(
                    [scalars[name] for name in parsed_scalar_names],
                    dtype=torch.float32,
                )
                _save_atomic({
                    "scalars": scalar_tensor,
                    "scalar_names": parsed_scalar_names,
                }, scalar_path)

                processed += 1

            except Exception as e:
                logger.warning(f"Failed to process {sample_id}: {e}")
                skipped += 1

            progress.advance(task)

            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "xpu":
                torch.xpu.empty_cache()

    # Record the partition so the val set is auditable and can be reproduced
    # (or intersected with the VAE's) without re-deriving the permutation.
    (output_path / "split_manifest.json").write_text(
        json.dumps(
            {
                "data_root": str(Path(data_root).resolve()),
                "split_seed": split_seed,
                "eval_sims": eval_sims,
                "num_train": len(train_ids),
                "num_val": len(val_ids),
                "train_ids": sorted(train_ids),
                "val_ids": sorted(val_ids),
            },
            indent=2,
        )
    )

    logger.info(f"Done: {processed} processed, {skipped} skipped")
    logger.info(f"Latents: {latents_dir}")
    logger.info(f"Scalars: {scalars_dir}")
    if eval_sims > 0:
        logger.info(f"Held-out split ({len(val_ids)} sims): {output_path / 'val'}")
    logger.info(f"Split manifest: {output_path / 'split_manifest.json'}")


if __name__ == "__main__":
    app()

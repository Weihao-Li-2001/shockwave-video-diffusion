#!/usr/bin/env python
"""Latent-distribution diagnostic for a fine-tuned VAE (single process, encode only).

LTX-Video normalises its latents with ``latents_mean``/``latents_std`` buffers
from video pretraining, and the DiT expects latents distributed accordingly.
This script measures how far the actual latents of the CFD fields are from
that, per latent channel, for:

  (a) the freshly inflated, not fine-tuned VAE (the starting point of every
      run; with zero-initialised new channels this is the pretrained encoder), and
  (b) the fine-tuned checkpoint.

For each it reports the raw posterior mean/std (comparable to
``latents_mean``/``latents_std``) and the normalised latents the DiT consumes
(ideally mean 0 / std 1 per channel), plus raw input-field statistics.

Usage:
    python scripts/latent_stats.py configs/vae/baseline.yaml \\
        --checkpoint outputs/vae/baseline/checkpoints/vae_shockwave_best.safetensors \\
        --split test --test-h5 data/euler_mq/256x256_ds/test.h5 \\
        --num-samples 32 --output outputs/latent_stats_baseline.json

Without --split test it uses the validation split of train.h5, which is held
out from training but was used for model selection.
"""

import json
from pathlib import Path

import torch
import typer
import yaml
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader, Subset

from windinet.config import VaeTrainerConfig
from windinet.inference.model_loader import load_vae
from windinet.training.shockwave_data import ShockWaveDataset, build_shockwave_video
from windinet.utils import get_default_device
from windinet.vae_adapter import inflate_vae_io_channels, load_inflated_vae_checkpoint

console = Console()
app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)

RAW_CHANNELS = ["density", "momentum_x", "momentum_y", "pressure"]


class ChannelStats:
    """Streaming per-channel mean/std over an arbitrary-rank tensor's channel axis."""

    def __init__(self, num_channels: int, device: torch.device | str = "cpu") -> None:
        self.count = 0
        self.sum = torch.zeros(num_channels, dtype=torch.float64, device=device)
        self.sumsq = torch.zeros(num_channels, dtype=torch.float64, device=device)

    def update(self, x: torch.Tensor, channel_dim: int) -> None:
        """x: any shape with `num_channels` at `channel_dim`."""
        x = x.movedim(channel_dim, 0).reshape(x.shape[channel_dim], -1).double()
        self.count += x.shape[1]
        self.sum += x.sum(dim=1)
        self.sumsq += x.square().sum(dim=1)

    def finalize(self) -> dict[str, list[float]]:
        mean = self.sum / self.count
        var = (self.sumsq / self.count - mean.square()).clamp_min(0.0)
        return {"mean": mean.tolist(), "std": var.sqrt().tolist()}


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
def _encode_stats(
    vae,
    loader: DataLoader,
    cfg: VaeTrainerConfig,
    device: torch.device,
    raw_stats: ChannelStats | None,
) -> dict:
    sf = float(getattr(vae.config, "scaling_factor", 1.0))
    latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(device, torch.float32)
    latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(device, torch.float32)
    num_latent_channels = vae.latents_mean.numel()

    posterior_mean_stats = ChannelStats(num_latent_channels, device=device)
    posterior_std_stats = ChannelStats(num_latent_channels, device=device)
    rescaled_stats = ChannelStats(num_latent_channels, device=device)

    for batch in loader:
        video = build_shockwave_video(
            batch,
            device=device,
            channel_mean=cfg.data.channel_mean,
            channel_std=cfg.data.channel_std,
            normalization_clip=cfg.data.normalization_clip,
            channel_order=cfg.data.channel_order,
        )
        if raw_stats is not None:
            # Raw physical units, on the same sims as the latent statistics.
            raw_fields = torch.stack([batch[name].to(device).float() for name in RAW_CHANNELS], dim=1)
            raw_stats.update(raw_fields, channel_dim=1)

        out = vae.encode(video)
        posterior_mean = out.latent_dist.mean.float()
        posterior_logvar = out.latent_dist.logvar.float()
        posterior_std = (0.5 * posterior_logvar).exp()
        rescaled = (posterior_mean - latents_mean) * sf / latents_std

        posterior_mean_stats.update(posterior_mean, channel_dim=1)
        posterior_std_stats.update(posterior_std, channel_dim=1)
        rescaled_stats.update(rescaled, channel_dim=1)

    return {
        "posterior_mean": posterior_mean_stats.finalize(),
        "posterior_std": posterior_std_stats.finalize(),
        "rescaled_latents": rescaled_stats.finalize(),
    }


def _summarize(block: dict) -> dict[str, float]:
    rescaled_mean = torch.tensor(block["rescaled_latents"]["mean"])
    rescaled_std = torch.tensor(block["rescaled_latents"]["std"])
    return {
        "rescaled_mean_abs_avg": rescaled_mean.abs().mean().item(),
        "rescaled_mean_abs_max": rescaled_mean.abs().max().item(),
        "rescaled_std_avg": rescaled_std.mean().item(),
        "rescaled_std_min": rescaled_std.min().item(),
        "rescaled_std_max": rescaled_std.max().item(),
    }


@app.command()
def main(
    config_path: str = typer.Argument(..., help="VaeTrainerConfig-shaped YAML (data_root/adapter/normalization only matter)"),
    checkpoint: str = typer.Option(..., help="Finetuned inflate-mode checkpoint (ltx-inflated-io-v1 safetensors)"),
    num_samples: int = typer.Option(32, help="How many sims to encode"),
    batch_size: int = typer.Option(4, help="Sims per encode() call"),
    split: str = typer.Option(
        "eval",
        help=(
            "'eval' (held-out within cfg.data.data_root's train.h5, same "
            "permutation VaeTrainer used to pick this checkpoint's best epoch) "
            "or 'train' (also within train.h5) -- both were exposed to training/"
            "tuning decisions, if only indirectly. 'test' is the clean option: "
            "a standalone file (test.h5) never touched by any train/eval split "
            "or hyperparameter decision in this project -- requires --test-h5."
        ),
    ),
    test_h5: str = typer.Option(
        None,
        help="Path to a standalone test.h5 (sibling of train.h5, e.g. data/euler_mq/128x128_ds/test.h5). Required and only used when --split test.",
    ),
    output: str = typer.Option("latent_stats.json", help="Where to write the full per-channel JSON report"),
) -> None:
    with open(config_path) as f:
        cfg = VaeTrainerConfig(**yaml.safe_load(f))

    device = get_default_device()
    console.print(f"Device: {device}")

    if split == "test":
        if not test_h5:
            raise typer.BadParameter("--test-h5 is required when --split test (a standalone file, not a slice of train.h5)")
        # test.h5 is entirely held out: take the first num_samples sims.
        full_dataset = ShockWaveDataset(Path(test_h5), num_sim_frames=cfg.data.num_sim_frames)
        chosen = list(range(len(full_dataset)))[:num_samples]
        console.print(
            f"Encoding {len(chosen)} sims from the standalone test set "
            f"'{test_h5}' (of {len(full_dataset)} total, never used in any "
            "train/eval split or tuning decision)"
        )
    else:
        # VaeTrainer's exact split, so these sims were held out from training.
        full_dataset = ShockWaveDataset(Path(cfg.data.data_root), num_sim_frames=cfg.data.num_sim_frames)
        split_generator = torch.Generator().manual_seed(cfg.seed)
        perm = torch.randperm(len(full_dataset), generator=split_generator).tolist()
        n_eval = min(cfg.data.eval_sims, len(full_dataset))
        n_train = len(full_dataset) - n_eval
        eval_indices = perm[n_train:n_train + n_eval]
        train_indices = perm[:n_train]
        chosen = (eval_indices if split == "eval" else train_indices)[:num_samples]
        console.print(f"Encoding {len(chosen)} sims from the '{split}' split (of {len(full_dataset)} total)")

    loader = DataLoader(
        Subset(full_dataset, chosen),
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.num_dataloader_workers,
    )

    console.print("\n[bold]Pretrained (freshly-inflated, no finetuning)[/bold]")
    vae_pretrained = _build_vae(cfg, checkpoint=None, device=device)
    raw_stats = ChannelStats(len(RAW_CHANNELS), device=device)
    pretrained_result = _encode_stats(vae_pretrained, loader, cfg, device, raw_stats=raw_stats)
    raw_result = raw_stats.finalize()
    reference_mean = vae_pretrained.latents_mean.tolist()
    reference_std = vae_pretrained.latents_std.tolist()
    del vae_pretrained
    if device.type != "cpu":
        getattr(torch, device.type).empty_cache()

    console.print("\n[bold]Finetuned[/bold]")
    vae_finetuned = _build_vae(cfg, checkpoint=checkpoint, device=device)
    finetuned_result = _encode_stats(vae_finetuned, loader, cfg, device, raw_stats=None)
    del vae_finetuned

    report = {
        "config": config_path,
        "checkpoint": checkpoint,
        "split": split,
        "data_source": test_h5 if split == "test" else cfg.data.data_root,
        "sim_indices": chosen,
        "num_latent_channels": len(reference_mean),
        "raw_channels": {name: {"mean": raw_result["mean"][i], "std": raw_result["std"][i]} for i, name in enumerate(RAW_CHANNELS)},
        "reference_latents_mean": reference_mean,
        "reference_latents_std": reference_std,
        "pretrained": pretrained_result,
        "finetuned": finetuned_result,
    }
    Path(output).write_text(json.dumps(report, indent=2))
    console.print(f"\nFull per-channel report written to {output}")

    raw_table = Table(title="Raw physical channels (this sample, not the full dataset)")
    raw_table.add_column("channel")
    raw_table.add_column("mean", justify="right")
    raw_table.add_column("std", justify="right")
    for name in RAW_CHANNELS:
        raw_table.add_row(name, f"{report['raw_channels'][name]['mean']:.4f}", f"{report['raw_channels'][name]['std']:.4f}")
    console.print(raw_table)

    summary_table = Table(title="Rescaled latents (target: mean 0, std 1 per channel)")
    summary_table.add_column("encoder")
    summary_table.add_column("|mean| avg", justify="right")
    summary_table.add_column("|mean| max", justify="right")
    summary_table.add_column("std avg", justify="right")
    summary_table.add_column("std min", justify="right")
    summary_table.add_column("std max", justify="right")
    for label, block in [("pretrained (no finetune)", pretrained_result), ("finetuned", finetuned_result)]:
        s = _summarize(block)
        summary_table.add_row(
            label,
            f"{s['rescaled_mean_abs_avg']:.4f}",
            f"{s['rescaled_mean_abs_max']:.4f}",
            f"{s['rescaled_std_avg']:.4f}",
            f"{s['rescaled_std_min']:.4f}",
            f"{s['rescaled_std_max']:.4f}",
        )
    console.print(summary_table)

    pre_mean = torch.tensor(pretrained_result["rescaled_latents"]["mean"])
    ft_mean = torch.tensor(finetuned_result["rescaled_latents"]["mean"])
    pre_std = torch.tensor(pretrained_result["rescaled_latents"]["std"])
    ft_std = torch.tensor(finetuned_result["rescaled_latents"]["std"])
    console.print(
        "\nFinetuning-induced shift in rescaled-latent mean (L2 over channels): "
        f"{(ft_mean - pre_mean).norm().item():.4f}"
    )
    console.print(
        f"Finetuning-induced std ratio (finetuned/pretrained, avg over channels): "
        f"{(ft_std / pre_std).mean().item():.4f}"
    )
    top_shift = (ft_mean - pre_mean).abs().topk(min(5, len(ft_mean)))
    console.print(f"Top-5 latent channels by |mean shift|: {[(i.item(), round(v.item(), 4)) for v, i in zip(*top_shift)]}")


if __name__ == "__main__":
    app()

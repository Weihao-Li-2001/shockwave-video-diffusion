#!/usr/bin/env python3
"""
Batch inference: roll out simulations from their initial condition.

Conditions on frame 0 of each simulation plus gamma, rolls out the remaining
frames with the DiT and decodes them with the inflate-mode fine-tuned VAE.
Saves one .npz per simulation with the four physical fields (density,
momentum_x, momentum_y, pressure); render them with
scripts/visualize_dit_predictions.py.

Usage:
    python scripts/inference_shockwave.py configs/dit/inference.yaml \\
        --h5 data/euler_mq/256x256_ds/test.h5 \\
        --out_dir predictions/ --num_samples 8
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from torch.amp import autocast

from windinet.checkpoints import ensure_checkpoint
from windinet.config import ScalarConditioningConfig
from windinet.inference.model_loader import load_ltxv_components, select_vae_env
from windinet.inference.pipeline import LTXConditionPipeline
from windinet.scalar_embeddings import ScalarEmbedding
from windinet.training.shockwave_data import (
    ShockWaveDataset,
    load_channel_normalization,
    normalize_fields,
)
from windinet.utils import get_default_device
from windinet.training.vae_visualization import denormalize_fields
from windinet.training.dit_visualization import pick_fixed_visualization_sample_ids
from windinet.vae_adapter import latent_space_fingerprint

DTYPE = torch.bfloat16


def verify_latent_space(vae_checkpoint, dit_checkpoint, stats) -> None:
    """Refuse to decode a DiT's latents with a VAE that did not produce them.

    A mismatched VAE gives plausible-looking but physically wrong fields with no
    error. DiT training records the latent-space fingerprint of its training
    data (latent_provenance.json); this compares it with the VAE about to be used.
    """
    provenance_path = Path(dit_checkpoint).parent.parent / "latent_provenance.json"
    if not provenance_path.is_file():
        print(
            f"WARNING: no {provenance_path} -- cannot verify that {vae_checkpoint} is the VAE "
            "the DiT's latents were built with. A mismatch here produces wrong physics silently."
        )
        return

    expected = json.loads(provenance_path.read_text()).get("latent_fingerprint")
    if not expected:
        print(f"WARNING: {provenance_path} records no latent_fingerprint; skipping the check.")
        return

    actual = latent_space_fingerprint(
        Path(vae_checkpoint),
        stats["channel_mean"],
        stats["channel_std"],
        stats["normalization_clip"],
    )
    if actual != expected:
        raise SystemExit(
            f"VAE / DiT latent space mismatch.\n"
            f"  DiT was trained on latents with fingerprint : {expected}\n"
            f"  this VAE + normalization fingerprints as    : {actual}\n"
            f"  vae_checkpoint : {vae_checkpoint}\n"
            f"  provenance     : {provenance_path}\n"
            f"Decoding would produce physically wrong fields. Point vae_checkpoint at the VAE "
            f"recorded in {provenance_path}, or re-run preprocessing and retrain the DiT.\n"
            f"(A decoder-only VAE refinement keeps the fingerprint and is safe to swap in; a "
            f"changed encoder.conv_in or changed channel normalization does not.)"
        )
    print(f"Latent space verified: fingerprint {actual} matches the DiT's training latents")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path, help="Path to inference YAML config")
    ap.add_argument("--h5", type=Path, default=None,
                    help="ShockWave HDF5 file to take initial conditions from")
    ap.add_argument("--out_dir", type=Path, default=Path("outputs/inference_shockwave"),
                    help="Output directory for .npz predictions")
    ap.add_argument("--sample_ids", type=str, default=None,
                    help="Comma-separated sample ids to run (default: the first --num_samples, "
                         "or the fixed picks from --preprocessed_data_root if given)")
    ap.add_argument("--num_samples", type=int, default=3,
                    help="Number of samples to run when --sample_ids is not given (default 3, "
                         "matching DitVisualizationConfig's training-time default)")
    ap.add_argument("--preprocessed_data_root", type=Path, default=None,
                    help="Lock onto the SAME fixed sample ids (and --h5, if not overridden) "
                         "DiT training's periodic visualization used for this run -- pass the "
                         "same preprocessed_data_root the training config's data.preprocessed_data_root "
                         "pointed at (post-substitution, e.g. dit_preprocessed/<name>), with the "
                         "same --num_samples the training config's visualization.num_samples used "
                         "(default 3 both sides). Overrides --sample_ids/--h5 when given.")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Override transformer checkpoint")
    ap.add_argument("--scalar_checkpoint", type=Path, default=None, help="Override scalar embedding checkpoint")
    ap.add_argument("--vae_checkpoint", type=Path, default=None, help="Override inflate-mode VAE checkpoint")
    ap.add_argument("--num_inference_steps", type=int, default=None, help="Override denoising steps")
    ap.add_argument("--guidance_scale", type=float, default=None, help="Override guidance scale")
    return ap.parse_args()


def load_config(args):
    """Load YAML config and apply CLI overrides."""
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for key in ("checkpoint", "scalar_checkpoint", "vae_checkpoint",
                "num_inference_steps", "guidance_scale"):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = str(value) if isinstance(value, Path) else value
    if args.h5 is not None:
        cfg["h5"] = str(args.h5)

    if not cfg.get("h5") and args.preprocessed_data_root is None:
        raise SystemExit(
            "No dataset given: pass --h5, set 'h5' in the config, or pass "
            "--preprocessed_data_root to derive it from that run's split_manifest.json."
        )
    return cfg


def make_pipe(model_source, device):
    c = load_ltxv_components(
        model_source=model_source,
        transformer_dtype=DTYPE,
        vae_dtype=DTYPE,
    )
    pipe = LTXConditionPipeline(
        scheduler=deepcopy(c.scheduler),
        vae=c.vae,
        text_encoder=None,
        tokenizer=None,
        transformer=c.transformer,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False)
    pipe.transformer.requires_grad_(False)

    # The diffusers video processor maps [0, 1] images to [-1, 1] and back with a
    # clamp, which would wipe out negative momentum. Our fields are already in
    # [-1, 1], so make it a pass-through.
    pipe.video_processor.register_to_config(do_normalize=False)
    return pipe


def load_transformer_weights(pipe, checkpoint):
    print(f"Loading transformer: {checkpoint}")
    sd = load_file(str(checkpoint))
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    if any(k.startswith("transformer.") for k in sd):
        sd = {k.replace("transformer.", "", 1): v for k, v in sd.items() if k.startswith("transformer.")}
    pipe.transformer.load_state_dict(sd, strict=False)


def load_scalar_embedding(checkpoint, scalar_cfg, device):
    print(f"Loading scalar embedding: {checkpoint}")
    emb = ScalarEmbedding(scalar_cfg)
    emb.load_state_dict(load_file(str(checkpoint)))
    return emb.to(device=device, dtype=DTYPE).eval()


def build_initial_condition(sample, stats, device):
    """Normalized IC frame [1, 1, C, H, W] taken from simulation frame 0, in the
    channel layout the VAE was trained with (stats["channel_names"])."""
    fields = torch.stack([sample[name][0] for name in stats["channel_names"]])  # [C, H, W]
    fields = fields.unsqueeze(0)  # [1, C, H, W]
    fields = normalize_fields(
        fields,
        stats["channel_mean"],
        stats["channel_std"],
        stats["normalization_clip"],
    )
    return fields.unsqueeze(1).to(device=device, dtype=DTYPE)  # [B, F, C, H, W]


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args)

    device = get_default_device()

    model_source = cfg.get("model_source", "LTXV_2B_0.9.6_DEV")
    checkpoint = ensure_checkpoint(cfg["checkpoint"])
    scalar_checkpoint = ensure_checkpoint(cfg["scalar_checkpoint"])
    num_inference_steps = cfg.get("num_inference_steps", 2)
    guidance_scale = cfg.get("guidance_scale", 1.0)
    num_frames = cfg.get("num_frames", 105)
    num_output_frames = cfg.get("num_output_frames", 101)
    image_cond_noise_scale = cfg.get("image_cond_noise_scale", 0.0)
    seed = cfg.get("seed", 42)

    if num_frames % 8 != 1:
        raise SystemExit(f"num_frames must be of the form 8k+1 for the LTX VAE, got {num_frames}")

    # The inflate-mode VAE the latents were built with (the loader is chosen
    # from the checkpoint's format metadata).
    vae_ckpt = cfg.get("vae_checkpoint")
    if not vae_ckpt:
        raise SystemExit(
            "vae_checkpoint is required: shockwave fields have 4 channels and the base "
            "LTX VAE reads 3. Point it at an inflate-mode finetuned checkpoint."
        )
    env_var = select_vae_env(ensure_checkpoint(vae_ckpt))
    print(f"VAE checkpoint: {vae_ckpt}  (loader: {env_var})")

    # Channel stats must match the ones the VAE was finetuned with and the ones
    # preprocessing used, or the decoded fields come out on the wrong scale.
    stats = load_channel_normalization(cfg["normalization"])
    print(f"Normalization from {cfg['normalization']}: clip={stats['normalization_clip']}")

    verify_latent_space(ensure_checkpoint(vae_ckpt), checkpoint, stats)

    sc = cfg.get("scalar_conditioning", {})
    scalar_cfg = ScalarConditioningConfig(
        enabled=True,
        scalar_names=sc.get("scalar_names", ["gamma"]),
        scalar_ranges={k: tuple(v) for k, v in sc.get("scalar_ranges", {"gamma": [1.0, 2.0]}).items()},
        embedding_dim=sc.get("embedding_dim", 4096),
        num_tokens_per_scalar=sc.get("num_tokens_per_scalar", 4),
    )

    if args.preprocessed_data_root is not None:
        wanted, manifest_h5 = pick_fixed_visualization_sample_ids(
            args.preprocessed_data_root, args.num_samples
        )
        h5_path = cfg.get("h5") or manifest_h5  # --h5 still wins if the caller passed one
        print(
            f"Locked onto {len(wanted)} fixed sample(s) from "
            f"{args.preprocessed_data_root}/split_manifest.json: {wanted}"
        )
    else:
        wanted, h5_path = (
            [s.strip() for s in args.sample_ids.split(",")] if args.sample_ids else None,
            cfg["h5"],
        )

    dataset = ShockWaveDataset(h5_path)
    if wanted:
        missing = [s for s in wanted if s not in dataset.ids]
        if missing:
            raise SystemExit(f"Sample ids not in {h5_path}: {missing}")
        indices = [dataset.ids.index(s) for s in wanted]
    else:
        indices = list(range(min(args.num_samples, len(dataset))))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pipe = make_pipe(model_source, device)
    load_transformer_weights(pipe, checkpoint)
    scalar_emb = load_scalar_embedding(scalar_checkpoint, scalar_cfg, device)

    print(f"Generating {len(indices)} predictions -> {args.out_dir}  [steps={num_inference_steps}]")

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        name = sample["id"]
        out_path = args.out_dir / f"{name}.npz"
        if out_path.exists():
            print(f"[{i+1}/{len(indices)}] {name} — exists, skipping")
            continue

        H, W = sample["density"].shape[-2:]
        cond_video = build_initial_condition(sample, stats, device)

        scalar_values = [sample["meta"][n] for n in scalar_cfg.scalar_names]
        scalars = torch.tensor([scalar_values], device=device, dtype=DTYPE)
        prompt_embeds = scalar_emb(scalars)
        prompt_mask = torch.ones(1, prompt_embeds.shape[1], device=device, dtype=torch.long)

        g = torch.Generator(device=device).manual_seed(seed + i)
        with autocast(device.type, dtype=DTYPE, enabled=(device.type in ("cuda", "xpu"))):
            out = pipe(
                prompt=None, negative_prompt=None,
                video=cond_video, frame_index=0, strength=1.0,
                width=W, height=H,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                image_cond_noise_scale=image_cond_noise_scale,
                generator=g,
                output_reference_comparison=False,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_mask,
                negative_prompt_embeds=torch.zeros_like(prompt_embeds),
                negative_prompt_attention_mask=prompt_mask.clone(),
                output_type="pt",
            )

        # [F, C, H, W] in normalized space -> physical units, trimmed to the
        # real simulation length (num_frames is padded up to 8k+1).
        pred = out.frames[0].float().cpu()[:num_output_frames]
        pred = pred.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, F, H, W]
        pred = denormalize_fields(
            pred,
            stats["channel_mean"],
            stats["channel_std"],
            stats["normalization_clip"],
        )[0]

        np.savez_compressed(
            out_path,
            **{name_: pred[c].numpy().astype(np.float32) for c, name_ in enumerate(stats["channel_names"])},
            gamma=np.float32(sample["meta"]["gamma"]),
        )

        print(f"[{i+1}/{len(indices)}] {name} — saved ({pred.shape[1]} frames, "
              f"gamma={sample['meta']['gamma']:.4f})")

        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "xpu":
            torch.xpu.empty_cache()

    print(f"\nDone! {len(indices)} .npz files -> {args.out_dir}")


if __name__ == "__main__":
    main()

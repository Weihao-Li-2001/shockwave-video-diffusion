#!/usr/bin/env python3
"""
VRMSE of the full pipeline: VAE-only reconstruction vs. VAE + DiT rollout.

Two passes per simulation, same ground truth, same metric:
  - vae_only:  encode the whole ground-truth sequence, decode, compare
               (the autoencoder's reconstruction floor, no forecasting).
  - vae_dit:   encode frame 0 only (the initial condition), let the DiT roll
               out the remaining frames in latent space, decode, compare
               (what scripts/inference_shockwave.py does).

VRMSE is computed on normalized fields, exactly like VaeTrainer's val_vrmse.
Both passes are reported in pixel space and, for the rollout, also in latent
space against the encoder's own latents, which isolates the DiT's forecast
error from the decoder.

Evaluates the held-out val sims recorded in
<preprocessed_data_root>/split_manifest.json, or every sim of --test_h5 (as
for the reported results). Any DiT checkpoint works, including mid-training ones.

Usage:
    python scripts/eval_dit_vrmse.py configs/dit/inference.yaml \\
        --preprocessed_data_root outputs/latents/baseline \\
        --test_h5 data/euler_mq/256x256_ds/test.h5 \\
        --out_dir outputs/eval/baseline

Checkpoints default to the ones in the config; override them with
--checkpoint / --scalar_checkpoint / --vae_checkpoint. --num_samples limits
the run for a quick check (each sim costs one rollout and three VAE passes).
--save_vis_samples N renders GT / prediction / residual panels for both
passes for N sims spread over the gamma range, under
<out_dir>/visualizations/{vae_only,vae_dit}/<sample_id>/.
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
from windinet.losses import rmse_loss, vrms_loss, vrms_per_channel
from windinet.scalar_embeddings import ScalarEmbedding
from windinet.training.shockwave_data import (
    ShockWaveDataset,
    build_shockwave_video,
    load_channel_normalization,
    normalize_fields,
    pick_gamma_spread_ids,
)
from windinet.training.vae_visualization import denormalize_fields, save_reconstruction_panels
from windinet.utils import get_default_device
from windinet.vae_adapter import latent_space_fingerprint

DTYPE = torch.bfloat16


# ----------------------------------------------------------------------
# Pipeline setup, copied from scripts/inference_shockwave.py (scripts/ is not
# a package). Keep the two in sync.
# ----------------------------------------------------------------------

def verify_latent_space(vae_checkpoint, dit_checkpoint, stats) -> None:
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
        Path(vae_checkpoint), stats["channel_mean"], stats["channel_std"], stats["normalization_clip"],
    )
    if actual != expected:
        raise SystemExit(
            f"VAE / DiT latent space mismatch.\n"
            f"  DiT was trained on latents with fingerprint : {expected}\n"
            f"  this VAE + normalization fingerprints as    : {actual}\n"
            f"  vae_checkpoint : {vae_checkpoint}\n"
            f"  provenance     : {provenance_path}\n"
            f"Point vae_checkpoint at the VAE recorded in {provenance_path}."
        )
    print(f"Latent space verified: fingerprint {actual} matches the DiT's training latents")


def make_pipe(model_source, device):
    c = load_ltxv_components(model_source=model_source, transformer_dtype=DTYPE, vae_dtype=DTYPE)
    pipe = LTXConditionPipeline(
        scheduler=deepcopy(c.scheduler), vae=c.vae, text_encoder=None, tokenizer=None, transformer=c.transformer,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    # Our fields are already in [-1, 1] via the CFD channel stats -- see
    # inference_shockwave.py's own comment on why do_normalize is disabled.
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
    fields = torch.stack([sample[name][0] for name in stats["channel_names"]]).unsqueeze(0)  # [1, C, H, W]
    fields = normalize_fields(fields, stats["channel_mean"], stats["channel_std"], stats["normalization_clip"])
    return fields.unsqueeze(1).to(device=device, dtype=DTYPE)  # [B, F, C, H, W]


# ----------------------------------------------------------------------
# VAE encode/decode with exactly VaeTrainer._encode/_decode's latent rescaling,
# so the vae_only pass matches the trainer's val_vrmse.
# ----------------------------------------------------------------------

def vae_encode(vae, video: torch.Tensor) -> torch.Tensor:
    """video: [B, C, F, H, W], normalized. Returns rescaled latents."""
    out = vae.encode(video)
    posterior_mean = out.latent_dist.mean
    norm_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(posterior_mean.device, posterior_mean.dtype)
    norm_std = vae.latents_std.view(1, -1, 1, 1, 1).to(posterior_mean.device, posterior_mean.dtype)
    sf = float(getattr(vae.config, "scaling_factor", 1.0))
    return (posterior_mean - norm_mean) * sf / norm_std


def vae_decode(vae, latents: torch.Tensor, default_temb: float) -> torch.Tensor:
    mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    std = vae.latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    sf = float(getattr(vae.config, "scaling_factor", 1.0))
    z = latents * std / sf + mean
    temb = torch.full((z.shape[0],), default_temb, device=z.device, dtype=z.dtype)
    return vae.decode(z, temb=temb, return_dict=True).sample


def pad_to(t: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Trim a [B,C,F,H,W] tensor's frame dim to n_frames (both passes may
    return a few more frames than the ground truth due to 8k+1 padding)."""
    return t[:, :, :n_frames]


def trim_latent_frames(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Trim two [B, C, F, H, W] latents to the shorter F (rollout and encode may differ by a frame)."""
    n = min(a.shape[2], b.shape[2])
    return a[:, :, :n], b[:, :, :n]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path, help="Path to inference YAML config")
    ap.add_argument("--preprocessed_data_root", type=Path, required=True,
                     help="The DiT's preprocessed latents (its split_manifest.json gives the h5 "
                          "and the held-out val ids)")
    ap.add_argument("--num_samples", type=int, default=None,
                     help="How many val sims to evaluate (default: all in the manifest)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Override transformer checkpoint")
    ap.add_argument("--scalar_checkpoint", type=Path, default=None, help="Override scalar embedding checkpoint")
    ap.add_argument("--untrained_dit", action="store_true",
                     help="Control run: the stock pretrained transformer with a randomly initialised "
                          "scalar embedding (ignores --checkpoint/--scalar_checkpoint)")
    ap.add_argument("--vae_checkpoint", type=Path, default=None, help="Override inflate-mode VAE checkpoint")
    ap.add_argument("--normalization", type=Path, default=None,
                     help="Override the normalization source (normalization.json, a VAE run's "
                          "training_config.yaml, or a stats yaml)")
    ap.add_argument("--num_inference_steps", type=int, default=None, help="Override denoising steps")
    ap.add_argument("--guidance_scale", type=float, default=None, help="Override guidance scale")
    ap.add_argument("--default_temb", type=float, default=0.0,
                     help="Decoder timestep for the VAE-only pass (the VAE configs' adapter.default_temb)")
    ap.add_argument("--save_vis_samples", type=int, default=3,
                     help="Render GT/prediction/residual panels (both passes) for N gamma-spread samples")
    ap.add_argument("--frame_numbers", type=int, nargs="+", default=[25, 50, 75, 97],
                     help="1-indexed frame numbers to render panels for")
    ap.add_argument("--dpi", type=int, default=150, help="Panel image DPI")
    ap.add_argument("--out_dir", type=Path, default=Path("outputs/eval_dit_vrmse"))
    ap.add_argument("--test_h5", type=Path, default=None,
                     help="Evaluate every sim of this test.h5 instead of the manifest's val ids "
                          "(the manifest is still recorded for provenance)")
    ap.add_argument("--save_npz_samples", type=int, default=0,
                     help="Save float16 fields (gt_raw, gt_norm, vae_only_norm, vae_dit_norm) of N "
                          "gamma-spread sims at --npz_frames, for scripts/make_figures.py")
    ap.add_argument("--npz_frames", type=int, nargs="+", default=[0, 25, 50, 75, 96],
                     help="0-indexed frames saved per npz sample (frame 0 = the conditioning IC)")
    ap.add_argument("--metric_frames", type=int, default=None,
                     help="Score only the first N frames in pixel space (e.g. to compare 101- and "
                          "97-frame models on the same frames); the model still runs its full length")
    ap.add_argument("--npz_dir", type=Path, default=None, help="Where the npz samples go (default: <out_dir>/samples)")
    ap.add_argument("--sim_ids", type=str, nargs="+", default=None,
                     help="Evaluate exactly these sim ids (from any split; each id's split is printed) "
                          "and save an npz for each, e.g. for a figure")
    return ap.parse_args()


def load_config(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for key in ("checkpoint", "scalar_checkpoint", "vae_checkpoint", "normalization",
                "num_inference_steps", "guidance_scale"):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = str(value) if isinstance(value, Path) else value
    return cfg


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args)
    device = get_default_device()

    manifest_path = args.preprocessed_data_root / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if args.test_h5 is not None:
        # val_ids keeps its name below but holds the test ids from here on.
        h5_path = str(args.test_h5)
        val_ids = list(ShockWaveDataset(h5_path).ids)
        print(f"Standalone test set: {len(val_ids)} sims from {h5_path} (manifest {manifest_path} not used for ids)")
    else:
        h5_path = manifest["data_root"]
        val_ids = manifest["val_ids"]
        print(f"Held-out split: {len(val_ids)} sims (seed={manifest['split_seed']}) from {manifest_path}")
    print(f"h5: {h5_path}")

    if args.sim_ids:
        for sid in args.sim_ids:
            split = next((k for k in ("train_ids", "val_ids") if sid in manifest.get(k, [])), "not in manifest")
            print(f"--sim_ids: {sid} ({split})")
        val_ids = list(args.sim_ids)
    if args.num_samples is not None:
        val_ids = val_ids[: args.num_samples]
    print(f"Evaluating {len(val_ids)} sim(s)")

    vis_ids = pick_gamma_spread_ids(val_ids, args.save_vis_samples)
    if vis_ids:
        print(f"Visualizing {len(vis_ids)} sample(s) spanning the gamma range: {sorted(vis_ids)}")
    npz_ids = set(val_ids) if args.sim_ids else pick_gamma_spread_ids(val_ids, args.save_npz_samples)
    npz_dir = args.npz_dir or args.out_dir / "samples"
    if npz_ids:
        print(f"Saving npz fields for {sorted(npz_ids)} (frames {args.npz_frames}) to {npz_dir}")

    model_source = cfg.get("model_source", "LTXV_2B_0.9.6_DEV")
    checkpoint = None if args.untrained_dit else ensure_checkpoint(cfg["checkpoint"])
    scalar_checkpoint = None if args.untrained_dit else ensure_checkpoint(cfg["scalar_checkpoint"])
    num_inference_steps = cfg.get("num_inference_steps", 2)
    guidance_scale = cfg.get("guidance_scale", 1.0)
    # Latents encoded from truncated sims (preprocess_dataset.py --num-sim-frames)
    # record that length; the eval must use the same one, and the DiT then
    # generates the matching 8k+1 length instead of the config's 105.
    num_sim_frames = json.loads(
        (args.preprocessed_data_root / "normalization.json").read_text()
    ).get("num_sim_frames")
    if num_sim_frames is not None:
        num_frames = ((num_sim_frames - 1 + 7) // 8) * 8 + 1
        print(f"Latents built from {num_sim_frames}-frame sims -> generating {num_frames} frames")
    else:
        num_frames = cfg.get("num_frames", 105)
    if args.metric_frames is not None:
        print(f"Scoring only the first {args.metric_frames} frames")
    image_cond_noise_scale = cfg.get("image_cond_noise_scale", 0.0)
    seed = cfg.get("seed", 42)

    vae_ckpt = cfg.get("vae_checkpoint")
    if not vae_ckpt:
        raise SystemExit("vae_checkpoint is required (inflate-mode finetuned checkpoint)")
    select_vae_env(ensure_checkpoint(vae_ckpt))  # must run BEFORE make_pipe: sets which VAE it loads

    stats = load_channel_normalization(cfg["normalization"])
    channel_names = stats["channel_names"]
    n_ch = len(channel_names)
    print(f"Normalization from {cfg['normalization']}: channels={channel_names} clip={stats['normalization_clip']}")
    if args.untrained_dit:
        print("--untrained_dit: skipping latent-space provenance check (no DiT training happened to verify against)")
    else:
        verify_latent_space(ensure_checkpoint(vae_ckpt), checkpoint, stats)

    sc = cfg.get("scalar_conditioning", {})
    scalar_cfg = ScalarConditioningConfig(
        enabled=True,
        scalar_names=sc.get("scalar_names", ["gamma"]),
        scalar_ranges={k: tuple(v) for k, v in sc.get("scalar_ranges", {"gamma": [1.0, 2.0]}).items()},
        embedding_dim=sc.get("embedding_dim", 4096),
        num_tokens_per_scalar=sc.get("num_tokens_per_scalar", 4),
    )

    dataset = ShockWaveDataset(h5_path, num_sim_frames=num_sim_frames)
    missing = [s for s in val_ids if s not in dataset.ids]
    if missing:
        raise SystemExit(f"val_ids not found in {h5_path}: {missing[:5]}...")

    pipe = make_pipe(model_source, device)  # pipe.vae is now the finetuned VAE from vae_ckpt
    if args.untrained_dit:
        print("--untrained_dit: leaving transformer at its stock pretrained weights (no checkpoint loaded)")
        scalar_emb = ScalarEmbedding(scalar_cfg).to(device=device, dtype=DTYPE).eval()
        print("--untrained_dit: using a freshly random-initialized ScalarEmbedding (no pretrained equivalent exists)")
    else:
        load_transformer_weights(pipe, checkpoint)
        scalar_emb = load_scalar_embedding(scalar_checkpoint, scalar_cfg, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_sample = []
    sum_overall = {"vae_only": 0.0, "vae_dit": 0.0}
    sum_channel = {"vae_only": [0.0] * n_ch, "vae_dit": [0.0] * n_ch}
    sum_lat_vrmse, sum_lat_rmse, sum_lat_mse = 0.0, 0.0, 0.0
    sum_lat_channel = None  # lazily sized to the latent's own channel count on first sample
    sum_lat_channel_mse = None

    for i, sid in enumerate(val_ids):
        idx = dataset.ids.index(sid)
        sample = dataset[idx]
        H, W = sample["density"].shape[-2:]
        orig_F = sample["density"].shape[0]
        score_F = min(orig_F, args.metric_frames or orig_F)

        # Ground truth, normalized, [1, C, orig_F, H, W] -- same call
        # preprocess_dataset.py used to build the VAE's own training/eval input.
        gt_video = build_shockwave_video(
            sample, device=device, channel_mean=stats["channel_mean"],
            channel_std=stats["channel_std"], normalization_clip=stats["normalization_clip"],
            channel_order=channel_names,
        )
        target = pad_to(gt_video, orig_F).float()

        # --- VAE-only pass: encode + decode the full ground-truth sequence ---
        # gt_latent is also the reference for the rollout's latent-space error.
        gt_latent = vae_encode(pipe.vae, gt_video.to(DTYPE))
        vae_recon = vae_decode(pipe.vae, gt_latent, args.default_temb).float()
        vae_recon = pad_to(vae_recon, orig_F)

        vo_overall = float(vrms_loss(vae_recon[:, :, :score_F], target[:, :, :score_F]).item())
        vo_channel = vrms_per_channel(vae_recon[:, :, :score_F], target[:, :, :score_F]).tolist()

        # --- VAE+DiT pass: condition on frame 0 only, roll the rest out ---
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
                output_type="latent",
            )
        # With output_type="latent" the pipeline returns [B, C, F, H, W] latents
        # in the same normalised space as vae_encode(), so they compare directly
        # with gt_latent and decode with vae_decode().
        dit_pred_latent = out.frames.float()
        pred_lat, cmp_gt_lat = trim_latent_frames(dit_pred_latent, gt_latent.float())

        lat_vrmse = float(vrms_loss(pred_lat, cmp_gt_lat).item())
        lat_rmse = float(rmse_loss(pred_lat, cmp_gt_lat).item())
        lat_sq_err = (pred_lat - cmp_gt_lat) ** 2
        lat_mse = float(lat_sq_err.mean().item())
        lat_channel = vrms_per_channel(pred_lat, cmp_gt_lat)
        lat_channel_mse = lat_sq_err.mean(dim=(0, 2, 3, 4))
        if sum_lat_channel is None:
            sum_lat_channel = torch.zeros(lat_channel.shape[0])
            sum_lat_channel_mse = torch.zeros(lat_channel_mse.shape[0])
        sum_lat_channel += lat_channel.cpu()
        sum_lat_channel_mse += lat_channel_mse.cpu()

        # Decode the same latent; the metric stays in normalized [-1, 1] space.
        dit_pred = vae_decode(pipe.vae, pred_lat.to(DTYPE), args.default_temb).float()
        dit_pred = pad_to(dit_pred, orig_F)
        n_px = min(dit_pred.shape[2], target.shape[2], score_F)
        dit_pred, target_cmp = dit_pred[:, :, :n_px], target[:, :, :n_px]

        vd_overall = float(vrms_loss(dit_pred, target_cmp).item())
        vd_channel = vrms_per_channel(dit_pred, target_cmp).tolist()

        per_sample.append({
            "id": sid, "gamma": float(sample["meta"]["gamma"]),
            "vae_only_vrmse": vo_overall, "vae_dit_vrmse": vd_overall,
            "vae_only_per_channel": dict(zip(channel_names, vo_channel)),
            "vae_dit_per_channel": dict(zip(channel_names, vd_channel)),
            "latent_vrmse": lat_vrmse, "latent_rmse": lat_rmse, "latent_mse": lat_mse,
        })
        sum_overall["vae_only"] += vo_overall
        sum_overall["vae_dit"] += vd_overall
        for c in range(n_ch):
            sum_channel["vae_only"][c] += vo_channel[c]
            sum_channel["vae_dit"][c] += vd_channel[c]
        sum_lat_vrmse += lat_vrmse
        sum_lat_rmse += lat_rmse
        sum_lat_mse += lat_mse

        print(f"[{i+1}/{len(val_ids)}] {sid}: vae_only={vo_overall:.5f}  vae+dit={vd_overall:.5f}  "
              f"latent_vrmse={lat_vrmse:.5f}")

        if sid in npz_ids:
            frames = [f for f in args.npz_frames if f < n_px]
            gt_raw = torch.stack([sample[name] for name in channel_names])  # [C, F, H, W]
            npz_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                npz_dir / f"{sid}.npz",
                frames=np.array(frames),
                channel_order=np.array(channel_names),
                gamma=float(sample["meta"]["gamma"]),
                gt_raw=gt_raw[:, frames].numpy().astype(np.float16),
                gt_norm=target_cmp[0][:, frames].cpu().numpy().astype(np.float16),
                vae_only_norm=vae_recon[0][:, frames].cpu().numpy().astype(np.float16),
                vae_dit_norm=dit_pred[0][:, frames].cpu().numpy().astype(np.float16),
                channel_mean=np.array(stats["channel_mean"]),
                channel_std=np.array(stats["channel_std"]),
                normalization_clip=float(stats["normalization_clip"]),
                vae_only_vrmse_chmean=sum(vo_channel) / n_ch,
                vae_dit_vrmse_chmean=sum(vd_channel) / n_ch,
            )

        if sid in vis_ids:
            # Panels are drawn in physical units, straight from the dataset.
            gt_physical = torch.stack([sample[name] for name in channel_names]).unsqueeze(0)  # [1,C,F,H,W]
            gt_physical = pad_to(gt_physical, orig_F)

            vae_recon_physical = denormalize_fields(
                vae_recon, stats["channel_mean"], stats["channel_std"], stats["normalization_clip"],
            )
            dit_pred_physical = denormalize_fields(
                dit_pred, stats["channel_mean"], stats["channel_std"], stats["normalization_clip"],
            )

            for label, pred_physical in (("vae_only", vae_recon_physical), ("vae_dit", dit_pred_physical)):
                save_reconstruction_panels(
                    prediction=pred_physical[0],
                    target=gt_physical[0],
                    sample_id=sid,
                    label=label,
                    frame_numbers=args.frame_numbers,
                    channel_names=channel_names,
                    output_dir=args.out_dir,
                    dpi=args.dpi,
                )

        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "xpu":
            torch.xpu.empty_cache()

    n = len(val_ids)
    mean_lat_channel = (sum_lat_channel / n).tolist()
    ranked = sorted(range(len(mean_lat_channel)), key=lambda c: mean_lat_channel[c], reverse=True)
    summary = {
        "n_samples": n,
        "split": "test" if args.test_h5 is not None else "val",
        "h5": h5_path,
        "checkpoint": str(checkpoint) if checkpoint else "untrained (stock pretrained transformer + random ScalarEmbedding)",
        "vae_checkpoint": str(vae_ckpt),
        "channel_names": channel_names,
        "num_sim_frames": num_sim_frames,
        "num_frames_generated": num_frames,
        "metric_frames": args.metric_frames,
        "num_inference_steps": num_inference_steps,
        "vae_only_vrmse_mean": sum_overall["vae_only"] / n,
        "vae_dit_vrmse_mean": sum_overall["vae_dit"] / n,
        "vae_only_vrmse_per_channel": {name: sum_channel["vae_only"][c] / n for c, name in enumerate(channel_names)},
        "vae_dit_vrmse_per_channel": {name: sum_channel["vae_dit"][c] / n for c, name in enumerate(channel_names)},
        # Channel-mean VRMSE (The Well's convention; the headline metric). The
        # pooled *_vrmse_mean above uses one variance over all channels.
        "vae_only_vrmse_chmean": sum(sum_channel["vae_only"]) / (n_ch * n),
        "vae_dit_vrmse_chmean": sum(sum_channel["vae_dit"]) / (n_ch * n),
        "latent_vrmse_mean": sum_lat_vrmse / n,
        "latent_rmse_mean": sum_lat_rmse / n,
        # Raw (unnormalized) latent MSE, per sim then averaged: the quantity the
        # DiT's flow-matching objective is closest to. Like latent_vrmse it is
        # in this VAE's own latent space -- compare DiTs on the same VAE only.
        "latent_mse_mean": sum_lat_mse / n,
        "latent_vrmse_per_channel": mean_lat_channel,
        "latent_mse_per_channel": (sum_lat_channel_mse / n).tolist(),
        "worst_5_latent_channels": [{"channel": c, "vrmse": mean_lat_channel[c]} for c in ranked[:5]],
        "best_5_latent_channels": [{"channel": c, "vrmse": mean_lat_channel[c]} for c in ranked[-5:]],
        "per_sample": per_sample,
    }
    (args.out_dir / "vrmse_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print(f"n={n} sims")
    print(f"VAE-only  val_vrmse (pixel)  : {summary['vae_only_vrmse_mean']:.5f}")
    print(f"VAE+DiT   val_vrmse (pixel)  : {summary['vae_dit_vrmse_mean']:.5f}")
    delta = summary["vae_dit_vrmse_mean"] - summary["vae_only_vrmse_mean"]
    pct = delta / summary["vae_only_vrmse_mean"] * 100
    print(f"Delta (DiT forecasting cost on top of VAE recon): {delta:+.5f} ({pct:+.1f}%)")
    print(f"VAE-only  val_vrmse_chmean   : {summary['vae_only_vrmse_chmean']:.5f}")
    print(f"VAE+DiT   val_vrmse_chmean   : {summary['vae_dit_vrmse_chmean']:.5f}")
    for name in channel_names:
        vo = summary["vae_only_vrmse_per_channel"][name]
        vd = summary["vae_dit_vrmse_per_channel"][name]
        print(f"  {name:12s} vae_only={vo:.5f}  vae+dit={vd:.5f}  delta={vd - vo:+.5f}")
    print("\nDiT rollout vs VAE-encoder ground truth, LATENT space (isolated from VAE decode):")
    print(f"  latent_vrmse : {summary['latent_vrmse_mean']:.5f}")
    print(f"  latent_rmse  : {summary['latent_rmse_mean']:.5f}")
    print(f"  latent_mse   : {summary['latent_mse_mean']:.5f}")
    print(f"  worst 5 latent channels: {summary['worst_5_latent_channels']}")
    print(f"\nSaved: {args.out_dir / 'vrmse_summary.json'}")
    if vis_ids:
        print(f"Saved GT/Prediction/Residual panels for {len(vis_ids)} sample(s) spanning the gamma "
              f"range, frames {args.frame_numbers}, under:")
        print(f"  {args.out_dir / 'visualizations' / 'vae_only'}/<sample_id>/frame_XXXX.png")
        print(f"  {args.out_dir / 'visualizations' / 'vae_dit'}/<sample_id>/frame_XXXX.png")


if __name__ == "__main__":
    main()

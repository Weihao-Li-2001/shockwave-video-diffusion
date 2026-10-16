# Originally from LTX-Video-Trainer by Lightricks (Apache 2.0).
# https://github.com/Lightricks/LTX-Video-Trainer
"""VAE latent space utilities: encode, decode, pack, and normalise."""

import torch
from diffusers import AutoencoderKLLTXVideo
from torch import Tensor


def pack_latents(
    latents: Tensor,
    spatial_patch_size: int = 1,
    temporal_patch_size: int = 1,
) -> Tensor:
    b, c, f, h, w = latents.shape
    latents = latents.reshape(
        b,
        -1,
        f // temporal_patch_size,
        temporal_patch_size,
        h // spatial_patch_size,
        spatial_patch_size,
        w // spatial_patch_size,
        spatial_patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def encode_video(
    vae: AutoencoderKLLTXVideo,
    image_or_video: Tensor,
    patch_size: int = 1,
    patch_size_t: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor | int]:
    device = device or vae.device

    if image_or_video.ndim == 4:
        image_or_video = image_or_video.unsqueeze(2)
    assert image_or_video.ndim == 5, f"Expected 5D tensor, got {image_or_video.ndim}D tensor"

    image_or_video = image_or_video.to(device=device, dtype=vae.dtype)
    image_or_video = image_or_video.permute(0, 2, 1, 3, 4).contiguous()  # [B,C,F,H,W] -> [B,F,C,H,W]

    latents = vae.encode(image_or_video).latent_dist.mean
    latents = latents.to(dtype=dtype)
    _, _, num_frames, height, width = latents.shape

    sf = float(getattr(vae.config, "scaling_factor", 1.0))
    latents = _normalize_latents(latents, vae.latents_mean, vae.latents_std, scaling_factor=sf)

    latents = pack_latents(latents, patch_size, patch_size_t)
    return {"latents": latents, "num_frames": num_frames, "height": height, "width": width}


def get_rope_scale_factors(fps: float) -> list[float]:
    if fps <= 0:
        raise ValueError("FPS must be a positive number.")

    temporal_compression_ratio = 8.0
    spatial_compression_ratio = 32.0

    return [
        temporal_compression_ratio / fps,
        spatial_compression_ratio,
        spatial_compression_ratio,
    ]


def _normalize_latents(
    latents: Tensor,
    mean: Tensor,
    std: Tensor,
    scaling_factor: float = 1.0,
) -> Tensor:
    mean = mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    std = std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    return (latents - mean) * scaling_factor / std

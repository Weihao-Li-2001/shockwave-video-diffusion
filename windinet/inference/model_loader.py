# Based on LTX-Video-Trainer by Lightricks (Apache 2.0).
# https://github.com/Lightricks/LTX-Video-Trainer
# Modified: removed text encoder/tokenizer loading, added VAE adapter support.
"""Load LTX-Video components (VAE, transformer, scheduler) from HuggingFace or local paths."""

import json
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Union
from urllib.parse import urlparse

import torch
from diffusers import (
    AutoencoderKLLTXVideo,
    FlowMatchEulerDiscreteScheduler,
    LTXVideoTransformer3DModel,
)
from pydantic import BaseModel, ConfigDict

from windinet.paths import HF_CACHE_DIR

HF_MAIN_REPO = "Lightricks/LTX-Video"

# None (huggingface_hub's default cache) unless WINDINET_HF_CACHE is set.
CACHE_DIR = str(HF_CACHE_DIR) if HF_CACHE_DIR is not None else None


class LtxvModelVersion(str, Enum):
    """Available LTX-Video model versions."""

    LTXV_2B_090 = "LTXV_2B_0.9.0"
    LTXV_2B_091 = "LTXV_2B_0.9.1"
    LTXV_2B_095 = "LTXV_2B_0.9.5"
    LTXV_2B_096_DEV = "LTXV_2B_0.9.6_DEV"
    LTXV_2B_096_DISTILLED = "LTXV_2B_0.9.6_DISTILLED"
    LTXV_13B_097_DEV = "LTXV_13B_097_DEV"
    LTXV_13B_097_DISTILLED = "LTXV_13B_097_DISTILLED"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def latest(cls) -> "LtxvModelVersion":
        return cls.LTXV_13B_097_DEV

    @property
    def hf_repo(self) -> str:
        match self:
            case LtxvModelVersion.LTXV_2B_090:
                return "Lightricks/LTX-Video"
            case LtxvModelVersion.LTXV_2B_091:
                return "Lightricks/LTX-Video-0.9.1"
            case LtxvModelVersion.LTXV_2B_095:
                return "Lightricks/LTX-Video-0.9.5"
            case LtxvModelVersion.LTXV_2B_096_DEV:
                raise ValueError("LTXV_2B_096_DEV does not have a HuggingFace repo")
            case LtxvModelVersion.LTXV_2B_096_DISTILLED:
                raise ValueError("LTXV_2B_096_DISTILLED does not have a HuggingFace repo")
            case LtxvModelVersion.LTXV_13B_097_DEV:
                return "Lightricks/LTX-Video-0.9.7-dev"
            case LtxvModelVersion.LTXV_13B_097_DISTILLED:
                return "Lightricks/LTX-Video-0.9.7-distilled"
        raise ValueError(f"Unknown version: {self}")

    @property
    def safetensors_url(self) -> str:
        match self:
            case LtxvModelVersion.LTXV_2B_090:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.safetensors"
            case LtxvModelVersion.LTXV_2B_091:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.1.safetensors"
            case LtxvModelVersion.LTXV_2B_095:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.5.safetensors"
            case LtxvModelVersion.LTXV_2B_096_DEV:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-2b-0.9.6-dev-04-25.safetensors"
            case LtxvModelVersion.LTXV_2B_096_DISTILLED:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-2b-0.9.6-distilled-04-25.safetensors"
            case LtxvModelVersion.LTXV_13B_097_DEV:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-dev.safetensors"
            case LtxvModelVersion.LTXV_13B_097_DISTILLED:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-distilled.safetensors"
        raise ValueError(f"Unknown version: {self}")


ModelSource = Union[str, Path, LtxvModelVersion]


class LtxvModelComponents(BaseModel):
    scheduler: FlowMatchEulerDiscreteScheduler
    vae: Any  # supports AdaptedVAE wrapper
    transformer: LTXVideoTransformer3DModel

    model_config = ConfigDict(arbitrary_types_allowed=True)


def load_scheduler(model_source: ModelSource | None = None) -> FlowMatchEulerDiscreteScheduler:
    """Load the flow-matching scheduler config for `model_source`'s model family.

    2B models read it from the same repo as their VAE, so it is already cached
    wherever the VAE is (important on offline compute nodes).
    """
    source = model_source
    if isinstance(source, str):
        if version := _try_parse_version(source):
            source = version

    repo = LtxvModelVersion.LTXV_13B_097_DEV.hf_repo
    if not isinstance(source, LtxvModelVersion) or source not in (
        LtxvModelVersion.LTXV_13B_097_DEV,
        LtxvModelVersion.LTXV_13B_097_DISTILLED,
    ):
        repo = LtxvModelVersion.LTXV_2B_095.hf_repo

    return FlowMatchEulerDiscreteScheduler.from_pretrained(
        repo,
        subfolder="scheduler",
        cache_dir=CACHE_DIR,
    )


def load_vae(
    source: ModelSource,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> AutoencoderKLLTXVideo:
    if isinstance(source, str):
        if version := _try_parse_version(source):
            source = version

    if isinstance(source, LtxvModelVersion):
        if source in (
            LtxvModelVersion.LTXV_2B_095,
            LtxvModelVersion.LTXV_2B_096_DEV,
            LtxvModelVersion.LTXV_2B_096_DISTILLED,
            LtxvModelVersion.LTXV_13B_097_DEV,
            LtxvModelVersion.LTXV_13B_097_DISTILLED,
        ):
            return AutoencoderKLLTXVideo.from_pretrained(
                LtxvModelVersion.LTXV_2B_095.hf_repo,
                subfolder="vae",
                torch_dtype=dtype,
                cache_dir=CACHE_DIR,
            )
        return AutoencoderKLLTXVideo.from_single_file(
            source.safetensors_url,
            torch_dtype=dtype,
            cache_dir=CACHE_DIR,
        )
    elif isinstance(source, (str, Path)):
        if _is_safetensors_url(source):
            try:
                return AutoencoderKLLTXVideo.from_single_file(source, torch_dtype=dtype, cache_dir=CACHE_DIR)
            except ValueError as e:
                if "Cannot load  because encoder.conv_out.conv.weight" in str(e):
                    return AutoencoderKLLTXVideo.from_pretrained(
                        LtxvModelVersion.LTXV_2B_095.hf_repo,
                        subfolder="vae",
                        torch_dtype=dtype,
                        cache_dir=CACHE_DIR,
                    )
                raise e
        elif _is_huggingface_repo(source):
            return AutoencoderKLLTXVideo.from_pretrained(
                source, subfolder="vae", torch_dtype=dtype, cache_dir=CACHE_DIR,
            )

    raise ValueError(f"Invalid model source: {source}")


def load_inflated_vae(
    source: ModelSource,
    ckpt_path: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
) -> AutoencoderKLLTXVideo:
    """Rebuild a VAE fine-tuned by VaeTrainer in inflate mode: base VAE + widened convs + checkpoint.

    Used to encode CFD fields into latents (preprocessing) and to decode them
    again (inference). The channel count comes from the checkpoint metadata.
    """
    from safetensors import safe_open

    from windinet.vae_adapter import inflate_vae_io_channels, load_inflated_vae_checkpoint

    vae = load_vae(source, dtype=dtype)

    meta = safe_open(str(ckpt_path), framework="pt", device="cpu").metadata() or {}
    inflate_vae_io_channels(vae, n=int(meta.get("n", 4)))
    load_inflated_vae_checkpoint(vae, ckpt_path, device=device, dtype=dtype)
    return vae


def select_vae_env(ckpt_path: str | Path) -> str:
    """Point ``load_vae_with_adapter`` at a fine-tuned VAE via the matching env var.

    The mode (adapter or inflate) comes from the checkpoint's ``format``
    metadata. An env var the caller already set is left alone. Returns the
    name of the env var.
    """
    from safetensors import safe_open

    for existing in ("WINDINET_VAE_ADAPTER_CKPT", "WINDINET_VAE_INFLATE_CKPT"):
        if os.getenv(existing):
            return existing

    metadata = safe_open(str(ckpt_path), framework="pt", device="cpu").metadata() or {}
    fmt = metadata.get("format")
    var = "WINDINET_VAE_INFLATE_CKPT" if fmt == "ltx-inflated-io-v1" else "WINDINET_VAE_ADAPTER_CKPT"
    os.environ[var] = str(ckpt_path)
    return var


def load_vae_with_adapter(
    source: ModelSource,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
) -> Any:
    """Load the base VAE and optionally wrap/grow it for CFD channels.

    Two mutually exclusive env-var driven modes (adapter takes precedence):
      * WINDINET_VAE_ADAPTER_CKPT=/path/to/vae.safetensors -- 1x1 in/out adapters
        around a frozen 3-channel VAE (mode: adapter).
      * WINDINET_VAE_INFLATE_CKPT=/path/to/vae.safetensors -- grown conv_in/conv_out
        reading/writing all channels natively (mode: inflate).
    """
    adapter_ckpt = os.getenv("WINDINET_VAE_ADAPTER_CKPT")
    inflate_ckpt = os.getenv("WINDINET_VAE_INFLATE_CKPT")

    if adapter_ckpt:
        vae = load_vae(source, dtype=dtype)
        from windinet.vae_adapter import load_adapted_vae

        v = os.getenv("WINDINET_VAE_ADAPTER_VERBOSE", "0").strip().lower()
        verbose = v not in ("0", "false", "no", "")
        vae, _ = load_adapted_vae(vae, ckpt_path=adapter_ckpt, device=device, dtype=dtype, verbose=verbose)
        return vae

    if inflate_ckpt:
        return load_inflated_vae(source, inflate_ckpt, dtype=dtype, device=device)

    return load_vae(source, dtype=dtype)


def load_transformer(
    source: ModelSource,
    *,
    dtype: torch.dtype = torch.float32,
) -> LTXVideoTransformer3DModel:
    if isinstance(source, str):
        if version := _try_parse_version(source):
            source = version

    if isinstance(source, LtxvModelVersion):
        if source in (LtxvModelVersion.LTXV_13B_097_DEV, LtxvModelVersion.LTXV_13B_097_DISTILLED):
            return _load_ltxv_13b_transformer(source.safetensors_url, dtype=dtype)
        return LTXVideoTransformer3DModel.from_single_file(
            source.safetensors_url, torch_dtype=dtype, cache_dir=CACHE_DIR,
        )
    elif isinstance(source, (str, Path)):
        if _is_safetensors_url(source):
            try:
                return LTXVideoTransformer3DModel.from_single_file(source, torch_dtype=dtype, cache_dir=CACHE_DIR)
            except ValueError as e:
                if "Cannot load  because time_embed.emb.timestep_embedder.linear_1.bias" in str(e):
                    return _load_ltxv_13b_transformer(source, dtype=dtype)
                raise e
        elif _is_huggingface_repo(source):
            return LTXVideoTransformer3DModel.from_pretrained(
                source, subfolder="transformer", torch_dtype=dtype, cache_dir=CACHE_DIR,
            )

    raise ValueError(f"Invalid model source: {source}")


def load_ltxv_components(
    model_source: ModelSource | None = None,
    *,
    transformer_dtype: torch.dtype = torch.float32,
    vae_dtype: torch.dtype = torch.bfloat16,
) -> LtxvModelComponents:
    if model_source is None:
        model_source = LtxvModelVersion.latest()

    vae = load_vae_with_adapter(model_source, dtype=vae_dtype, device="cpu")

    return LtxvModelComponents(
        scheduler=load_scheduler(model_source),
        vae=vae,
        transformer=load_transformer(model_source, dtype=transformer_dtype),
    )


def _try_parse_version(source: str | Path) -> LtxvModelVersion | None:
    try:
        return LtxvModelVersion(str(source))
    except ValueError:
        return None


def _is_huggingface_repo(source: str | Path) -> bool:
    return "/" in source and not urlparse(source).scheme


def _is_safetensors_url(source: str | Path) -> bool:
    return str(source).endswith(".safetensors")


def _load_ltxv_13b_transformer(safetensors_url: str, *, dtype: torch.dtype) -> LTXVideoTransformer3DModel:
    transformer_13b_config = {
        "_class_name": "LTXVideoTransformer3DModel",
        "_diffusers_version": "0.33.0.dev0",
        "activation_fn": "gelu-approximate",
        "attention_bias": True,
        "attention_head_dim": 128,
        "attention_out_bias": True,
        "caption_channels": 4096,
        "cross_attention_dim": 4096,
        "in_channels": 128,
        "norm_elementwise_affine": False,
        "norm_eps": 1e-06,
        "num_attention_heads": 32,
        "num_layers": 48,
        "out_channels": 128,
        "patch_size": 1,
        "patch_size_t": 1,
        "qk_norm": "rms_norm_across_heads",
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        json.dump(transformer_13b_config, f)
        f.flush()
        return LTXVideoTransformer3DModel.from_single_file(
            safetensors_url, config=f.name, torch_dtype=dtype, cache_dir=CACHE_DIR,
        )

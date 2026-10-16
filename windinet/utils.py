# Originally from LTX-Video-Trainer by Lightricks (Apache 2.0).
# https://github.com/Lightricks/LTX-Video-Trainer
"""Shared utilities: GPU memory, image loading, checkpoint conversion, logging."""

import logging
import os
import subprocess
import sys

import rich
import torch
from safetensors.torch import load_file, save_file


def get_default_device() -> torch.device:
    """Pick the best available accelerator: Intel XPU > NVIDIA CUDA > CPU."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_gpu_memory_gb(device: torch.device) -> float:
    """Get current GPU memory usage in GB."""
    if device.type == "xpu":
        return torch.xpu.memory_allocated(device) / 1024**3

    if device.type == "cuda":
        try:
            device_id = device.index if device.index is not None else 0
            result = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,nounits,noheader",
                    "-i",
                    str(device_id),
                ],
                encoding="utf-8",
            )
            return float(result.strip()) / 1024
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
            logger.error(f"Failed to get GPU memory from nvidia-smi: {e}")
            return torch.cuda.memory_allocated(device) / 1024**3

    return 0.0


def convert_checkpoint(input_path: str, output_path: str, to_comfy: bool = True) -> None:
    """Convert checkpoint format between Diffusers and ComfyUI formats."""
    state_dict = load_file(input_path)

    source_prefix = "transformer." if to_comfy else "diffusion_model."
    target_prefix = "diffusion_model." if to_comfy else "transformer."
    format_name = "ComfyUI" if to_comfy else "Diffusers"

    converted_state_dict = {}
    replaced_count = 0
    for k, v in state_dict.items():
        new_key = k.replace(source_prefix, target_prefix)
        converted_state_dict[new_key] = v
        if new_key != k:
            replaced_count += 1

    if replaced_count == 0:
        rich.print(
            f"No keys were converted. The checkpoint may already be in {format_name} format or "
            f"doesn't contain '{source_prefix}' keys."
        )
        rich.print("[red]Aborting[/red]")
        sys.exit(1)

    save_file(converted_state_dict, output_path)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
from rich.logging import RichHandler

IS_MULTI_GPU = os.environ.get("LOCAL_RANK") is not None
RANK = int(os.environ.get("LOCAL_RANK", "0"))

logging.basicConfig(
    level="INFO",
    format=f"\\[rank {RANK}] %(message)s" if IS_MULTI_GPU else "%(message)s",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_time=False,
            markup=True,
        )
    ],
)

logger = logging.getLogger("windinet")
logger.setLevel(logging.DEBUG)
logger.propagate = True

if RANK != 0:
    logger.setLevel(logging.WARNING)

#!/usr/bin/env python
"""Download the Euler multi-quadrant (euler_mq) dataset from HuggingFace.

Fetches one resolution folder of ``rha6696/euler_mq`` into
``data/euler_mq/<resolution>/{train,test}.h5``, the path the configs expect.

Usage:
    python scripts/download_data.py                 # 256x256_ds (default)
    python scripts/download_data.py 128x128_ds
"""

import typer
from huggingface_hub import snapshot_download

RESOLUTIONS = ["128x128_ds", "256x256_ds", "512x512_orig"]


def main(
    resolution: str = typer.Argument("256x256_ds", help=f"One of: {', '.join(RESOLUTIONS)}."),
    out_dir: str = typer.Option("data/euler_mq", help="Download destination."),
) -> None:
    if resolution not in RESOLUTIONS:
        raise typer.BadParameter(f"resolution must be one of {RESOLUTIONS}, got {resolution!r}")
    snapshot_download(
        repo_id="rha6696/euler_mq",
        repo_type="dataset",
        allow_patterns=f"{resolution}/*",
        local_dir=out_dir,
    )


if __name__ == "__main__":
    typer.run(main)

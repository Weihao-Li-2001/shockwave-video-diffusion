"""
Checkpoint management for WinDiNet.

All scripts that need checkpoints should call ``ensure_checkpoint()``
to resolve paths.
"""

from pathlib import Path


def ensure_checkpoint(name_or_path: str) -> str:
    """Resolve a checkpoint path to an absolute filesystem path.

    Args:
        name_or_path: A filesystem path to an existing checkpoint file.

    Returns:
        Absolute path to the checkpoint file.
    """
    p = Path(name_or_path)
    if p.exists():
        return str(p.resolve())

    raise FileNotFoundError(f"Checkpoint not found: {name_or_path}")

"""Optional relocation of the HuggingFace cache for the base LTX-Video weights.

Set ``WINDINET_HF_CACHE`` to point the HuggingFace hub cache at another
directory (e.g. shared project storage on a cluster whose compute nodes are
offline). Left unset, this module does nothing and ``huggingface_hub`` keeps
its own default (``~/.cache/huggingface``).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: None unless WINDINET_HF_CACHE is set -- see module docstring. None means
#: "leave huggingface_hub's own default alone", not "use pretrained/hub".
HF_CACHE_DIR = Path(os.environ["WINDINET_HF_CACHE"]) if os.environ.get("WINDINET_HF_CACHE") else None


def use_repo_hf_cache() -> Path | None:
    """Redirect huggingface_hub to :data:`HF_CACHE_DIR` for this process and children,
    if ``WINDINET_HF_CACHE`` is set. No-op otherwise (returns ``None``).

    Call sites pass ``cache_dir=`` explicitly, so this only covers the paths we
    do not control (diffusers internals, subprocesses). It runs on ``import
    windinet``, which is early enough as long as ``windinet`` is imported before
    ``huggingface_hub`` reads its constants.
    """
    if HF_CACHE_DIR is None:
        return None
    cache = str(HF_CACHE_DIR)
    os.environ["HF_HUB_CACHE"] = cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache  # legacy alias, still read by older deps
    return HF_CACHE_DIR

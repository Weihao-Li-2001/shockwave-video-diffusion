"""WinDiNet: pretrained video diffusion models repurposed as CFD surrogates (ShockWaveNet)."""

from windinet.paths import use_repo_hf_cache

# Must run before diffusers/huggingface_hub are imported: they snapshot the cache
# location into module constants at import time.
use_repo_hf_cache()

__version__ = "0.1.0"

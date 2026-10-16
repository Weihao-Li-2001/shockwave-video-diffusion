"""
ShockWave (Euler multi-quadrant) CFD simulations from HDF5, and their normalisation.

Expected layout of an HDF5 file (e.g. train.h5)::

    <sample_id>/            one simulation; gamma is encoded in the id
        density       [T, H, W]
        momentum_x    [T, H, W]
        momentum_y    [T, H, W]
        pressure      [T, H, W]

:func:`build_shockwave_video` turns a sample into the normalized
[B, C, F, H, W] tensor the VAE consumes.
"""

import json
import os
import random
import time
from pathlib import Path

import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset


def parse_gamma(sid: str) -> float:
    """Physical adiabatic index encoded in a sample id, e.g. "0000_gamma1.22" -> 1.22."""
    return float(sid.split("gamma")[1])


def pick_gamma_spread_ids(ids: list[str], n: int) -> set[str]:
    """N sample ids spanning the gamma range of `ids`: min/median/max for n=3,
    evenly-spaced quantiles otherwise. Deterministic, so every eval script
    that calls it on the same id list shows the same sims in its figures."""
    if n <= 0:
        return set()
    gamma_order = sorted(range(len(ids)), key=lambda k: parse_gamma(ids[k]))
    n = min(n, len(gamma_order))
    if n == 1:
        picks = [gamma_order[len(gamma_order) // 2]]
    else:
        picks = [gamma_order[round(i * (len(gamma_order) - 1) / (n - 1))] for i in range(n)]
    return {ids[i] for i in picks}


class ShockWaveDataset(Dataset):
    """One item per simulation: the four physical fields [T, H, W], ``meta`` (gamma) and ``id``.

    Robust to shared network filesystems under multi-process training: file
    access is staggered per rank and retried with jittered backoff.
    """

    def __init__(
        self,
        h5_path: str | Path,
        num_sim_frames: int | None = None,
    ):
        self.h5_path = Path(h5_path)
        self.file = None

        if not self.h5_path.is_file():
            raise FileNotFoundError(f"Shockwave HDF5 file not found: {self.h5_path}")

        time.sleep(self._rank_stagger_delay())
        with self._open(self.h5_path) as f:
            self.ids = sorted(list(f.keys()))
            self._prewarm_ids = self._pick_prewarm_ids(f, self.ids)

        self.num_sim_frames = num_sim_frames

    @staticmethod
    def _pick_prewarm_ids(f: h5py.File, ids: list[str]) -> list[str]:
        """One sample id per distinct external-link target file.

        Samples in train.h5 are external links into per-gamma shard files, and
        the first access to a link opens its shard. Touching one sample per
        shard at startup (while ranks are staggered) keeps those opens out of
        the training loop.
        """
        seen_targets = set()
        prewarm_ids = []
        for sid in ids:
            link = f.get(sid, getlink=True)
            target = getattr(link, "filename", None)
            if target is None:
                # Not an external link (e.g. a self-contained h5 file) --
                # nothing to pre-resolve.
                continue
            if target not in seen_targets:
                seen_targets.add(target)
                prewarm_ids.append(sid)
        return prewarm_ids


    def __len__(self):
        return len(self.ids)


    @staticmethod
    def _rank_stagger_delay(spread: float = 0.3) -> float:
        """Per-rank delay so all ranks do not hit the filesystem at the same instant (0 if not distributed)."""
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        return rank * spread

    @staticmethod
    def _backoff_sleep(attempt: int) -> None:
        """Exponential backoff with random jitter, so retrying ranks do not stay in lockstep."""
        base = 0.5 * (2 ** attempt)
        time.sleep(base + random.uniform(0, base))

    @staticmethod
    def _open(path: Path, retries: int = 8) -> h5py.File:
        """Open the HDF5 file, retrying with backoff on transient filesystem errors.

        HDF5_EXT_PREFIX is set to the file's directory: some HDF5 builds resolve
        the per-gamma external links relative to the working directory instead.
        """
        os.environ["HDF5_EXT_PREFIX"] = str(path.parent)
        for attempt in range(retries):
            try:
                try:
                    return h5py.File(path, "r", locking=False)
                except TypeError:
                    # Installed h5py/libhdf5 too old to support the `locking` kwarg.
                    return h5py.File(path, "r")
            except OSError:
                if attempt == retries - 1:
                    raise
                ShockWaveDataset._backoff_sleep(attempt)
        raise AssertionError("unreachable")

    def _init_file(self):
        """
        Lazy open h5 file.
        Important for DataLoader workers.
        """
        if self.file is None:
            time.sleep(self._rank_stagger_delay())
            self.file = self._open(self.h5_path)

            for sid in self._prewarm_ids:
                self._get_group(sid)

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def _get_group(self, sid: str, retries: int = 8):
        """Look up one simulation, reopening the file and retrying on transient failures.

        Under multi-process training on a network filesystem, lookups can fail
        transiently ("can't open file") although the key exists.
        """
        for attempt in range(retries):
            try:
                return self.file[sid]
            except KeyError:
                if attempt == retries - 1:
                    raise
                self._backoff_sleep(attempt)
                self.file.close()
                self.file = self._open(self.h5_path)
        raise AssertionError("unreachable")

    def __getitem__(self, idx):

        self._init_file()

        sid = self.ids[idx]

        sample = self._get_group(sid)


        density = torch.from_numpy(
            sample["density"][:]
        ).float().squeeze(1)


        momentum_x = torch.from_numpy(
            sample["momentum_x"][:]
        ).float().squeeze(1)


        momentum_y = torch.from_numpy(
            sample["momentum_y"][:]
        ).float().squeeze(1)


        pressure = torch.from_numpy(
            sample["pressure"][:]
        ).float().squeeze(1)


        if self.num_sim_frames is not None:

            density = density[:self.num_sim_frames]

            momentum_x = momentum_x[:self.num_sim_frames]

            momentum_y = momentum_y[:self.num_sim_frames]

            pressure = pressure[:self.num_sim_frames]


        gamma = torch.tensor(
            parse_gamma(sid),
            dtype=torch.float32
        )


        return {
            "density": density,
            "momentum_x": momentum_x,
            "momentum_y": momentum_y,
            "pressure": pressure,
            "meta": {
                "gamma": gamma.item()
            },
            "id": sid,
        }



def build_shockwave_video(
    sample: dict[str, Tensor],
    device: torch.device,
    channel_mean: list[float] | None = None,
    channel_std: list[float] | None = None,
    normalization_clip: float = 5.0,
    channel_order: list[str] | None = None,
) -> Tensor:
    """Build a shockwave video [B, C, F, H, W], C = len(channel_order) (default 4).

    Default channel layout:
        0 - density
        1 - momentum_x
        2 - momentum_y
        3 - pressure

    ``channel_order`` selects and orders the stacked fields (a permutation or
    subset of CHANNEL_NAMES), aligned with ``channel_mean``/``channel_std``.
    Each channel is normalized as clip((x - mean) / (std * clip), -1, 1).

    Frame 0 is the physical initial condition (no artificial conditioning
    frame is prepended). F is padded to 8n+1 as the LTX-Video VAE requires.
    """

    order = channel_order or CHANNEL_NAMES

    fields = {name: sample[name] for name in order}

    # Ensure batch dimension
    for name, tensor in fields.items():
        if tensor.ndim == 3:
            fields[name] = tensor.unsqueeze(0)

    # each field: [B, T, H, W]
    x = torch.stack([fields[name] for name in order], dim=1)  # [B, C, F, H, W]

    if channel_mean is not None and channel_std is not None:
        mean = x.new_tensor(channel_mean).view(1, len(order), 1, 1, 1)
        std = x.new_tensor(channel_std).view(1, len(order), 1, 1, 1)
        x = ((x - mean) / (std * normalization_clip)).clamp(-1.0, 1.0)

    x = pad_frames_8n1(x)

    return x.to(
        device=device,
        dtype=torch.float32,
    ).contiguous()


CHANNEL_NAMES = ["density", "momentum_x", "momentum_y", "pressure"]


def load_channel_normalization(source: str | Path) -> dict:
    """Read the channel normalization stats that a VAE / latent set was built with.

    Accepts a ``normalization.json`` written by ``preprocess_dataset.py`` or a
    VAE run's ``training_config.yaml``. Encoding, decoding and preprocessing
    must all use the same numbers, or the latents / decoded fields are silently
    wrong.

    Returns a dict with ``channel_names``, ``channel_mean``, ``channel_std`` and
    ``normalization_clip``. ``channel_names`` (the stacked field order) is the
    json's ``channel_names`` or the config's ``channel_order``, falling back to
    the 4-channel CHANNEL_NAMES for sources that predate recording it.
    """
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Normalization source not found: {path}")

    if path.suffix == ".json":
        payload = json.loads(path.read_text())
    else:
        import yaml

        # A dumped training_config.yaml carries python-tagged values (e.g. the
        # model-source enum). Ignore unknown tags rather than reaching for
        # unsafe_load, which would execute whatever the file names.
        class _TolerantLoader(yaml.SafeLoader):
            pass

        _TolerantLoader.add_multi_constructor("", lambda loader, suffix, node: None)
        payload = (yaml.load(path.read_text(), Loader=_TolerantLoader) or {}).get("data", {})

    mean, std = payload.get("channel_mean"), payload.get("channel_std")
    if not mean or not std:
        raise ValueError(f"{path} has no channel_mean / channel_std")
    names = list(payload.get("channel_names") or payload.get("channel_order") or CHANNEL_NAMES)
    if len(names) != len(mean) or len(std) != len(mean):
        raise ValueError(f"{path}: {len(names)} channel names vs {len(mean)} means / {len(std)} stds")
    return {
        "channel_names": names,
        "channel_mean": list(mean),
        "channel_std": list(std),
        "normalization_clip": float(payload.get("normalization_clip", 5.0)),
    }


def normalize_fields(
    x: Tensor,
    channel_mean: list[float],
    channel_std: list[float],
    normalization_clip: float,
) -> Tensor:
    """Normalize physical fields [B, C, ...] to [-1, 1]; inverse of denormalize_fields."""
    shape = (1, len(channel_mean)) + (1,) * (x.ndim - 2)
    mean = x.new_tensor(channel_mean).view(shape)
    scale = x.new_tensor(channel_std).view(shape) * normalization_clip
    return ((x - mean) / scale).clamp(-1.0, 1.0)


def extract_scalars(
    meta: dict,
    scalar_names: list[str] | None = None,
) -> dict[str, float]:
    """Extract scalar conditioning values from a sample's metadata.

    Args:
        meta: metadata dictionary.
        scalar_names: names to extract. Defaults to ["gamma"].

    Returns:
        Dictionary mapping scalar names to float values.
    """
    if scalar_names is None:
        scalar_names = ["gamma"]

    scalars = {}

    for name in scalar_names:
        scalars[name] = float(meta[name])

    return scalars

def pad_frames_8n1(x: Tensor) -> Tensor:
    """Pad temporal dimension so F = 8n + 1 (required by LTX-Video VAE).

    Pads by repeating the last frame.

    Args:
        x: [B, C, F, H, W] tensor.

    Returns:
        Padded tensor with F' = 8n + 1 >= F.
    """
    orig_F = x.shape[2]
    target = ((orig_F - 1 + 7) // 8) * 8 + 1
    if target == orig_F:
        return x
    pad_frames = target - orig_F
    last = x[:, :, -1:]
    return torch.cat([x, last.expand(-1, -1, pad_frames, -1, -1)], dim=2)

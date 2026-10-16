# checked 

# Based on LTX-Video-Trainer by Lightricks (Apache 2.0).
# https://github.com/Lightricks/LTX-Video-Trainer
# Modified: removed text conditioning data sources, scalar-only.
"""Dataset for loading precomputed latent tensors and scalar conditioning."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from windinet.utils import logger

PRECOMPUTED_DIR_NAME = ".precomputed"


class PrecomputedDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        data_sources: dict[str, str] | list[str] | None = None,
    ) -> None:
        """
        Generic dataset for loading precomputed data from multiple sources.

        Args:
            data_root: Root directory containing preprocessed data
            data_sources: Either:
                         - Dict mapping directory names to output keys
                         - List of directory names (keys will equal values)
                         - None (defaults to ["latents", "scalars"])
        """
        super().__init__()

        self.data_root = self._setup_data_root(data_root)
        self.data_sources = self._normalize_data_sources(data_sources)
        self.source_paths = self._setup_source_paths()
        self.sample_files = self._discover_samples()
        self._validate_setup()

    @staticmethod
    def _setup_data_root(data_root: str) -> Path:
        data_root = Path(data_root)
        if not data_root.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {data_root}")
        if (data_root / PRECOMPUTED_DIR_NAME).exists():
            data_root = data_root / PRECOMPUTED_DIR_NAME
        return data_root

    @staticmethod
    def _normalize_data_sources(data_sources: dict[str, str] | list[str] | None) -> dict[str, str]:
        if data_sources is None:
            return {"latents": "latent_conditions", "scalars": "scalars"}
        elif isinstance(data_sources, list):
            return {source: source for source in data_sources}
        elif isinstance(data_sources, dict):
            return data_sources.copy()
        else:
            raise TypeError(f"data_sources must be dict, list, or None, got {type(data_sources)}")

    def _setup_source_paths(self) -> dict[str, Path]:
        source_paths = {}
        for dir_name in self.data_sources:
            source_path = self.data_root / dir_name
            source_paths[dir_name] = source_path
            if not source_path.exists():
                raise FileNotFoundError(f"Required {dir_name} directory does not exist: {source_path}")
        return source_paths

    def _discover_samples(self) -> dict[str, list[Path]]:
        data_key = "latents" if "latents" in self.data_sources else next(iter(self.data_sources.keys()))
        data_path = self.source_paths[data_key]
        data_files = list(data_path.glob("**/*.pt"))

        if not data_files:
            raise ValueError(f"No data files found in {data_path}")

        sample_files = {output_key: [] for output_key in self.data_sources.values()}

        for data_file in data_files:
            rel_path = data_file.relative_to(data_path)
            if self._all_source_files_exist(data_file, rel_path):
                self._fill_sample_data_files(data_file, rel_path, sample_files)

        return sample_files

    def _all_source_files_exist(self, data_file: Path, rel_path: Path) -> bool:
        for dir_name in self.data_sources:
            expected_path = self.source_paths[dir_name] / rel_path
            if not expected_path.exists():
                logger.warning(
                    f"No matching {dir_name} file found for: {data_file.name} (expected in: {expected_path})"
                )
                return False
        return True

    def _fill_sample_data_files(self, data_file: Path, rel_path: Path, sample_files: dict[str, list[Path]]) -> None:
        for dir_name, output_key in self.data_sources.items():
            sample_files[output_key].append(rel_path)

    def _validate_setup(self) -> None:
        if not self.sample_files:
            raise ValueError("No valid samples found - all data sources must have matching files")
        sample_counts = {key: len(files) for key, files in self.sample_files.items()}
        if len(set(sample_counts.values())) > 1:
            raise ValueError(f"Mismatched sample counts across sources: {sample_counts}")

    def __len__(self) -> int:
        first_key = next(iter(self.sample_files.keys()))
        return len(self.sample_files[first_key])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = {}
        for dir_name, output_key in self.data_sources.items():
            source_path = self.source_paths[dir_name]
            file_rel_path = self.sample_files[output_key][index]
            file_path = source_path / file_rel_path

            try:
                data = torch.load(file_path, map_location="cpu", weights_only=True)
                result[output_key] = data
            except Exception as e:
                raise RuntimeError(f"Failed to load {output_key} from {file_path}: {e}") from e

        result["idx"] = index
        return result

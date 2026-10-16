# Originally from LTX-Video-Trainer by Lightricks (Apache 2.0).
# https://github.com/Lightricks/LTX-Video-Trainer
"""Timestep sampling strategies for flow-matching training."""

import torch


class TimestepSampler:
    """Base class for timestep samplers."""

    def sample(self, batch_size: int, seq_length: int | None = None, device: torch.device = None) -> torch.Tensor:
        raise NotImplementedError

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class UniformTimestepSampler(TimestepSampler):
    """Samples timesteps uniformly between min_value and max_value."""

    def __init__(self, min_value: float = 0.0, max_value: float = 1.0):
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, batch_size: int, seq_length: int | None = None, device: torch.device = None) -> torch.Tensor:
        return torch.rand(batch_size, device=device) * (self.max_value - self.min_value) + self.min_value

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")
        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size, device=batch.device)


class ShiftedLogitNormalTimestepSampler:
    """Samples timesteps from a shifted logit-normal distribution."""

    def __init__(self, std: float = 1.0):
        self.std = std

    def sample(self, batch_size: int, seq_length: int, device: torch.device = None) -> torch.Tensor:
        shift = self._get_shift_for_sequence_length(seq_length)
        normal_samples = torch.randn((batch_size,), device=device) * self.std + shift
        timesteps = torch.sigmoid(normal_samples)
        return timesteps

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")
        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size, seq_length, device=batch.device)

    @staticmethod
    def _get_shift_for_sequence_length(
        seq_length: int,
        min_tokens: int = 1024,
        max_tokens: int = 4096,
        min_shift: float = 0.95,
        max_shift: float = 2.05,
    ) -> float:
        m = (max_shift - min_shift) / (max_tokens - min_tokens)
        b = min_shift - m * min_tokens
        shift = m * seq_length + b
        return shift


SAMPLERS = {
    "uniform": UniformTimestepSampler,
    "shifted_logit_normal": ShiftedLogitNormalTimestepSampler,
}

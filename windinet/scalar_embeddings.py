"""
Scalar conditioning: physical parameters (here gamma) as transformer tokens.

Each scalar is normalized to [0, 1] with its configured range, Fourier-encoded
and mapped by an MLP to ``num_tokens_per_scalar`` tokens of the transformer's
hidden size. These tokens take the place of the text-encoder output, so the
DiT's cross-attention attends to the physical parameters instead of a prompt.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from windinet.config import ScalarConditioningConfig


class FourierFeatures(nn.Module):
    """Fourier feature encoding for continuous scalar values."""

    def __init__(self, num_features: int, scale: float = 1.0):
        super().__init__()
        self.num_features = num_features
        self.register_buffer(
            "frequencies",
            scale * torch.pow(2.0, torch.linspace(0, num_features - 1, num_features) / num_features * 4),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Compute in fp32 (see ScalarEmbedding.forward).
        x = x.unsqueeze(-1).float()
        x_f32 = x
        freq_f32 = self.frequencies.float()
        freq_x = 2 * math.pi * x_f32 * freq_f32
        result = torch.cat([torch.sin(freq_x), torch.cos(freq_x)], dim=-1)
        return result.to(dtype=self.frequencies.dtype)


class ScalarEmbedding(nn.Module):
    """Scalars [B, S] -> conditioning tokens [B, S * num_tokens_per_scalar, embedding_dim]."""

    def __init__(self, config: ScalarConditioningConfig):
        super().__init__()
        self.config = config
        self.scalar_names = config.scalar_names
        self.num_scalars = len(config.scalar_names)

        ranges_min = []
        ranges_max = []
        for name in config.scalar_names:
            min_val, max_val = config.scalar_ranges[name]
            ranges_min.append(min_val)
            ranges_max.append(max_val)

        self.register_buffer("ranges_min", torch.tensor(ranges_min))
        self.register_buffer("ranges_max", torch.tensor(ranges_max))

        self.fourier = FourierFeatures(config.fourier_features)
        fourier_dim = 2 * config.fourier_features

        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, config.mlp_hidden_dim),
            nn.LayerNorm(config.mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, config.mlp_hidden_dim),
            nn.LayerNorm(config.mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, config.num_tokens_per_scalar * config.embedding_dim),
        )

        self._init_weights()

    def _init_weights(self):
        layers = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)]
        for i, module in enumerate(layers):
            if i == len(layers) - 1:
                nn.init.normal_(module.weight, std=0.001)
            else:
                nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def normalize(self, scalars: Tensor) -> Tensor:
        scalars_f32 = scalars.float()
        ranges_min_f32 = self.ranges_min.float()
        ranges_max_f32 = self.ranges_max.float()
        normalized = (scalars_f32 - ranges_min_f32) / (ranges_max_f32 - ranges_min_f32 + 1e-8)
        return normalized.clamp(0.0, 1.0)

    def forward(self, scalars: Tensor) -> Tensor:
        # Always fp32, even under the trainer's bf16 autocast: NaN embeddings
        # were observed in this module under bf16, most likely from the
        # high-frequency Fourier features. The module is tiny, so fp32 is free.
        with torch.autocast(device_type=scalars.device.type, enabled=False):
            batch_size = scalars.shape[0]
            normalized = self.normalize(scalars)
            fourier_features = self.fourier(normalized).float()
            embeddings = self.mlp(fourier_features.to(self.mlp[0].weight.dtype))
            embeddings = embeddings.view(
                batch_size,
                self.num_scalars * self.config.num_tokens_per_scalar,
                self.config.embedding_dim,
            )
        return embeddings

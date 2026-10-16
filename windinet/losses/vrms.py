"""
Variance-normalized RMSE (VRMS) loss.

The evaluation metric (val_vrmse, and the test-set VRMSE in RESULTS.md), also
usable as a differentiable loss. Normalizing by the target's variance keeps a
low-variance channel from being "cheap" the way plain RMSE would let it be.

Expected input:
    pred, target: [B, C, T, H, W]
"""

import torch


def vrms_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute variance-normalized RMSE, per sample, averaged over the batch.

    VRMSE = sqrt(mean((pred - target)^2) / var(target))

    Args:
        pred:
            Predicted fields.
            Shape [B, C, T, H, W]

        target:
            Ground truth fields.
            Shape [B, C, T, H, W]

        eps:
            Numerical stability term.

    Returns:
        Scalar VRMS loss.
    """

    dims = tuple(range(1, pred.dim()))
    diff = pred - target
    mse = diff.square().mean(dim=dims)
    variance = target.var(dim=dims, unbiased=False)

    return torch.sqrt(mse / (variance + eps)).mean()


def vrms_per_channel(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Same formula as vrms_loss, broken down per channel instead of collapsed.

    Reduces over every dim except batch and channel (instead of every dim
    but batch), so each physical field gets its own variance-normalized RMSE
    -- lets e.g. "is pressure specifically worse than momentum_x" be read
    off directly instead of only ever seeing the all-channels-mixed scalar.

    Args:
        pred, target: [B, C, T, H, W]
        eps: numerical stability term.

    Returns:
        [C] tensor, one VRMSE per channel, averaged over the batch.
    """
    dims = tuple(range(2, pred.dim()))
    diff = pred - target
    mse = diff.square().mean(dim=dims)
    variance = target.var(dim=dims, unbiased=False)

    return torch.sqrt(mse / (variance + eps)).mean(dim=0)

"""
Root Mean Squared Error loss.

Expected input:
    pred, target: [B, C, T, H, W]

Returns:
    scalar RMSE loss
"""

import torch
import torch.nn.functional as F


def rmse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute Root Mean Squared Error.

    RMSE = sqrt(mean((pred - target)^2))

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
        Scalar RMSE loss.
    """

    mse = F.mse_loss(
        pred,
        target,
        reduction="mean",
    )

    return torch.sqrt(mse + eps)
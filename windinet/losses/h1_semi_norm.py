"""
H1 semi-norm loss.

Measures the difference between spatial gradients
of prediction and target.

Expected input:
    pred, target: [B, C, T, H, W]

The H1 seminorm is:

    |u|_H1^2 = ||∇u||^2

Here we minimize:

    ||∇u_pred - ∇u_target||^2
"""

import torch
import torch.nn.functional as F


def spatial_gradients(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute first-order spatial gradients.

    Args:
        x:
            Tensor with shape [B, C, T, H, W]

    Returns:
        dx:
            Gradient along width direction.
        dy:
            Gradient along height direction.
    """

    # Difference along W dimension
    dx = x[..., :, 1:] - x[..., :, :-1]

    # Difference along H dimension
    dy = x[..., 1:, :] - x[..., :-1, :]

    # Keep original tensor size
    dx = F.pad(
        dx,
        (0, 1, 0, 0),
    )

    dy = F.pad(
        dy,
        (0, 0, 0, 1),
    )

    return dx, dy


def h1_seminorm_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute H1 semi-norm loss.

    Penalizes differences between spatial derivatives.

    Args:
        pred:
            Predicted fields.
            Shape [B, C, T, H, W]

        target:
            Ground truth fields.
            Shape [B, C, T, H, W]

    Returns:
        Scalar H1 loss.
    """

    pred_dx, pred_dy = spatial_gradients(pred)
    target_dx, target_dy = spatial_gradients(target)

    loss_dx = F.mse_loss(
        pred_dx,
        target_dx,
    )

    loss_dy = F.mse_loss(
        pred_dy,
        target_dy,
    )

    return loss_dx + loss_dy
"""Computes all VAE loss components in one call; VaeTrainer applies the weights."""


import torch

from .rmse import rmse_loss
from .h1_semi_norm import h1_seminorm_loss
from .mlw import mlw_loss
from .vrms import vrms_loss
from .kl_divergence import kl_divergence_loss


def reconstruction_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    ssim_module,
    wavelet: str = "db2",
    spatial_level: int | None = None,
    temporal_level: int | None = None,
    mlw_beta: float = 10.0,
    mlw_eps: float = 1e-6,
    latent_mean: torch.Tensor | None = None,
    latent_logvar: torch.Tensor | None = None,
    compute_mlw: bool = True,
    kl_reduction: str = "mean",
) -> dict[str, torch.Tensor]:
    """Every VAE loss term that depends on (pred, target), plus KL.

    Returns a dict with:
        rmse   pixel-wise error
        h1     error of the spatial gradients (H1 semi-norm)
        ssim   1 - SSIM
        mlw    multi-level wavelet error (zero unless compute_mlw, as it is
               the only expensive term)
        vrms   variance-normalized RMSE (the val_vrmse metric)
        kl     KL of the posterior from N(0, I); only if latent_mean and
               latent_logvar (raw, before normalisation) are given;
               kl_reduction as in kl_divergence_loss

    pred/target are [B, C, T, H, W]. Which terms enter the objective is decided
    by the config's loss weights, not here.
    """

    losses = {}

    # ------------------------------------
    # Pixel reconstruction
    # ------------------------------------

    losses["rmse"] = rmse_loss(
        pred,
        target,
    )


    # ------------------------------------
    # Gradient / curvature consistency
    # ------------------------------------

    losses["h1"] = h1_seminorm_loss(
        pred,
        target,
    )



    # ------------------------------------
    # Structural similarity / correlation
    # ------------------------------------

    losses["ssim"] = ssim_module(
        pred,
        target,
    )



    # ------------------------------------
    # Frequency / interface preservation
    # ------------------------------------

    losses["mlw"] = (
        mlw_loss(
            pred,
            target,
            wavelet=wavelet,
            beta=mlw_beta,
            eps=mlw_eps,
            spatial_level=spatial_level,
            temporal_level=temporal_level,
        )
        if compute_mlw
        else pred.new_zeros(())
    )


    # ------------------------------------
    # Variance-normalized error
    # ------------------------------------

    losses["vrms"] = vrms_loss(
        pred,
        target,
    )


    # ------------------------------------
    # Latent regularization (opt-in: needs the encoder distribution)
    # ------------------------------------

    if latent_mean is not None and latent_logvar is not None:
        losses["kl"] = kl_divergence_loss(
            latent_mean,
            latent_logvar,
            reduction=kl_reduction,
        )


    return losses
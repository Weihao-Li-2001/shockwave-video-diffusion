"""
VAE training losses on [B, C, T, H, W] fields, and latent regularisers.

:func:`reconstruction_losses` computes every term; VaeTrainer combines them
with the fixed weights from the config.
"""


from .rmse import rmse_loss
from .h1_semi_norm import h1_seminorm_loss
from .ssim import SSIMLoss
from .mlw import mlw_loss
from .vrms import vrms_loss, vrms_per_channel
from .kl_divergence import kl_divergence_loss
from .sliced_wasserstein import SlicedWassersteinPrior
from .reconstruction import reconstruction_losses


__all__ = [
    "rmse_loss",
    "h1_seminorm_loss",
    "SSIMLoss",
    "mlw_loss",
    "vrms_loss",
    "vrms_per_channel",
    "kl_divergence_loss",
    "SlicedWassersteinPrior",
    "reconstruction_losses",
]
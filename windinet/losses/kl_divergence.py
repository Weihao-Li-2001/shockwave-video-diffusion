"""
KL divergence of the encoder posterior from the prior, KL(N(mean, exp(logvar)) || N(0, I)).

Takes the raw encoder posterior (before the latents_mean / latents_std /
scaling_factor normalisation), [B, C, T, H, W].

By default averaged over latent elements, like the reconstruction terms are
averaged over pixels, so a weight means the same at every resolution. The
summed variant (used for the KL runs in RESULTS.md) with weight w equals the
averaged one with weight w * N, N = latent elements per sample
(128 x 14 x 8 x 8 = 114,688 for a 101-frame 256x256 simulation).
"""

import torch


def kl_divergence_loss(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """KL(N(mean, exp(logvar)) || N(0, I)) for [B, ...] posteriors.

    reduction="mean": averaged over every latent element and the batch.
    reduction="sum":  summed over each sample's latent elements, averaged over
                      the batch (the convention of the KL runs in RESULTS.md).

    Reduces over all non-batch dimensions itself; diffusers'
    DiagonalGaussianDistribution.kl() assumes 4D image latents and would leave
    one axis of the 5D video latents unreduced.
    """
    # float32: exp(logvar) in bf16 loses most of its precision
    mean = mean.float()
    logvar = logvar.float()
    kl = mean.square() + logvar.exp() - logvar - 1.0
    if reduction == "mean":
        return 0.5 * torch.mean(kl)
    if reduction == "sum":
        return (0.5 * torch.sum(kl, dim=tuple(range(1, kl.dim())))).mean()
    raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")

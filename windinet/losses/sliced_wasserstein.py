"""
Sliced-Wasserstein prior loss on the aggregate posterior (WAE-style).

Unlike KL(q(z|x) || N(0, I)), which pulls every sample's posterior towards
the prior (forcing per-sample noise and shrinking the mean), this matches
the AGGREGATE latent distribution q(z) = E_x q(z|x) to N(0, I), as in
Wasserstein autoencoders (Tolstikhin et al. 2018) with the sliced
Wasserstein distance of SWAE (Kolouri et al. 2019). The encoder stays
deterministic (its posterior mean, which is also what DiT preprocessing
encodes), and only the shape of the latent cloud is regularized: the
distribution the DiT's flow matching interpolates against N(0, I) noise.

Points: every latent token (one C-dim vector per latent t, h, w position)
of the rescaled latents -- 13 x 8 x 8 = 832 tokens of 128 channels per
256x256 / 97-frame sim. A batch of one sim per GPU is too small a sample
of q(z), so each call also ranks the current tokens against a bank of the
last `bank_size` sims' tokens (detached; on this rank only). Only the
current tokens get a gradient.

Estimator, for L random unit directions theta_l:
    p_il = <z_i, theta_l>                     (projected current + bank tokens)
    r_il = rank of p_il among all N projected tokens along theta_l
    q_il = Phi^-1((r_il + 0.5) / N)           (N(0, I) projects to N(0, 1),
                                               so the target quantiles are exact,
                                               no prior samples needed)
    loss = mean over current i and l of (p_il - q_il)^2
This is the one-dimensional optimal-transport cost of the current tokens
inside the empirical aggregate, i.e. a per-point SW_2^2 against N(0, I);
its value does not depend on the bank size or resolution.
"""

from collections import deque

import torch


class SlicedWassersteinPrior:
    """Stateful SW_2^2(aggregate latents, N(0, I)) estimator; see module docstring."""

    def __init__(
        self,
        num_projections: int,
        bank_size: int,
        device: torch.device,
        seed: int,
    ) -> None:
        self.num_projections = num_projections
        self._bank: deque[torch.Tensor] = deque(maxlen=bank_size) if bank_size > 0 else deque(maxlen=0)
        # Own generator: drawing the directions never touches the global RNG,
        # so runs that only log this term (weight 0) train bit-identically.
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(seed)

    @staticmethod
    def _tokens(latents: torch.Tensor) -> torch.Tensor:
        """[B, C, T, H, W] -> [B * T * H * W, C] in float32."""
        c = latents.shape[1]
        return latents.float().movedim(1, -1).reshape(-1, c)

    def __call__(self, latents: torch.Tensor, update_bank: bool = True) -> torch.Tensor:
        """Loss for this batch's latents; appends them (detached) to the bank if update_bank."""
        current = self._tokens(latents)
        n_cur, c = current.shape
        points = torch.cat([current, *self._bank], dim=0) if self._bank else current
        n = points.shape[0]

        theta = torch.randn(c, self.num_projections, device=current.device, generator=self._generator)
        theta = theta / theta.norm(dim=0, keepdim=True)
        proj = points @ theta  # [N, L]

        with torch.no_grad():
            order = proj.argsort(dim=0)
            ranks = torch.empty_like(order)
            ranks.scatter_(0, order, torch.arange(n, device=proj.device).unsqueeze(1).expand(-1, self.num_projections))
            target = torch.special.ndtri((ranks[:n_cur].float() + 0.5) / n)

        loss = (proj[:n_cur] - target).square().mean()
        if update_bank and self._bank.maxlen:
            self._bank.append(current.detach())
        return loss

# Results

All numbers are on the **held-out test set** (500 simulations, a separate file
from training, 10 γ values × 50 sims) at 256². Configs for every row are in
[`configs/vae/`](configs/vae/) and [`configs/dit/`](configs/dit/).

## Setup

- **Base model:** LTX-Video 2B (v0.9.6 dev). Its causal video VAE compresses
  32× in space and 8× in time into 128 latent channels; a 101-frame 256²
  simulation (padded to 105 frames) becomes a 14×8×8×128 latent.
- **Channel adaptation:** the VAE's `conv_in` / `conv_out` are widened from 3
  (RGB) to 4 channels in the order ρ, mₓ, m_y, p. Pressure gets the new,
  zero-initialised 4th slot; the three pretrained slots keep their weights.
- **Data:** Euler multi-quadrant (2D compressible Euler, parameterised by γ).
  4000 train / 500 val (seeded split of the train file) / 500 test.
  Per-channel normalisation u' = clip((u − μ) / 5σ, −1, 1).
- **VAE baseline ("Full FT"):** whole encoder + decoder trained with
  1.0·RMSE + 50·H1, AdamW LR 5e-5 (200 warm-up steps, cosine to 1e-6),
  20 epochs, effective batch 32, seed 42. **Every ablation changes exactly one
  setting.**
- **DiT:** full transformer fine-tuned with flow matching on each VAE's
  posterior-mean latents: 8000 steps, LR 1e-5, effective batch 32. γ enters
  through a Fourier-feature MLP as 4 tokens; the first latent frame is kept
  clean as the condition. Inference: 20 steps, no CFG.
- **Metric:** VRMSE = sqrt(mean((û − u)²) / var(u)) per simulation over all
  frames and pixels. The headline number is the **channel-mean VRMSE**
  (mean of the four per-channel VRMSEs, so each field counts equally),
  averaged over the 500 test sims; ± is the SEM over sims.
- **Noise floor:** a 3-seed repeat at 128² varied by ~1.1%, and the same
  config on two clusters by 0.8%. **Differences of ≲1% are ties.**
- **Hardware:** Intel Data Center GPU Max 1550 nodes; one arm on 4× NVIDIA
  GH200 at the same effective batch.

## 1. VAE reconstruction

Δ is relative to the baseline. "Paired" = number of test sims (of 500) on
which the arm beats the baseline.

| Group | Arm | **Ch-mean ± SEM** | Δ | Paired | ρ | mₓ | m_y | p |
|---|---|---|---|---|---|---|---|---|
| Adaptation | Unfinetuned (widened only) | 0.5569 ± 0.0117 | +595.9% | 0/500 | 0.1269 | 0.1116 | 0.1302 | 1.8589 |
| Adaptation | Colour adapter (frozen VAE) | 0.4643 ± 0.0102 | +480.2% | 0/500 | 0.1050 | 0.0863 | 0.0924 | 1.5736 |
| Adaptation | Decoder only | 0.1441 ± 0.0023 | +80.0% | 0/500 | 0.0974 | 0.0870 | 0.0938 | 0.2980 |
| Adaptation | Decoder + conv_in | 0.0929 ± 0.0015 | +16.1% | 0/500 | 0.0843 | 0.0772 | 0.0803 | 0.1299 |
| — | **Full FT (baseline)** | **0.0800 ± 0.0014** | — | — | 0.0727 | 0.0707 | 0.0704 | 0.1064 |
| Loss | RMSE only (no H1) | 0.0814 ± 0.0014 | +1.7% | 9/500 | 0.0767 | 0.0718 | 0.0724 | 0.1048 |
| Loss | RMSE + H1 + SSIM (0.15) | 0.0889 ± 0.0015 | +11.1% | 0/500 | 0.0808 | 0.0787 | 0.0786 | 0.1174 |
| LR | LR 1e-5 | 0.0938 ± 0.0015 | +17.2% | 0/500 | 0.0805 | 0.0773 | 0.0778 | 0.1395 |
| Resolution | Trained at 128² | 0.1200 ± 0.0019 | +49.9% | 0/500 | 0.1202 | 0.1009 | 0.1016 | 0.1572 |
| Latent reg. | KL 1e-7 | 0.0804 ± 0.0014 | +0.5% | 122/500 | 0.0728 | 0.0707 | 0.0705 | 0.1077 |
| Latent reg. | KL 1e-5 | 0.0877 ± 0.0014 | +9.6% | 0/500 | 0.0807 | 0.0792 | 0.0784 | 0.1126 |
| Latent reg. | SDS 1e-4 | 0.0793 ± 0.0014 | −0.9% | 488/500 | 0.0723 | 0.0703 | 0.0700 | 0.1046 |
| Latent reg. | SDS 1e-2 | 0.0806 ± 0.0014 | +0.8% | 59/500 | 0.0734 | 0.0714 | 0.0709 | 0.1069 |
| Latent reg. | SDS 1 | 0.1040 ± 0.0015 | +29.9% | 0/500 | 0.0963 | 0.0941 | 0.0940 | 0.1316 |

The validation split gives the same ranking (each arm's Δ agrees with its
test Δ to within 0.6 points, except the two arms far from the baseline).
The 128² arm is applied directly to the 256² test file, which works because
the VAE is fully convolutional; on 128² validation data it is also ~50% worse.
KL weights refer to the KL summed over each sample's latent elements, with the
decoder seeing the posterior mean during training (`kl: {reduction: sum,
decode: mean}` in the configs; the code's default is the per-element mean with
posterior sampling).

**Findings**

1. **Full fine-tuning is needed, and pressure is the reason.** The pretrained
   VAE cannot see or produce the zero-initialised pressure channel (p 1.86,
   while ρ/m are ~0.11–0.13). A learned colour adapter around a frozen VAE
   fixes ρ/m but not pressure (1.57). Training only the decoder reaches 0.30
   on pressure because the frozen encoder never sees it; unfreezing just
   `conv_in` closes most of the gap (0.130) and the full encoder closes the
   rest (0.106). Every step of this ordering is far above noise.
2. **Loss:** H1 is a small but consistent gain over RMSE alone (baseline
   better on 491/500 sims; size at the noise floor). Adding SSIM costs 11%,
   so it is left out.
3. **Learning rate:** 1e-5 is 17% worse after 20 epochs; 1e-4 diverged.
   5e-5 is close to the largest stable LR.
4. **Resolution:** training at 128² instead of 256² costs 50%.
5. **Latent regularisation:** KL 1e-7 ties the baseline while cutting the KL
   term ~1000× (without KL the posterior variance collapses); KL 1e-5 costs
   9.6%. SDS at 1e-4 and 1e-2 ties the baseline; SDS at 1 costs 30%.
6. Pressure is the hardest channel for every VAE.

## 2. Full pipeline: VAE + DiT

Each VAE gets its own DiT, trained for 8k steps on that VAE's latents with
identical settings. **VAE only** = encode → decode the full ground-truth
sequence (the reconstruction floor). **VAE + DiT** = encode frame 0, roll out
with the DiT, decode. Both are channel-mean VRMSE.

| VAE | VAE only | **VAE + DiT ± SEM** | Δ vs. baseline | Paired | DiT / floor | ρ | mₓ | m_y | p |
|---|---|---|---|---|---|---|---|---|---|
| **Full FT (baseline)** | 0.0802 | **0.4984 ± 0.0060** | — | — | 6.2× | 0.514 | 0.506 | 0.516 | 0.459 |
| Unfinetuned | 0.5579 | 0.9501 ± 0.0128 | +90.6% | 0/500 | 1.7× | 0.637 | 0.646 | 0.657 | 1.859 |
| KL 1e-7 | 0.0807 | 0.4891 ± 0.0059 | −1.9% | 323/500 | 6.1× | 0.507 | 0.499 | 0.506 | 0.445 |
| KL 1e-5 | 0.0879 | 0.4733 ± 0.0053 | −5.0% | 369/500 | 5.4× | 0.493 | 0.480 | 0.488 | 0.433 |
| SDS 1e-4 | 0.0795 | 0.4974 ± 0.0060 | −0.2% | 261/500 (n.s.) | 6.3× | 0.512 | 0.504 | 0.514 | 0.459 |
| **SDS 1e-2** | 0.0808 | **0.4583 ± 0.0054** | **−8.1%** | **443/500** | 5.7× | 0.483 | 0.465 | 0.474 | 0.411 |
| SDS 1 | 0.1044 | 0.5678 ± 0.0051 | +13.9% | 85/500 | 5.4× | 0.572 | 0.578 | 0.585 | 0.536 |

Paired Wilcoxon tests against the baseline give p < 1e-14 for every arm
except SDS 1e-4 (p = 0.38). These cover test-set sampling only; each arm has
one DiT seed.

### Per-γ VAE + DiT channel-mean VRMSE (50 test sims per γ)

| γ | Baseline | Unfinetuned | KL 1e-7 | KL 1e-5 | SDS 1e-4 | SDS 1e-2 | SDS 1 | Baseline VAE-only |
|---|---|---|---|---|---|---|---|---|
| 1.130 | 0.526 | 0.947 | 0.523 | 0.511 | 0.527 | **0.495** | 0.587 | 0.094 |
| 1.220 | 0.495 | 1.009 | 0.494 | 0.472 | 0.499 | **0.463** | 0.570 | 0.085 |
| 1.300 | 0.486 | 0.948 | 0.477 | 0.463 | 0.479 | **0.443** | 0.552 | 0.081 |
| 1.330 | 0.494 | 1.018 | 0.479 | 0.470 | 0.493 | **0.451** | 0.564 | 0.077 |
| 1.365 | 0.478 | 0.924 | 0.476 | 0.469 | 0.479 | **0.450** | 0.578 | 0.076 |
| 1.400 | 0.487 | 0.866 | 0.478 | 0.465 | 0.487 | **0.456** | 0.562 | 0.077 |
| 1.404 | 0.470 | 0.940 | 0.460 | 0.453 | 0.466 | **0.435** | 0.542 | 0.077 |
| 1.453 | 0.489 | 0.956 | 0.478 | 0.471 | 0.488 | **0.452** | 0.570 | 0.077 |
| 1.597 | 0.540 | 0.928 | 0.526 | 0.486 | 0.541 | **0.483** | 0.582 | 0.081 |
| 1.760 | 0.521 | 0.965 | 0.500 | 0.472 | 0.515 | **0.456** | 0.572 | 0.080 |

**Findings**

1. **The DiT is the bottleneck.** For every fine-tuned VAE the rollout error
   (~0.46–0.57) is 5–6× the reconstruction floor (~0.08), so reconstruction
   gains of a few percent are invisible downstream.
2. **Fine-tuning the VAE is still essential.** With the unfinetuned VAE the
   pipeline error nearly doubles (+90.6%), almost entirely from pressure.
3. **Moderate latent regularisation helps the DiT even when it does not help,
   or even hurts, reconstruction.** SDS 1e-2 is the best pipeline (−8.1%,
   better at every γ) while tying the baseline on reconstruction; KL 1e-5 is
   −5.0% downstream despite +9.6% worse reconstruction. The SDS response is
   non-monotonic: 1e-4 does nothing and 1 degrades both stages.
4. **Reconstruction quality does not predict pipeline quality.** The ranking
   of VAEs by reconstruction differs from their ranking end to end, so a VAE
   for latent diffusion has to be judged by the downstream model.
5. **γ dependence:** the extreme γ values are hardest for the DiT, and the
   SDS 1e-2 / KL 1e-5 gains are largest there (at γ = 1.76: −12.5% / −9.4%).

A plausible interpretation (a hypothesis, not proven): both KL and SDS make
the latent distribution better conditioned for the generative model. KL keeps
the posterior from collapsing to near-deterministic, badly scaled latents;
SDS pulls the latents toward what a pretrained video DiT finds easy to
denoise. The latent statistics support the "better conditioned" part.

Latent-space VRMSE is **not** comparable across VAEs (each VAE has its own
latent space and variance); only pixel-space numbers are used to compare them.

## Evaluation protocol note

The DiT is trained on posterior-mean latents, so at inference the
conditioning frame must also be encoded with the **posterior mean**, not a
posterior sample. An earlier evaluation used a sample, which made the
KL-regularised VAEs look catastrophically worse (pipeline VRMSE 1.48 and 2.06
instead of 0.49 and 0.47) and led to the opposite conclusion about KL. VAEs
without KL have a collapsed posterior (σ ≈ 0), so their numbers were unaffected.
All numbers above use the corrected protocol.

## Excluded runs

- **LR 1e-4** and **KL 1e-6** collapsed in epoch 2, right after warm-up reached
  the peak LR, and never recovered. KL 1e-6 is treated as a training
  instability, since 1e-7 and 1e-5 both trained normally.
- Runs with an earlier SSIM implementation that was wrong for signed [−1, 1]
  fields are excluded; all reported runs use the corrected setup.

## Limitations

- One seed per arm. The VAE noise floor (~1%) was measured separately; the
  DiT has no seed repeats, so pipeline differences of a few percent (e.g.
  KL 1e-7's −1.9%) are suggestive only. SDS 1e-2 (−8.1%, consistent across
  all γ) and KL 1e-5 (−5.0%) are well above the VAE-level noise floor.
- The DiT recipe (8k steps) is identical for every VAE, not tuned per VAE, and
  probably not converged.
- One PDE family and one conditioning scalar (γ); one DiT resolution (256²).
- No wall-clock comparison against a numerical solver.

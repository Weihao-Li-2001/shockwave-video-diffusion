# Based on LTX-Video-Trainer by Lightricks (Apache 2.0).
# https://github.com/Lightricks/LTX-Video-Trainer
# Modified: added ScalarConditioningConfig and the VAE fine-tuning configs; removed LoRA and text conditioning.
"""Pydantic configuration models for WinDiNet training and inference."""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

try:
    from windinet.inference.model_loader import LtxvModelVersion
except ImportError:
    LtxvModelVersion = None


class ConfigBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(ConfigBaseModel):
    """Configuration for the base model."""

    model_source: str | Path | LtxvModelVersion = Field(
        default=LtxvModelVersion.latest(),
        description="Model source - can be a HuggingFace repo ID, local path, or LtxvModelVersion",
    )

    load_checkpoint: str | Path | None = Field(
        default=None,
        description="Checkpoint file (or directory: latest checkpoint) to start from. If a "
        "sibling '<stem>.state.pt' exists, optimizer/scheduler/step/RNG are restored too, so "
        "training resumes instead of restarting the LR schedule (see resume_weights_only).",
    )

    # noinspection PyNestedDecorators
    @field_validator("model_source", mode="before")
    @classmethod
    def validate_model_source(cls, v):  # noqa: ANN001, ANN206
        """Try to convert model source to LtxvModelVersion if possible."""
        if isinstance(v, (str, LtxvModelVersion)):
            try:
                return LtxvModelVersion(v)
            except ValueError:
                return v
        return v


class ConditioningConfig(ConfigBaseModel):
    """Configuration for conditioning during training."""

    first_frame_conditioning_p: float = Field(default=0.1, ge=0.0, le=1.0)


class ScalarConditioningConfig(ConfigBaseModel):
    """Scalar (physical-parameter) conditioning that replaces the text prompt, e.g. gamma."""

    enabled: bool = Field(
        default=False,
        description="Whether to enable scalar conditioning",
    )

    scalar_names: list[str] = Field(
        default=["inlet_speed_mps", "field_size_m"],
        description="Names of scalars to embed",
    )

    scalar_ranges: dict[str, tuple[float, float]] = Field(
        default={
            "inlet_speed_mps": (0.1, 20.0),
            "field_size_m": (900.0, 1400.0),
        },
        description="Min/max ranges for each scalar (used for normalization to [0, 1])",
    )

    embedding_dim: int = Field(
        default=4096,
        description="Dimension of scalar embeddings (should match transformer hidden size)",
    )

    num_tokens_per_scalar: int = Field(
        default=4,
        description="Number of embedding tokens to generate per scalar",
        ge=1,
        le=16,
    )

    fourier_features: int = Field(
        default=64,
        description="Number of Fourier features for positional encoding of scalars",
        ge=8,
    )

    mlp_hidden_dim: int = Field(
        default=256,
        description="Hidden dimension of the MLP that processes scalar embeddings",
    )

    dropout: float = Field(
        default=0.0,
        description="Dropout probability for scalar embedding MLP",
        ge=0.0,
        le=1.0,
    )

    @field_validator("scalar_ranges")
    @classmethod
    def validate_scalar_ranges(cls, v: dict, info: ValidationInfo) -> dict:
        """Validate that ranges are provided for all scalar names."""
        scalar_names = info.data.get("scalar_names", [])
        for name in scalar_names:
            if name not in v:
                raise ValueError(f"Range must be provided for scalar '{name}'")
            min_val, max_val = v[name]
            if min_val >= max_val:
                raise ValueError(f"Invalid range for scalar '{name}': min ({min_val}) >= max ({max_val})")
        return v


class OptimizationConfig(ConfigBaseModel):
    """Configuration for optimization parameters."""

    learning_rate: float = Field(default=5e-4)
    steps: int = Field(default=3000)
    batch_size: int = Field(default=2)
    gradient_accumulation_steps: int = Field(default=1)
    max_grad_norm: float = Field(default=1.0)
    scheduler_type: Literal["constant", "linear", "cosine", "cosine_with_restarts", "polynomial"] = Field(
        default="linear"
    )
    scheduler_params: dict = Field(default_factory=dict)
    enable_gradient_checkpointing: bool = Field(default=False)


class AccelerationConfig(ConfigBaseModel):
    """Configuration for hardware acceleration and compute optimization."""

    mixed_precision_mode: Literal["no", "fp16", "bf16"] | None = Field(default="bf16")
    compile_with_inductor: bool = Field(default=True)
    compilation_mode: Literal["default", "reduce-overhead", "max-autotune"] = Field(default="reduce-overhead")


class DataConfig(ConfigBaseModel):
    """Configuration for data loading and processing."""

    preprocessed_data_root: str = Field(description="Path to folder containing preprocessed training data")
    num_dataloader_workers: int = Field(default=2, ge=0)


class ValidationConfig(ConfigBaseModel):
    """Held-out flow-matching loss during DiT training.

    Enabled by pointing ``data_root`` at the val/ split written by
    ``preprocess_dataset.py --eval-sims``.
    """

    data_root: str | None = Field(
        default=None,
        description="Preprocessed held-out split (a dir with latents/ and scalars/). "
        "None disables validation.",
    )
    interval: int | None = Field(
        default=None,
        gt=0,
        description="Optimizer steps between validation passes. Defaults to checkpoints.interval.",
    )
    max_samples: int = Field(
        default=0,
        ge=0,
        description="Cap on validation simulations per pass (0 = the whole split).",
    )


class DitVisualizationConfig(ConfigBaseModel):
    """Periodic GT-vs-rollout panels during DiT training.

    Each pass runs a full rollout and decodes it with the fine-tuned VAE that
    produced the training latents (see windinet/training/dit_visualization.py).
    """

    enabled: bool = Field(default=False)
    interval: int | None = Field(
        default=None,
        gt=0,
        description="Optimizer steps between visualization passes. Defaults to "
        "checkpoints.interval. A final pass always runs at the end of training "
        "regardless of this value. Ignored when `steps` is set.",
    )
    steps: list[int] | None = Field(
        default=None,
        description="Exact optimizer steps to visualize at (e.g. [4000, 8000]). When set, "
        "replaces `interval` and the forced end-of-training pass: only these steps "
        "are rendered.",
    )
    save_video: bool = Field(
        default=False,
        description="Also write a four-channel GT/Prediction/Residual MP4 over the whole "
        "sequence per sample (visualizations/step_<n>/<sample>/video.mp4).",
    )
    video_fps: int = Field(default=8, ge=1)
    num_samples: int = Field(
        default=3,
        ge=1,
        description="Fixed validation samples visualized every pass, picked once "
        "as an even spread over validation.data_root's split_manifest.json val_ids "
        "(not resampled after that).",
    )
    frame_numbers: list[int] = Field(default=[25, 50, 75, 97], min_length=1)
    num_inference_steps: int = Field(
        default=20,
        ge=1,
        description="Denoising steps of the visualization rollout (inference default: 20).",
    )
    dpi: int = Field(default=150, ge=72)

    @field_validator("steps")
    @classmethod
    def validate_steps(cls, values: list[int] | None) -> list[int] | None:
        if values is not None and any(value < 1 for value in values):
            raise ValueError("visualization steps must be >= 1")
        return values


class CheckpointsConfig(ConfigBaseModel):
    """Checkpointing (shared by the VAE and DiT trainers)."""

    interval: int | None = Field(default=None, gt=0)
    keep_last_n: int = Field(default=1, ge=-1)
    save_best_only: bool = Field(
        default=False,
        description=(
            "VAE: keep vae_shockwave_best.safetensors (replaced when best_metric improves) "
            "plus a rolling vae_shockwave_last pair for resuming, instead of one checkpoint "
            "per epoch. 'best' has no optimizer state: best-epoch weights with last-epoch "
            "optimizer moments would make an inconsistent resume."
        ),
    )
    best_metric: Literal["val_total_loss", "val_vrmse"] = Field(
        default="val_total_loss",
        description=(
            "Metric that decides 'best' (lower is better). val_vrmse is comparable across "
            "runs; val_total_loss depends on the loss weights."
        ),
    )
    save_last_state: bool = Field(
        default=True,
        description=(
            "save_best_only mode: also keep the rolling 'last' pair so the run can resume."
        ),
    )


class WandbConfig(ConfigBaseModel):
    """Configuration for Weights & Biases logging."""

    enabled: bool = Field(default=False)
    project: str = Field(default="windinet")
    entity: str | None = Field(default=None)
    tags: list[str] = Field(default_factory=list)


class FlowMatchingConfig(ConfigBaseModel):
    """Flow-matching timestep (noise level) sampling."""

    timestep_sampling_mode: Literal["uniform", "shifted_logit_normal"] = Field(default="shifted_logit_normal")
    timestep_sampling_params: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# VAE fine-tuning configs
# ---------------------------------------------------------------------------


class VaeReconstructionLossConfig(ConfigBaseModel):
    """Parameters of the multi-level wavelet (MLW) loss term."""

    wavelet: str = Field(default="db2")
    spatial_level: int | None = Field(default=2, ge=1)
    temporal_level: int | None = Field(default=2, ge=1)
    mlw_beta: float = Field(default=10.0, ge=0.0)
    mlw_eps: float = Field(default=1e-6, gt=0.0)


class VaeDataConfig(ConfigBaseModel):
    """Dataset and per-channel normalisation for VAE fine-tuning."""

    data_root: str = Field(description="Path to the shockwave HDF5 file")
    eval_sims: int = Field(default=10, description="Simulations held out (seeded random split) for validation")
    num_dataloader_workers: int = Field(default=4, ge=0)
    num_sim_frames: int | None = Field(default=None, ge=1, description="Use only the first N frames (None = all)")
    channel_mean: list[float] | None = Field(
        default=None,
        description="Per-channel training-set means, positionally aligned with channel_order. "
        "Mutually exclusive with normalization_stats_file -- set exactly one of the two.",
    )
    channel_std: list[float] | None = Field(
        default=None,
        description="Per-channel training-set standard deviations, positionally aligned with "
        "channel_order. Mutually exclusive with normalization_stats_file -- set exactly one of the two.",
    )
    normalization_stats_file: str | Path | None = Field(
        default=None,
        description=(
            "YAML/JSON file with a data_normalization_stats block "
            "({channel: {mean, std, ...}}, as in configs/data/ or written by "
            "scripts/compute_channel_stats.py). Fills channel_mean/channel_std by name, "
            "in channel_order, so all configs of one resolution share one stats file."
        ),
    )
    normalization_clip: float = Field(
        default=5.0,
        gt=0.0,
        description="Map mean +/- this many standard deviations to [-1, 1]",
    )
    channel_order: list[str] = Field(
        default=["density", "momentum_x", "momentum_y", "pressure"],
        description=(
            "Physical fields stacked as tensor channels, in order (all four, or a subset "
            "such as [density, momentum_x, momentum_y] to keep the VAE's native 3 channels). "
            "Must equal adapter.channels. Inflating grows the VAE at index 3, so the last "
            "field gets the new zero-initialised slot; pressure, which has no RGB "
            "analogue, goes last."
        ),
    )

    @field_validator("channel_order")
    @classmethod
    def validate_channel_order(cls, values: list[str]) -> list[str]:
        expected = {"density", "momentum_x", "momentum_y", "pressure"}
        if not values or len(set(values)) != len(values) or not set(values) <= expected:
            raise ValueError(
                f"channel_order must be a non-empty, duplicate-free subset (in any order) of "
                f"{sorted(expected)}, got {values}"
            )
        return values

    @model_validator(mode="after")
    def validate_normalization_stats(self):
        has_inline = self.channel_mean is not None or self.channel_std is not None
        has_file = self.normalization_stats_file is not None
        if has_inline and has_file:
            raise ValueError(
                "set either channel_mean/channel_std or normalization_stats_file, not both"
            )
        if has_file:
            if self.channel_mean is not None or self.channel_std is not None:
                raise ValueError("normalization_stats_file set: channel_mean/channel_std must be unset")
            import yaml

            with open(self.normalization_stats_file) as fh:
                # yaml.safe_load also reads compute_channel_stats.py's JSON output.
                stats = yaml.safe_load(fh).get("data_normalization_stats", {})
            missing = [name for name in self.channel_order if name not in stats]
            if missing:
                raise ValueError(
                    f"normalization_stats_file={self.normalization_stats_file!r} is missing "
                    f"stats for {missing} (needed for channel_order={self.channel_order})"
                )
            self.channel_mean = [stats[name]["mean"] for name in self.channel_order]
            self.channel_std = [stats[name]["std"] for name in self.channel_order]
            # Values are now inline; drop the file reference so the dumped
            # training_config.yaml (which has both) can be loaded again.
            self.normalization_stats_file = None
        elif self.channel_mean is None or self.channel_std is None:
            raise ValueError("set either channel_mean and channel_std, or normalization_stats_file")

        n = len(self.channel_order)
        if len(self.channel_mean) != n or len(self.channel_std) != n:
            raise ValueError(f"channel_mean and channel_std must each contain {n} values (one per channel_order entry)")
        if any(value <= 0 for value in self.channel_std):
            raise ValueError("all channel_std values must be positive")
        return self


class VaeOptimizationConfig(ConfigBaseModel):
    """Optimisation for VAE fine-tuning: AdamW, linear warmup, then cosine decay."""

    learning_rate: float = Field(default=5e-5)
    min_learning_rate: float = Field(
        default=1e-6,
        description=(
            "LR floor reached at the end of the cosine decay (linear warmup, then cosine "
            "from learning_rate down to this value)."
        ),
    )
    epochs: int = Field(default=10)
    batch_size: int = Field(default=1)
    gradient_accumulation_steps: int = Field(default=32)
    max_grad_norm: float = Field(default=5.0)
    weight_decay: float = Field(default=0.0)
    warmup_steps: int = Field(default=50, description="Linear warmup optimizer steps")
    warmup_start_factor: float = Field(default=0.01)
    enable_gradient_checkpointing: bool = Field(default=True)



class VaeAdapterConfig(ConfigBaseModel):
    """How the 4 CFD channels are fed through the 3-channel VAE, and which parts train."""

    enabled: bool = Field(default=False, description="Adapt the VAE to the CFD channels (inflate or adapter mode)")
    checkpoint: str | Path | None = Field(default=None, description="adapter mode: checkpoint to initialise the adapters + decoder from")
    channels: list[str] = Field(
        default=["density", "momentum_x", "momentum_y", "pressure"],
        min_length=1,
    )
    mode: Literal["adapter", "inflate"] = Field(
        default="adapter",
        description="'adapter': 1x1 in/out adapters around a frozen 3-ch VAE. 'inflate': grow the "
        "VAE's conv_in/conv_out to read/write all channels natively; the pretrained RGB slots keep "
        "their weights and the new channel starts at zero.",
    )
    freeze_conv_in: bool = Field(
        default=False,
        description=(
            "mode='inflate' only: keep encoder.conv_in frozen (decoder-only training). The new "
            "channel then never reaches the encoder; the latent space stays fixed, so existing "
            "latents and DiTs stay valid."
        ),
    )
    unfreeze_encoder_trunk: bool = Field(
        default=True,
        description=(
            "mode='inflate' only: also train the rest of the encoder (full fine-tune, the "
            "default). False trains only conv_in on the encoder side (decoder + conv_in). "
            "The encoder trunk is always frozen in adapter mode."
        ),
    )
    hidden_channels: int = Field(default=32, ge=1, description="adapter mode: hidden width of the channel adapters")
    activation: Literal["relu", "silu", "swish", "gelu", "tanh"] = "gelu"
    identity_init: bool = Field(
        default=False,
        description="adapter mode: initialise the adapters as identity + zero residual.",
    )
    default_temb: float = Field(default=0.0)

    @model_validator(mode="after")
    def validate_encoder_scope(self) -> "VaeAdapterConfig":
        # The latent-space fingerprint hashes conv_in, not the encoder trunk, so
        # the trunk must never train while conv_in stays frozen.
        if self.mode == "inflate" and self.freeze_conv_in and self.unfreeze_encoder_trunk:
            raise ValueError(
                "adapter.freeze_conv_in=true requires adapter.unfreeze_encoder_trunk=false "
                "(training the encoder trunk changes the latent space)"
            )
        return self


# Loss terms a config may weight. rmse/h1/ssim/mlw/vrms/kl/swd are cheap and
# always computed (and logged), so a zero weight only removes them from the
# objective. sds runs a frozen 2B-parameter transformer, so it is computed only
# when sds.enabled is set as well (see SdsLossConfig).
LOSS_NAMES = {"rmse", "h1", "ssim", "mlw", "vrms", "kl", "sds", "swd"}


class LossWeightingConfig(ConfigBaseModel):
    """Fixed weights of the VAE objective: total = sum(weights[name] * loss[name])."""

    weights: dict[str, float] = Field(default={"rmse": 1.0, "h1": 50.0})

    @field_validator("weights")
    @classmethod
    def validate_loss_names(cls, weights: dict[str, float]) -> dict[str, float]:
        unknown = set(weights) - LOSS_NAMES
        if unknown:
            raise ValueError(f"unknown loss names {sorted(unknown)}; known: {sorted(LOSS_NAMES)}")
        return weights


class VaeVisualizationConfig(ConfigBaseModel):
    """Periodic reconstruction plots for a fixed validation subset."""

    enabled: bool = Field(default=True)
    interval_epochs: int = Field(default=1, ge=1)
    epochs: list[int] | None = Field(
        default=None,
        description=(
            "Exact epochs to visualize at (e.g. [10, 20]). When set, replaces "
            "interval_epochs and the forced last-epoch pass: only these epochs are rendered."
        ),
    )
    save_video: bool = Field(
        default=False,
        description=(
            "Also write a four-channel GT/Prediction/Residual MP4 over the whole sequence "
            "per sample (visualizations/epoch_<n>/<sample>/video.mp4)."
        ),
    )
    video_fps: int = Field(default=8, ge=1)
    num_samples: int = Field(
        default=3,
        ge=1,
        description=(
            "How many eval sims to render, chosen as an evenly-spaced spread over the eval "
            "set's gamma range rather than eval order -- 1 is the median gamma, 2 is "
            "min/max, 3 (default) is min/median/max, N>3 adds evenly-spaced points between "
            "the extremes. Selection is fixed per run (recomputed from eval_set, not "
            "random), so the same sims are plotted every epoch."
        ),
    )
    frame_numbers: list[int] = Field(default=[25, 50, 75, 97], min_length=1)
    dpi: int = Field(default=150, ge=72)

    @field_validator("frame_numbers")
    @classmethod
    def validate_frame_numbers(cls, values: list[int]) -> list[int]:
        if any(value < 1 for value in values):
            raise ValueError("visualization frame_numbers are one-based and must be >= 1")
        return values

    @field_validator("epochs")
    @classmethod
    def validate_epochs(cls, values: list[int] | None) -> list[int] | None:
        if values is not None and any(value < 1 for value in values):
            raise ValueError("visualization epochs must be >= 1")
        return values


class SdsLossConfig(ConfigBaseModel):
    """Score distillation (SDS) from a frozen video DiT, as a VAE latent regulariser (loss "sds").

    On the encoder's own latents z:

        z_sigma = (1 - sigma) * z + sigma * eps     sigma ~ timestep_sampling_mode
        v_hat   = v_phi(z_sigma; sigma, gamma)       (frozen DiT)
        loss    = || v_hat - (eps - z) ||^2

    This pulls the latent space toward what a pretrained video DiT finds easy
    to denoise, which pixel-space reconstruction losses know nothing about.

    The critic v_phi is either the stock LTX-Video transformer
    (``untrained_dit=true``, used in all reported runs) or a DiT already
    fine-tuned on some VAE's latents (``dit_checkpoint``). The stock model
    avoids a circularity: a DiT trained on one VAE's latents would mostly
    pull a new VAE toward that VAE's quirks.

    The DiT forward runs under no_grad, so the gradient reaches the encoder
    only through the ``-z`` in the target, i.e. the standard SDS gradient
    without the score network's Jacobian (Poole et al., DreamFusion). As in
    DiT training, frame 0 stays clean (first-frame conditioning).
    """

    enabled: bool = Field(
        default=False,
        description="Load the frozen DiT. Needed in addition to a non-zero "
        "loss_weighting.weights.sds, since every step then runs a 2B-parameter forward pass.",
    )
    model_source: str | Path | LtxvModelVersion = Field(
        default=LtxvModelVersion.latest(),
        description="Base transformer (same meaning as ModelConfig.model_source).",
    )
    dit_checkpoint: str | Path | None = Field(
        default=None,
        description="A trained DiT's model_weights_*.safetensors to distill from (when "
        "untrained_dit=false). Its sibling scalar_embedding_*.safetensors is loaded too.",
    )
    untrained_dit: bool = Field(
        default=False,
        description="Distill from the stock pretrained transformer (a task-independent "
        "critic). It has no notion of gamma, so the scalar embedding stays at its random "
        "initialisation. Mutually exclusive with dit_checkpoint.",
    )
    scalar_conditioning: ScalarConditioningConfig = Field(
        default_factory=lambda: ScalarConditioningConfig(
            enabled=True,
            scalar_names=["gamma"],
            scalar_ranges={"gamma": (1.0, 2.0)},
            dropout=0.1,
        ),
        description="Must match the scalar_conditioning of the DiT that dit_checkpoint comes from.",
    )
    timestep_sampling_mode: Literal["uniform", "shifted_logit_normal"] = Field(default="shifted_logit_normal")
    timestep_sampling_params: dict = Field(default_factory=dict)

    # noinspection PyNestedDecorators
    @field_validator("model_source", mode="before")
    @classmethod
    def validate_model_source(cls, v):  # noqa: ANN001, ANN206
        if isinstance(v, (str, LtxvModelVersion)):
            try:
                return LtxvModelVersion(v)
            except ValueError:
                return v
        return v

    @model_validator(mode="after")
    def validate_dit_checkpoint_when_enabled(self):
        if self.enabled and self.untrained_dit and self.dit_checkpoint:
            raise ValueError("sds.untrained_dit=true and sds.dit_checkpoint are mutually exclusive")
        if self.enabled and not self.untrained_dit and not self.dit_checkpoint:
            raise ValueError("sds.enabled=true requires sds.dit_checkpoint (or sds.untrained_dit=true)")
        return self


class KlLossConfig(ConfigBaseModel):
    """How the KL term (loss "kl") is computed; its weight is loss_weighting.weights.kl."""

    reduction: Literal["mean", "sum"] = Field(
        default="mean",
        description="'mean': averaged over latent elements (weight independent of resolution). "
        "'sum': summed per sample, as in the KL runs reported in RESULTS.md.",
    )
    decode: Literal["sample", "mean"] = Field(
        default="sample",
        description="What the decoder sees during training when KL is weighted: a "
        "reparameterized posterior sample (so KL actually regularises the variance), or the "
        "posterior mean (as in the KL runs reported in RESULTS.md).",
    )


class SwdLossConfig(ConfigBaseModel):
    """Sliced-Wasserstein prior on the aggregate latent distribution (loss "swd").

    WAE-style alternative to KL: matches the distribution of all latent tokens
    (normalised posterior means, as the DiT sees them) to N(0, I), rather than
    each sample's posterior. See windinet.losses.sliced_wasserstein.
    """

    num_projections: int = Field(
        default=256,
        gt=0,
        description="Random projection directions per call (estimator noise ~ 1/sqrt(n)).",
    )
    bank_size: int = Field(
        default=32,
        ge=0,
        description="Detached latents of previous sims (per rank) pooled with the current "
        "one, so the match is to the aggregate distribution. 0 matches each sim on its own.",
    )


class VaeTrainerConfig(ConfigBaseModel):
    """Top-level config for scripts/finetune_vae.py."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    adapter: VaeAdapterConfig = Field(default_factory=VaeAdapterConfig)
    data: VaeDataConfig = Field(default_factory=VaeDataConfig)
    optimization: VaeOptimizationConfig = Field(default_factory=VaeOptimizationConfig)
    loss: VaeReconstructionLossConfig = Field(default_factory=VaeReconstructionLossConfig)
    loss_weighting: LossWeightingConfig = Field(default_factory=LossWeightingConfig)
    sds: SdsLossConfig = Field(default_factory=SdsLossConfig)
    kl: KlLossConfig = Field(default_factory=KlLossConfig)
    swd: SwdLossConfig = Field(default_factory=SwdLossConfig)
    visualization: VaeVisualizationConfig = Field(default_factory=VaeVisualizationConfig)
    acceleration: AccelerationConfig = Field(default_factory=AccelerationConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    seed: int = Field(default=42)
    output_dir: str = Field(default="outputs/vae/vae_finetune")
    clean_output_dir: bool = Field(
        default=False,
        description="Delete output_dir before training.",
    )
    resume_from: str | Path | None = Field(
        default=None,
        description="Checkpoint .safetensors to resume from. Its sibling '<stem>.state.pt' "
        "(optimizer, scheduler, RNG, epoch) is restored too, so training continues where it "
        "stopped. Safe to resume into the same output_dir.",
    )
    resume_weights_only: bool = Field(
        default=False,
        description=(
            "Load only the weights from resume_from (warm restart with a fresh optimizer and "
            "LR schedule), ignoring its .state.pt."
        ),
    )

    @field_validator("output_dir")
    @classmethod
    def expand_vae_output_path(cls, v: str) -> str:
        return str(Path(os.path.expandvars(v)).expanduser().resolve())

    @model_validator(mode="after")
    def validate_channel_order_consistency(self):
        if self.data.channel_order != self.adapter.channels:
            raise ValueError(
                "data.channel_order and adapter.channels must be identical lists -- "
                f"got data.channel_order={self.data.channel_order} vs "
                f"adapter.channels={self.adapter.channels}. data.channel_order controls "
                "the actual tensor stack order; adapter.channels is metadata (labels, "
                "checkpoint info, visualization titles) that must describe it correctly."
            )
        return self


# ---------------------------------------------------------------------------
# DiT training config
# ---------------------------------------------------------------------------


class LtxvTrainerConfig(ConfigBaseModel):
    """Top-level config for scripts/train.py (DiT training)."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    conditioning: ConditioningConfig = Field(default_factory=ConditioningConfig)
    scalar_conditioning: ScalarConditioningConfig = Field(default_factory=ScalarConditioningConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    acceleration: AccelerationConfig = Field(default_factory=AccelerationConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    flow_matching: FlowMatchingConfig = Field(default_factory=FlowMatchingConfig)
    visualization: DitVisualizationConfig = Field(default_factory=DitVisualizationConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    seed: int = Field(default=42)
    output_dir: str = Field(default="outputs")
    resume_weights_only: bool = Field(
        default=False,
        description=(
            "Load only the weights from model.load_checkpoint (warm restart with a fresh "
            "optimizer and LR schedule), ignoring its .state.pt."
        ),
    )

    # noinspection PyNestedDecorators
    @field_validator("output_dir")
    @classmethod
    def expand_output_path(cls, v: str) -> str:
        return str(Path(os.path.expandvars(v)).expanduser().resolve())

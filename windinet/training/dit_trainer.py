# Based on LTX-Video-Trainer by Lightricks (Apache 2.0).
# https://github.com/Lightricks/LTX-Video-Trainer
# Modified: scalar (gamma) conditioning instead of text, full fine-tuning only, validation,
# resumable checkpoints and latent-provenance tracking.
"""Flow-matching trainer for the WinDiNet diffusion transformer."""

import json
import math
import os
import random
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import MagicMock

import rich
import torch
import wandb
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from pydantic import BaseModel, ConfigDict, computed_field
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    LRScheduler,
    PolynomialLR,
    StepLR,
)
from torch.utils.data import DataLoader

from windinet.utils import logger, convert_checkpoint, get_gpu_memory_gb
from windinet.config import LtxvTrainerConfig
from windinet.training.precomputed_dataset import PrecomputedDataset
from windinet.inference.model_loader import load_ltxv_components
from windinet.scalar_embeddings import ScalarEmbedding
from windinet.training.timestep_samplers import SAMPLERS
from windinet.training.dit_visualization import DitVisualizer
from windinet.latent_utils import get_rope_scale_factors

os.environ["TOKENIZERS_PARALLELISM"] = "true"

IS_MAIN_PROCESS = os.environ.get("LOCAL_RANK", "0") == "0"
if not IS_MAIN_PROCESS:
    from transformers.utils.logging import disable_progress_bar
    disable_progress_bar()

StepCallback = Callable[[int, int, list[Path]], None]

COMPILE_WARMUP_STEPS = 5
MEMORY_CHECK_INTERVAL = 200
DEFAULT_FPS = 24


# ---------------------------------------------------------------------------
# Training batch container
# ---------------------------------------------------------------------------

class TrainingBatch(BaseModel):
    """Container for prepared training data."""

    latents: Tensor
    targets: Tensor
    prompt_embeds: Tensor
    prompt_attention_mask: Tensor
    scalars: Tensor | None = None
    timesteps: Tensor
    sigmas: Tensor
    conditioning_mask: Tensor
    num_frames: int
    height: int
    width: int
    fps: float
    rope_interpolation_scale: list[float]

    @computed_field
    @property
    def batch_size(self) -> int:
        return self.latents.shape[0]

    @computed_field
    @property
    def sequence_length(self) -> int:
        return self.latents.shape[1]

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# Training statistics
# ---------------------------------------------------------------------------

class TrainingStats(BaseModel):
    total_time_seconds: float
    compilation_time_seconds: Optional[float]
    training_time: float
    steps_per_second: float
    samples_per_second: float
    peak_gpu_memory_gb: float
    global_batch_size: int
    num_processes: int


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class LtxvTrainer:
    def __init__(self, trainer_config: LtxvTrainerConfig) -> None:
        self._config = trainer_config
        # Resume bookkeeping, read into memory in __init__ (before train() may
        # touch output_dir) so resuming into the same output_dir is safe.
        self._resume_global_step = 0
        self._resume_state: dict | None = None
        self._print_config(trainer_config)
        self._setup_accelerator()
        self._load_models()
        self._init_scalar_embedding()
        self._compile_transformer()
        self._collect_trainable_params()
        self._load_checkpoint()
        self._prepare_models_for_training()
        self._dataset = None
        self._val_dataset = None
        self._val_dataloader = None
        self._visualizer: DitVisualizer | None = None
        self._global_step = -1
        self._checkpoint_paths = []
        self._epoch_losses = []
        self._epoch_train_losses = []
        self._last_val_loss: float | None = None
        self._last_val_step: int | None = None
        self._best_val_loss: float = math.inf
        self._best_step: int | None = None
        self._init_wandb()

    @property
    def _validation_interval(self) -> int | None:
        """Optimizer steps between validation passes, or None if disabled."""
        if self._config.validation.data_root is None:
            return None
        return self._config.validation.interval or self._config.checkpoints.interval

    @property
    def _visualization_interval(self) -> int | None:
        """Optimizer steps between visualization passes, or None if disabled."""
        if not self._config.visualization.enabled or self._config.visualization.steps is not None:
            return None
        return self._config.visualization.interval or self._config.checkpoints.interval

    def _visualization_due(self, step: int) -> bool:
        """Whether a periodic visualization pass is due at optimizer step `step`."""
        vis_cfg = self._config.visualization
        if not vis_cfg.enabled or step <= 0:
            return False
        if vis_cfg.steps is not None:
            return step in vis_cfg.steps
        return step % self._visualization_interval == 0

    def _init_visualizer(self) -> None:
        """Build the DitVisualizer (rank 0 only) if visualization is enabled."""
        if not self._config.visualization.enabled or not IS_MAIN_PROCESS:
            return
        vis_cfg = self._config.visualization
        self._visualizer = DitVisualizer(
            preprocessed_data_root=self._config.data.preprocessed_data_root,
            model_source=self._config.model.model_source,
            scalar_names=self._config.scalar_conditioning.scalar_names,
            num_samples=vis_cfg.num_samples,
            frame_numbers=vis_cfg.frame_numbers,
            num_inference_steps=vis_cfg.num_inference_steps,
            dpi=vis_cfg.dpi,
            save_video=vis_cfg.save_video,
            video_fps=vis_cfg.video_fps,
            output_dir=self._config.output_dir,
            seed=self._config.seed,
            device=self._accelerator.device,
        )

    def _run_visualization(self, step: int) -> None:
        """Render GT-vs-rollout panels for the fixed visualization samples.

        Like _validate(), it restores the training RNG afterwards and uses the
        unwrapped (non-DDP) modules, since only rank 0 runs it.
        """
        if self._visualizer is None:
            return

        device = self._accelerator.device
        py_rng_state = random.getstate()
        cpu_rng_state = torch.get_rng_state()
        if device.type == "cuda":
            accel_rng_state = torch.cuda.get_rng_state_all()
        elif device.type == "xpu":
            accel_rng_state = torch.xpu.get_rng_state_all()
        else:
            accel_rng_state = None

        transformer = self._accelerator.unwrap_model(self._transformer)
        scalar_embedding = (
            self._accelerator.unwrap_model(self._scalar_embedding)
            if self._scalar_embedding is not None
            else None
        )
        transformer.eval()
        try:
            self._visualizer.run(
                transformer=transformer,
                scalar_embedding=scalar_embedding,
                scheduler=self._scheduler,
                step=step,
            )
        finally:
            transformer.train()
            random.setstate(py_rng_state)
            torch.set_rng_state(cpu_rng_state)
            if accel_rng_state is not None:
                if device.type == "cuda":
                    torch.cuda.set_rng_state_all(accel_rng_state)
                elif device.type == "xpu":
                    torch.xpu.set_rng_state_all(accel_rng_state)

    # ------------------------------------------------------------------
    # Batch preparation
    # ------------------------------------------------------------------

    def _get_data_sources(self) -> list[str]:
        """Data sources required for training."""
        sources = ["latents"]
        if self._config.scalar_conditioning.enabled:
            sources.append("scalars")
        return sources

    def _prepare_batch(
        self, batch: dict[str, Any], timestep_sampler, force_conditioning: bool = False
    ) -> TrainingBatch:
        """Prepare a training batch with noise and conditioning.

        ``force_conditioning`` always applies the first-frame conditioning mask
        instead of sampling it. Validation uses it so the measured loss matches
        what inference actually does (the initial condition is always given) and
        does not fluctuate with the Bernoulli draw.
        """
        latents = batch["latents"]
        target_latents = latents["latents"]

        latent_frames = latents["num_frames"][0].item()
        latent_height = latents["height"][0].item()
        latent_width = latents["width"][0].item()

        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values in batch. Found: {fps.tolist()}, using first: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Scalar-only mode: always use zero placeholder text embeddings
        batch_size = target_latents.shape[0]
        prompt_embeds = torch.zeros(batch_size, 1, 4096, device=target_latents.device, dtype=target_latents.dtype)
        prompt_attention_mask = torch.ones(batch_size, 1, device=target_latents.device, dtype=torch.bool)

        # Get scalar conditioning if enabled
        scalars = None
        if self._config.scalar_conditioning.enabled and "scalars" in batch:
            scalars = batch["scalars"]["scalars"]

        # Create conditioning mask (first frame conditioning)
        conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=target_latents.shape[1],
            height=latent_height,
            width=latent_width,
            device=target_latents.device,
            force=force_conditioning,
        )

        # Create noise for the target latents
        sigmas = timestep_sampler.sample_for(target_latents)
        noise = torch.randn_like(target_latents, device=target_latents.device)

        sigmas = sigmas.view(-1, 1, 1)
        noisy_latents = (1 - sigmas) * target_latents + sigmas * noise

        # For conditioning tokens, use clean latents
        conditioning_mask_expanded = conditioning_mask.unsqueeze(-1)
        noisy_latents = torch.where(conditioning_mask_expanded, target_latents, noisy_latents)

        targets = noise - target_latents

        # Create timesteps based on conditioning mask
        sampled_timestep_values = torch.round(sigmas.squeeze(-1).squeeze(-1) * 1000.0).long()
        timesteps = self._create_timesteps_from_conditioning_mask(conditioning_mask, sampled_timestep_values)

        rope_interpolation_scale_factors = get_rope_scale_factors(fps)

        return TrainingBatch(
            latents=noisy_latents,
            targets=targets,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            scalars=scalars,
            timesteps=timesteps,
            sigmas=sigmas,
            conditioning_mask=conditioning_mask,
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            fps=fps,
            rope_interpolation_scale=rope_interpolation_scale_factors,
        )

    @staticmethod
    def _prepare_model_inputs(batch: TrainingBatch) -> dict[str, Any]:
        """Prepare inputs for the transformer model."""
        return {
            "hidden_states": batch.latents,
            "encoder_hidden_states": batch.prompt_embeds,
            "timestep": batch.timesteps,
            "encoder_attention_mask": batch.prompt_attention_mask,
            "num_frames": batch.num_frames,
            "height": batch.height,
            "width": batch.width,
            "rope_interpolation_scale": batch.rope_interpolation_scale,
            "return_dict": False,
        }

    @staticmethod
    def _compute_loss(model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        """Compute masked MSE loss using conditioning mask."""
        loss = (model_pred - batch.targets).pow(2)
        loss_mask = (~batch.conditioning_mask.unsqueeze(-1)).float()
        loss = loss.mul(loss_mask).div(loss_mask.mean())
        return loss.mean()

    @staticmethod
    def _create_timesteps_from_conditioning_mask(
        conditioning_mask: Tensor, sampled_timestep_values: Tensor
    ) -> Tensor:
        expanded_timesteps = sampled_timestep_values.unsqueeze(1).expand_as(conditioning_mask)
        return torch.where(conditioning_mask, 0, expanded_timesteps)

    def _create_first_frame_conditioning_mask(
        self,
        batch_size: int,
        sequence_length: int,
        height: int,
        width: int,
        device: torch.device,
        force: bool = False,
    ) -> Tensor:
        conditioning_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=device)
        if force or (
            self._config.conditioning.first_frame_conditioning_p > 0
            and random.random() < self._config.conditioning.first_frame_conditioning_p
        ):
            first_frame_end_idx = height * width
            if first_frame_end_idx < sequence_length:
                conditioning_mask[:, :first_frame_end_idx] = True
        return conditioning_mask

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        disable_progress_bars: bool = False,
        step_callback: StepCallback | None = None,
    ) -> tuple[Path, TrainingStats]:
        """Start the training process."""
        device = self._accelerator.device
        cfg = self._config
        start_mem = get_gpu_memory_gb(device)

        train_start_time = time.time()
        set_seed(cfg.seed)
        logger.debug(f"Process {self._accelerator.process_index} using seed: {cfg.seed}")

        self._init_optimizer()
        self._init_dataloader()
        self._init_val_dataloader()
        self._init_visualizer()
        data_iter = iter(self._dataloader)
        self._init_timestep_sampler()
        val_interval = self._validation_interval
        if val_interval is None and IS_MAIN_PROCESS:
            logger.warning(
                "No validation.data_root configured: this run will only report training loss "
                "and cannot tell you whether the model generalizes."
            )

        self._accelerator.wait_for_everyone()
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self._save_config()
        self._record_latent_provenance()

        logger.info("Starting training...")

        if disable_progress_bars or not IS_MAIN_PROCESS:
            train_progress = MagicMock()
            live = nullcontext()
            if IS_MAIN_PROCESS:
                logger.warning("Progress bars disabled.")
        else:
            train_progress = Progress(
                TextColumn("Training Step"),
                MofNCompleteColumn(),
                BarColumn(bar_width=40, style="blue"),
                TextColumn("Loss: {task.fields[loss]:.4f}"),
                TextColumn("LR: {task.fields[lr]:.2e}"),
                TextColumn("Time/Step: {task.fields[step_time]:.2f}s"),
                TimeElapsedColumn(),
                TextColumn("ETA:"),
                TimeRemainingColumn(compact=True),
            )
            live = Live(Panel(train_progress), refresh_per_second=2)

        self._transformer.train()
        self._global_step = self._resume_global_step
        if self._global_step >= cfg.optimization.steps:
            raise ValueError(
                f"resume global_step {self._global_step} >= configured optimization.steps "
                f"{cfg.optimization.steps}; increase optimization.steps to train further."
            )
        if self._global_step > 0 and IS_MAIN_PROCESS:
            logger.info(f"Resuming training at global_step {self._global_step}")
        remaining_opt_steps = cfg.optimization.steps - self._global_step

        compilation_time = None
        peak_mem_during_training = start_mem
        actual_training_start = None

        with live:
            task = train_progress.add_task(
                "Training",
                total=cfg.optimization.steps,
                completed=self._global_step,
                loss=0.0,
                lr=cfg.optimization.learning_rate,
                step_time=0.0,
            )

            for step in range(remaining_opt_steps * cfg.optimization.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self._dataloader)
                    batch = next(data_iter)

                if step == COMPILE_WARMUP_STEPS and cfg.acceleration.compile_with_inductor:
                    compilation_time = time.time() - train_start_time
                    actual_training_start = time.time()
                elif step == COMPILE_WARMUP_STEPS and not cfg.acceleration.compile_with_inductor:
                    actual_training_start = train_start_time

                step_start_time = time.time()
                with self._accelerator.accumulate(self._transformer):
                    loss = self._training_step(batch)
                    self._accelerator.backward(loss)

                    is_optimization_step = self._accelerator.sync_gradients

                    if is_optimization_step:
                        self._global_step += 1
                        avg_loss = self._accelerator.gather(loss.detach()).mean()

                        if IS_MAIN_PROCESS and (self._global_step % int(os.getenv("LTXV_TRAIN_LOG_EVERY", "1")) == 0):
                            current_lr = self._optimizer.param_groups[0]["lr"]
                            self._accelerator.print(
                                f"[train] step={self._global_step} loss={avg_loss.item():.6f} lr={current_lr:.3e}"
                            )

                        if cfg.optimization.max_grad_norm > 0:
                            self._accelerator.clip_grad_norm_(
                                self._trainable_params,
                                cfg.optimization.max_grad_norm,
                            )

                        self._optimizer.step()
                        self._optimizer.zero_grad()

                        if self._lr_scheduler is not None:
                            self._lr_scheduler.step()

                    # Validation before checkpointing, so the epoch record below
                    # can carry the val loss belonging to the weights just saved.
                    if (
                        val_interval
                        and is_optimization_step
                        and self._global_step > 0
                        and self._global_step % val_interval == 0
                    ):
                        if IS_MAIN_PROCESS:
                            val_loss = self._validate()
                            if val_loss is not None:
                                self._last_val_loss = val_loss
                                self._last_val_step = self._global_step
                                self._accelerator.print(
                                    f"[val] step={self._global_step} val_loss={val_loss:.6f}"
                                )
                                self._log_metrics({
                                    "val/loss": val_loss,
                                    "val/global_step": self._global_step,
                                })
                        self._accelerator.wait_for_everyone()

                    if (
                        cfg.checkpoints.interval
                        and is_optimization_step
                        and self._global_step > 0
                        and self._global_step % cfg.checkpoints.interval == 0
                        and IS_MAIN_PROCESS
                    ):
                        self._save_checkpoint()

                    if (
                        is_optimization_step
                        and self._visualization_due(self._global_step)
                        and IS_MAIN_PROCESS
                    ):
                        self._run_visualization(self._global_step)
                    self._accelerator.wait_for_everyone()

                    if (
                        cfg.checkpoints.interval
                        and is_optimization_step
                        and self._global_step > 0
                        and self._global_step % cfg.checkpoints.interval == 0
                    ):
                        epoch = self._global_step // cfg.checkpoints.interval
                        if self._epoch_train_losses:
                            epoch_train_loss = sum(self._epoch_train_losses) / len(self._epoch_train_losses)
                            self._epoch_train_losses.clear()
                        else:
                            epoch_train_loss = None

                        # Only attach the val loss if it was measured on these
                        # exact weights; a stale one would misrepresent the gap.
                        epoch_val_loss = (
                            self._last_val_loss if self._last_val_step == self._global_step else None
                        )

                        if IS_MAIN_PROCESS:
                            train_str = f"{epoch_train_loss:.6f}" if epoch_train_loss else "N/A"
                            val_str = f" val_loss={epoch_val_loss:.6f}" if epoch_val_loss is not None else ""
                            self._accelerator.print(
                                f"[epoch {epoch}] step={self._global_step} train_loss={train_str}{val_str}"
                            )
                            epoch_metrics = {"epoch": epoch}
                            if epoch_train_loss is not None:
                                epoch_metrics["epoch/train_loss"] = epoch_train_loss
                            if epoch_val_loss is not None:
                                epoch_metrics["epoch/val_loss"] = epoch_val_loss
                            self._log_metrics(epoch_metrics)

                            self._epoch_losses.append({
                                "epoch": epoch,
                                "step": self._global_step,
                                "train_loss": epoch_train_loss,
                                "val_loss": epoch_val_loss,
                            })
                            self._save_epoch_log()

                    self._accelerator.wait_for_everyone()

                    if step_callback and is_optimization_step:
                        step_callback(self._global_step, cfg.optimization.steps, [])

                    self._accelerator.wait_for_everyone()

                    if IS_MAIN_PROCESS:
                        current_lr = self._optimizer.param_groups[0]["lr"]
                        elapsed = time.time() - train_start_time
                        progress_percentage = self._global_step / cfg.optimization.steps
                        if progress_percentage > 0:
                            total_estimated = elapsed / progress_percentage
                            total_time = f"{total_estimated // 3600:.0f}h {(total_estimated % 3600) // 60:.0f}m"
                        else:
                            total_time = "calculating..."

                        step_time = (time.time() - step_start_time) * cfg.optimization.gradient_accumulation_steps
                        display_loss = avg_loss.item() if is_optimization_step else loss.item()

                        train_progress.update(
                            task,
                            advance=1 if is_optimization_step else 0,
                            loss=display_loss,
                            lr=current_lr,
                            step_time=step_time,
                            total_time=total_time,
                        )

                        if is_optimization_step:
                            self._log_metrics({
                                "train/loss": display_loss,
                                "train/learning_rate": current_lr,
                                "train/step_time": step_time,
                                "train/global_step": self._global_step,
                            })
                            self._epoch_train_losses.append(display_loss)

                        if disable_progress_bars and is_optimization_step and self._global_step % 20 == 0:
                            logger.info(
                                f"Step {self._global_step}/{cfg.optimization.steps} - "
                                f"Loss: {display_loss:.4f}, LR: {current_lr:.2e}, "
                                f"Time/Step: {step_time:.2f}s, Total Time: {total_time}",
                            )

                    if step % MEMORY_CHECK_INTERVAL == 0:
                        current_mem = get_gpu_memory_gb(device)
                        peak_mem_during_training = max(peak_mem_during_training, current_mem)

        train_end_time = time.time()
        end_mem = get_gpu_memory_gb(device)
        peak_mem = max(start_mem, end_mem, peak_mem_during_training)

        # Steps run by THIS job: a resumed job starts at _resume_global_step,
        # so dividing the full optimization.steps by its time overstates speed.
        steps_this_job = cfg.optimization.steps - self._resume_global_step
        if cfg.acceleration.compile_with_inductor:
            training_time = train_end_time - actual_training_start
            steps_per_second = (steps_this_job - COMPILE_WARMUP_STEPS) / training_time
        else:
            training_time = train_end_time - train_start_time
            steps_per_second = steps_this_job / training_time

        effective_batch_size = (
            cfg.optimization.batch_size
            * self._accelerator.num_processes
            * cfg.optimization.gradient_accumulation_steps
        )
        samples_per_second = steps_per_second * effective_batch_size

        stats = TrainingStats(
            total_time_seconds=train_end_time - train_start_time,
            training_time=training_time,
            compilation_time_seconds=compilation_time,
            steps_per_second=steps_per_second,
            samples_per_second=samples_per_second,
            peak_gpu_memory_gb=peak_mem,
            num_processes=self._accelerator.num_processes,
            global_batch_size=effective_batch_size,
        )

        final_step = self._global_step if self._global_step > 0 else cfg.optimization.steps
        comfy_path: Path = Path(cfg.output_dir) / "checkpoints" / f"model_weights_step_{final_step:05d}.safetensors"

        train_progress.remove_task(task)
        self._accelerator.end_training()

        if IS_MAIN_PROCESS:
            saved_path = self._save_checkpoint()
            comfy_path = saved_path

            # Final pass, unless the periodic one just rendered this step or an
            # explicit visualization.steps list was given.
            if (
                self._visualizer is not None
                and cfg.visualization.steps is None
                and not self._visualization_due(self._global_step)
            ):
                self._run_visualization(self._global_step)

            if os.getenv("LTXV_SAVE_COMFY", "0").lower() in ("1", "true", "yes", "y"):
                maybe_comfy = saved_path.parent / f"comfy_{saved_path.name}"
                try:
                    convert_checkpoint(
                        input_path=str(saved_path),
                        to_comfy=True,
                        output_path=str(maybe_comfy),
                    )
                    comfy_path = maybe_comfy
                except Exception as e:
                    logger.warning(f"Comfy conversion failed, continuing without it: {e}")

            self._log_training_stats(stats)

            if self._wandb_run is not None:
                self._log_metrics({
                    "stats/total_time_minutes": stats.total_time_seconds / 60,
                    "stats/training_time_minutes": stats.training_time / 60,
                    "stats/compilation_time_seconds": stats.compilation_time_seconds,
                    "stats/steps_per_second": stats.steps_per_second,
                    "stats/samples_per_second": stats.samples_per_second,
                    "stats/peak_gpu_memory_gb": stats.peak_gpu_memory_gb,
                })
                self._wandb_run.finish()

        self._accelerator.end_training()
        return comfy_path, stats

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def _training_step(self, batch: dict[str, dict[str, Tensor]]) -> Tensor:
        training_batch = self._prepare_batch(batch, self._timestep_sampler)

        if self._scalar_embedding is not None and training_batch.scalars is not None:
            training_batch = self._embed_and_concat_scalars(training_batch)

        model_inputs = self._prepare_model_inputs(training_batch)

        for key, val in model_inputs.items():
            if isinstance(val, torch.Tensor) and val.is_floating_point():
                if torch.isnan(val).any() or torch.isinf(val).any():
                    logger.error(f"NaN/Inf in model_inputs['{key}']: shape={val.shape}")
                    raise ValueError(f"NaN/Inf detected in model_inputs['{key}']")

        model_pred = self._transformer(**model_inputs)[0]

        if torch.isnan(model_pred).any() or torch.isinf(model_pred).any():
            logger.error("NaN/Inf in model_pred after transformer forward pass")
            raise ValueError("NaN/Inf detected in model prediction")

        loss = self._compute_loss(model_pred, training_batch)

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.error(f"NaN/Inf in loss: {loss.item()}")
            raise ValueError("NaN/Inf detected in loss")

        return loss

    def _embed_and_concat_scalars(
        self, batch: TrainingBatch, scalar_embedding: nn.Module | None = None
    ) -> TrainingBatch:
        """Embed the scalar conditions; they take the place of the text-prompt embeddings.

        Validation passes the unwrapped ``scalar_embedding`` so rank 0 alone
        never launches a DDP collective.
        """
        scalar_embedding = scalar_embedding if scalar_embedding is not None else self._scalar_embedding

        if torch.isnan(batch.scalars).any() or torch.isinf(batch.scalars).any():
            raise ValueError("Input scalars contain NaN or Inf values")

        for name, param in scalar_embedding.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                raise ValueError(f"Scalar embedding weights contain NaN or Inf: {name}")

        scalar_embeds = scalar_embedding(batch.scalars)

        if torch.isnan(scalar_embeds).any() or torch.isinf(scalar_embeds).any():
            # Inputs and weights are finite (checked above), so the NaN comes
            # from forward() itself: log each stage before raising. Skipping the
            # step instead would need cross-rank agreement to avoid a DDP hang.
            with torch.no_grad():
                # DDP does not proxy custom members such as .normalize()/.fourier.
                unwrapped = self._accelerator.unwrap_model(scalar_embedding)
                normalized = unwrapped.normalize(batch.scalars)
                fourier_out = unwrapped.fourier(normalized)
                logger.error(
                    f"Scalar embeddings contain NaN or Inf values\n"
                    f"  raw scalars       : {batch.scalars.tolist()}\n"
                    f"  normalized [0,1]  : {normalized.tolist()}\n"
                    f"  fourier features  : nan={torch.isnan(fourier_out).any().item()} "
                    f"inf={torch.isinf(fourier_out).any().item()} "
                    f"min={fourier_out.min().item():.4g} max={fourier_out.max().item():.4g}\n"
                    f"  scalar_embeds     : nan={torch.isnan(scalar_embeds).any().item()} "
                    f"inf={torch.isinf(scalar_embeds).any().item()}"
                )
            raise ValueError("Scalar embeddings contain NaN or Inf values")

        batch_size = batch.scalars.shape[0]
        num_scalar_tokens = scalar_embeds.shape[1]

        # Scalar-only mode: use only scalar embeddings
        new_prompt_embeds = scalar_embeds
        new_attention_mask = torch.ones(
            batch_size, num_scalar_tokens,
            dtype=batch.prompt_attention_mask.dtype,
            device=batch.prompt_attention_mask.device,
        )

        return batch.model_copy(
            update={
                "prompt_embeds": new_prompt_embeds,
                "prompt_attention_mask": new_attention_mask,
            }
        )

    # ------------------------------------------------------------------
    # Model loading and setup
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        components = load_ltxv_components(
            model_source=self._config.model.model_source,
            transformer_dtype=torch.float32,
            vae_dtype=torch.bfloat16,
        )

        self._scheduler = components.scheduler
        self._vae = components.vae
        self._transformer = components.transformer

        self._vae.requires_grad_(False)
        self._transformer.requires_grad_(False)

    def _init_scalar_embedding(self) -> None:
        if self._config.scalar_conditioning.enabled:
            logger.info(
                f"Initializing scalar embeddings for: {self._config.scalar_conditioning.scalar_names}"
            )
            self._scalar_embedding = ScalarEmbedding(self._config.scalar_conditioning)
        else:
            self._scalar_embedding = None

    def _compile_transformer(self) -> None:
        if not self._config.acceleration.compile_with_inductor:
            return
        torch._dynamo.config.inline_inbuilt_nn_modules = True
        torch._dynamo.config.cache_size_limit = 128
        compile_module = partial(torch.compile, mode=self._config.acceleration.compilation_mode)
        self._transformer.transformer_blocks = nn.ModuleList(
            [compile_module(block) for block in self._transformer.transformer_blocks],
        )

    def _collect_trainable_params(self) -> None:
        self._transformer.requires_grad_(True)  # full fine-tune of the transformer
        self._trainable_params = [p for p in self._transformer.parameters() if p.requires_grad]

        if self._scalar_embedding is not None:
            self._trainable_params.extend(self._scalar_embedding.parameters())
            logger.debug(
                f"Scalar embedding params: {sum(p.numel() for p in self._scalar_embedding.parameters()):,}"
            )

        logger.debug(f"Total trainable params count: {sum(p.numel() for p in self._trainable_params):,}")

    def _init_timestep_sampler(self) -> None:
        sampler_cls = SAMPLERS[self._config.flow_matching.timestep_sampling_mode]
        self._timestep_sampler = sampler_cls(**self._config.flow_matching.timestep_sampling_params)

    def _load_checkpoint(self) -> None:
        if not self._config.model.load_checkpoint:
            return

        checkpoint_path = self._find_checkpoint(self._config.model.load_checkpoint)
        if not checkpoint_path:
            logger.warning(f"Could not find checkpoint at {self._config.model.load_checkpoint}")
            return

        transformer = self._accelerator.unwrap_model(self._transformer)

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = load_file(checkpoint_path)
        transformer.load_state_dict(state_dict)

        if self._scalar_embedding is not None:
            scalar_checkpoint = self._find_scalar_checkpoint(checkpoint_path)
            if scalar_checkpoint:
                logger.info(f"Loading scalar embedding from {scalar_checkpoint}")
                scalar_state_dict = load_file(scalar_checkpoint)
                self._scalar_embedding.load_state_dict(scalar_state_dict)

        self._load_resume_state(checkpoint_path)

    @staticmethod
    def _state_file_for(ckpt_path: str | Path) -> Path:
        p = Path(ckpt_path)
        # model_weights_step_00500.safetensors -> model_weights_step_00500.state.pt
        return p.parent / (p.stem + ".state.pt")

    def _load_resume_state(self, checkpoint_path: Path) -> None:
        """Read the checkpoint's sibling .state.pt; train() applies it once the optimizer exists."""
        if self._config.resume_weights_only:
            logger.info(
                "resume_weights_only=true: loaded the checkpoint's weights but ignoring its "
                "training state. Optimizer, LR schedule and step counter start fresh, so the "
                "configured learning_rate/steps take effect (warm restart)."
            )
            return
        state_path = self._state_file_for(checkpoint_path)
        if not state_path.exists():
            logger.warning(
                f"model.load_checkpoint set but no training-state file at {state_path}; "
                "model weights will be warm-started, but optimizer/scheduler/step counter "
                "will start fresh (step 0)."
            )
            return
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if "optimizer" in state:
            # Undo _save_training_state's bf16 downcast -- the live optimizer
            # (and AdamW's own arithmetic) expects fp32 moment buffers.
            state["optimizer"] = self._cast_optimizer_state(state["optimizer"], torch.float32)
        self._resume_state = state
        self._resume_global_step = int(state.get("global_step", 0))
        logger.info(
            f"Loaded training state from {state_path}: resuming at global_step "
            f"{self._resume_global_step}"
        )

    def _apply_resume_state(self) -> None:
        """Restore optimizer/scheduler/RNG (after set_seed, so the restored RNG wins)."""
        state = self._resume_state
        if state is None:
            return
        if "optimizer" in state:
            self._optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None and self._lr_scheduler is not None:
            self._lr_scheduler.load_state_dict(state["scheduler"])
        rng = state.get("rng")
        if rng is not None:
            import random as _random

            import numpy as _np

            if rng.get("python") is not None:
                _random.setstate(rng["python"])
            if rng.get("numpy") is not None:
                _np.random.set_state(rng["numpy"])
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])

        # Free the loaded copy of the optimizer state (~15 GB in fp32 for the
        # 2B transformer); keeping it alive starves later checkpoint saves.
        self._resume_state = None

    def _find_scalar_checkpoint(self, model_checkpoint: Path) -> Path | None:
        stem = model_checkpoint.stem
        if "_step_" not in stem:
            return None
        step_str = stem.split("_step_")[-1]
        scalar_filename = f"scalar_embedding_step_{step_str}.safetensors"
        scalar_path = model_checkpoint.parent / scalar_filename
        return scalar_path if scalar_path.exists() else None

    def _prepare_models_for_training(self) -> None:
        prepare = self._accelerator.prepare
        self._vae = prepare(self._vae).to("cpu")

        if self._config.optimization.enable_gradient_checkpointing:
            self._transformer.enable_gradient_checkpointing()
        self._transformer = prepare(self._transformer)

        if self._scalar_embedding is not None:
            self._scalar_embedding = prepare(self._scalar_embedding)

    @staticmethod
    def _find_checkpoint(checkpoint_path: str | Path) -> Path | None:
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_file():
            if not checkpoint_path.suffix == ".safetensors":
                raise ValueError(f"Checkpoint file must have a .safetensors extension: {checkpoint_path}")
            return checkpoint_path

        if checkpoint_path.is_dir():
            # Not "*step_*": that would also match scalar_embedding_step_*.
            checkpoints = list(checkpoint_path.rglob("model_weights_step_*.safetensors"))
            if not checkpoints:
                return None

            def _get_step_num(p: Path) -> int:
                try:
                    return int(p.stem.split("step_")[1])
                except (IndexError, ValueError):
                    return -1

            return max(checkpoints, key=_get_step_num)

        raise ValueError(f"Invalid checkpoint path: {checkpoint_path}. Must be a file or directory.")

    # ------------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------------

    def _init_optimizer(self) -> None:
        opt_cfg = self._config.optimization
        optimizer = AdamW(self._trainable_params, lr=opt_cfg.learning_rate)

        optimizer = self._accelerator.prepare(optimizer)
        lr_scheduler = self._create_scheduler(optimizer)
        self._optimizer, self._lr_scheduler = optimizer, lr_scheduler
        self._apply_resume_state()

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> LRScheduler | None:
        scheduler_type = self._config.optimization.scheduler_type
        steps = self._config.optimization.steps
        params = dict(self._config.optimization.scheduler_params or {})

        if scheduler_type is None or scheduler_type == "constant":
            return None

        if scheduler_type == "linear":
            return LinearLR(optimizer, start_factor=params.pop("start_factor", 1.0),
                            end_factor=params.pop("end_factor", 0.1), total_iters=steps, **params)
        elif scheduler_type == "cosine":
            return CosineAnnealingLR(optimizer, T_max=steps, eta_min=params.pop("eta_min", 0), **params)
        elif scheduler_type == "cosine_with_restarts":
            return CosineAnnealingWarmRestarts(optimizer, T_0=params.pop("T_0", steps // 4),
                                               T_mult=params.pop("T_mult", 1),
                                               eta_min=params.pop("eta_min", 5e-5), **params)
        elif scheduler_type == "polynomial":
            return PolynomialLR(optimizer, total_iters=steps, power=params.pop("power", 1.0), **params)
        elif scheduler_type == "step":
            return StepLR(optimizer, step_size=params.pop("step_size", steps // 2),
                          gamma=params.pop("gamma", 0.1), **params)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    # ------------------------------------------------------------------
    # Data loader
    # ------------------------------------------------------------------

    def _init_dataloader(self) -> None:
        if self._dataset is None:
            data_sources = self._get_data_sources()
            self._dataset = PrecomputedDataset(
                self._config.data.preprocessed_data_root,
                data_sources=data_sources,
            )
            logger.debug(f"Loaded dataset with {len(self._dataset):,} samples from sources: {list(data_sources)}")

        dataloader = DataLoader(
            self._dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self._config.data.num_dataloader_workers,
            pin_memory=self._config.data.num_dataloader_workers > 0,
        )
        self._dataloader = self._accelerator.prepare(dataloader)

    def _record_latent_provenance(self) -> None:
        """Record which VAE produced the training latents (latent_provenance.json).

        The DiT is only valid with that VAE; inference checks this fingerprint
        and refuses to decode with a different one.
        """
        if not IS_MAIN_PROCESS:
            return

        norm_path = Path(self._config.data.preprocessed_data_root) / "normalization.json"
        if not norm_path.is_file():
            logger.warning(
                f"No normalization.json at {norm_path}: the latent space these latents came "
                "from cannot be recorded, so inference will not be able to verify that it is "
                "decoding with the matching VAE. Re-run preprocess_dataset.py to produce one."
            )
            return

        payload = json.loads(norm_path.read_text())
        fingerprint = payload.get("latent_fingerprint")
        provenance = {
            "latent_fingerprint": fingerprint,
            "vae_checkpoint": payload.get("vae_checkpoint"),
            "channel_mean": payload.get("channel_mean"),
            "channel_std": payload.get("channel_std"),
            "normalization_clip": payload.get("normalization_clip"),
            "preprocessed_data_root": str(Path(self._config.data.preprocessed_data_root).resolve()),
        }
        out = Path(self._config.output_dir) / "latent_provenance.json"
        out.write_text(json.dumps(provenance, indent=2))
        logger.info(
            f"Training on latents from VAE {payload.get('vae_checkpoint')} "
            f"(fingerprint {fingerprint}); recorded in {out.name}"
        )

    def _init_val_dataloader(self) -> None:
        """Build the held-out loader. Not accelerator-prepared: validation runs on
        the main process only, over the whole split, so the reported number is one
        unambiguous average rather than a sharded one that depends on world size."""
        val_root = self._config.validation.data_root
        if val_root is None:
            return
        if not IS_MAIN_PROCESS:
            return

        data_sources = self._get_data_sources()
        self._val_dataset = PrecomputedDataset(val_root, data_sources=data_sources)
        if len(self._val_dataset) == 0:
            raise ValueError(f"Validation split at {val_root} is empty")

        train_root = Path(self._config.data.preprocessed_data_root).resolve()
        if Path(val_root).resolve() == train_root:
            raise ValueError(
                f"validation.data_root and data.preprocessed_data_root are the same directory "
                f"({train_root}). Validating on the training set measures nothing. Re-run "
                f"preprocess_dataset.py with --eval-sims to produce a held-out split."
            )

        # Fixed order + fixed RNG in _validate: successive val losses differ only by the model.
        self._val_dataloader = DataLoader(
            self._val_dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self._config.data.num_dataloader_workers,
            pin_memory=self._config.data.num_dataloader_workers > 0,
        )
        logger.info(f"Validation split: {len(self._val_dataset):,} samples from {val_root}")

    @torch.no_grad()
    def _validate(self) -> float | None:
        """Average held-out flow-matching loss.

        The noise-level draw dominates this loss's variance, so every pass uses
        the same fixed seed (the training RNG is restored afterwards): each
        validation scores the same sims at the same noise levels.
        """
        if self._val_dataloader is None:
            return None

        device = self._accelerator.device
        max_samples = self._config.validation.max_samples

        py_rng_state = random.getstate()
        cpu_rng_state = torch.get_rng_state()
        if device.type == "cuda":
            accel_rng_state = torch.cuda.get_rng_state_all()
        elif device.type == "xpu":
            accel_rng_state = torch.xpu.get_rng_state_all()
        else:
            accel_rng_state = None
        random.seed(self._config.seed)
        torch.manual_seed(self._config.seed)

        # Only rank 0 runs this, so use the unwrapped (non-DDP) modules.
        transformer = self._accelerator.unwrap_model(self._transformer)
        scalar_embedding = (
            self._accelerator.unwrap_model(self._scalar_embedding)
            if self._scalar_embedding is not None
            else None
        )
        transformer.eval()
        total_loss, seen = 0.0, 0
        try:
            for batch in self._val_dataloader:
                if max_samples and seen >= max_samples:
                    break
                batch = {
                    key: {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in value.items()}
                    if isinstance(value, dict)
                    else value
                    for key, value in batch.items()
                }
                val_batch = self._prepare_batch(batch, self._timestep_sampler, force_conditioning=True)
                if scalar_embedding is not None and val_batch.scalars is not None:
                    val_batch = self._embed_and_concat_scalars(val_batch, scalar_embedding)
                model_pred = transformer(**self._prepare_model_inputs(val_batch))[0]
                loss = self._compute_loss(model_pred, val_batch)

                total_loss += loss.item() * val_batch.batch_size
                seen += val_batch.batch_size
        finally:
            transformer.train()
            random.setstate(py_rng_state)
            torch.set_rng_state(cpu_rng_state)
            if accel_rng_state is not None:
                if device.type == "cuda":
                    torch.cuda.set_rng_state_all(accel_rng_state)
                elif device.type == "xpu":
                    torch.xpu.set_rng_state_all(accel_rng_state)

        return total_loss / seen if seen else None

    # ------------------------------------------------------------------
    # Checkpointing and logging
    # ------------------------------------------------------------------

    def _save_checkpoint(self) -> Path:
        save_dir = Path(self._config.output_dir) / "checkpoints"
        save_dir.mkdir(exist_ok=True, parents=True)

        filename = f"model_weights_step_{self._global_step:05d}.safetensors"
        saved_weights_path = save_dir / filename
        rel_saved_weights_path = saved_weights_path.relative_to(self._config.output_dir)

        unwrapped_model = self._accelerator.unwrap_model(self._transformer)
        state_dict = unwrapped_model.state_dict()
        save_file(state_dict, saved_weights_path)
        logger.info(f"Model weights for step {self._global_step} saved in {rel_saved_weights_path}")

        scalar_state_dict = None
        if self._scalar_embedding is not None:
            scalar_filename = f"scalar_embedding_step_{self._global_step:05d}.safetensors"
            scalar_path = save_dir / scalar_filename
            unwrapped_scalar = self._accelerator.unwrap_model(self._scalar_embedding)
            scalar_state_dict = unwrapped_scalar.state_dict()
            save_file(scalar_state_dict, scalar_path)
            logger.info(f"Scalar embedding weights for step {self._global_step} saved")

        self._save_training_state(self._state_file_for(saved_weights_path))

        # Weights-only "best" slot (requires validation), never pruned, so
        # evaluation can always refer to model_weights_best.safetensors.
        fresh_val_loss = self._last_val_loss if self._last_val_step == self._global_step else None
        if fresh_val_loss is not None and fresh_val_loss < self._best_val_loss:
            self._best_val_loss = fresh_val_loss
            self._best_step = self._global_step
            best_path = save_dir / "model_weights_best.safetensors"
            save_file(state_dict, best_path)
            if scalar_state_dict is not None:
                save_file(scalar_state_dict, save_dir / "scalar_embedding_best.safetensors")
            logger.info(
                f"New best checkpoint (step {self._global_step}, val_loss={fresh_val_loss:.6f}): "
                f"{best_path.relative_to(self._config.output_dir)}"
            )

        self._checkpoint_paths.append(saved_weights_path)
        self._cleanup_checkpoints()
        return saved_weights_path

    @staticmethod
    def _cast_optimizer_state(obj, dtype: torch.dtype, device: str | None = None):
        """Recursively cast the float tensors of an optimizer state_dict.

        AdamW's "step" counter is excluded: it drives bias correction, and bf16
        would round it (e.g. 9720 -> 9728). With device="cpu" each tensor leaves
        the GPU as it is cast, so saving never holds a second full copy of the
        optimizer state in GPU memory.
        """
        if isinstance(obj, torch.Tensor):
            if obj.is_floating_point():
                return obj.to(device=device, dtype=dtype) if device is not None else obj.to(dtype)
            return obj.to(device=device) if device is not None else obj
        if isinstance(obj, dict):
            return {
                k: (
                    LtxvTrainer._cast_optimizer_state(v, dtype, device)
                    if k != "step"
                    else (v.to(device=device) if device is not None and isinstance(v, torch.Tensor) else v)
                )
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [LtxvTrainer._cast_optimizer_state(v, dtype, device) for v in obj]
        return obj

    def _save_training_state(self, state_path: Path) -> None:
        """Write optimizer/scheduler/RNG state as <stem>.state.pt next to the weights.

        Adam's moment buffers are stored in bf16 (the live optimizer stays fp32
        and _load_resume_state upcasts again), halving the ~15 GB file. Casting
        happens on the CPU to avoid a GPU memory spike at save time.
        """
        import numpy as np

        rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        state = {
            "global_step": self._global_step,
            "optimizer": self._cast_optimizer_state(
                self._optimizer.state_dict(), torch.bfloat16, device="cpu"
            ),
            "scheduler": self._lr_scheduler.state_dict() if self._lr_scheduler is not None else None,
            "rng": rng,
        }
        torch.save(state, state_path)
        logger.info(f"Training state saved: {state_path.relative_to(self._config.output_dir)}")

    def _cleanup_checkpoints(self) -> None:
        if 0 < self._config.checkpoints.keep_last_n < len(self._checkpoint_paths):
            checkpoints_to_remove = self._checkpoint_paths[: -self._config.checkpoints.keep_last_n]
            for old_checkpoint in checkpoints_to_remove:
                files_to_remove = [old_checkpoint, self._state_file_for(old_checkpoint)]
                scalar_checkpoint = self._find_scalar_checkpoint(old_checkpoint)
                if scalar_checkpoint is not None:
                    files_to_remove.append(scalar_checkpoint)
                for f in files_to_remove:
                    if f.exists():
                        f.unlink()
                        logger.debug(f"Removed old checkpoint: {f}")
            self._checkpoint_paths = self._checkpoint_paths[-self._config.checkpoints.keep_last_n :]

        # Only the newest checkpoint is resumed from; older kept ones are
        # weights-only fallbacks, so their (large) .state.pt is dropped.
        for older_checkpoint in self._checkpoint_paths[:-1]:
            state_path = self._state_file_for(older_checkpoint)
            if state_path.exists():
                state_path.unlink()
                logger.debug(f"Removed optimizer state for older kept checkpoint: {state_path}")

    def _save_epoch_log(self) -> None:
        if not IS_MAIN_PROCESS:
            return
        log_path = Path(self._config.output_dir) / "epoch_log.json"
        with open(log_path, "w") as f:
            json.dump(self._epoch_losses, f, indent=2)

    def _save_config(self) -> None:
        if not IS_MAIN_PROCESS:
            return
        config_path = Path(self._config.output_dir) / "training_config.yaml"
        with open(config_path, "w") as f:
            # mode="json" keeps enums as plain strings, so yaml.safe_load can read it back.
            yaml.dump(self._config.model_dump(mode="json"), f, default_flow_style=False, indent=2)
        logger.info(f"Training configuration saved to: {config_path.relative_to(self._config.output_dir)}")

    def _setup_accelerator(self) -> None:
        self._accelerator = Accelerator(
            mixed_precision=self._config.acceleration.mixed_precision_mode,
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
        )
        if self._accelerator.num_processes > 1:
            logger.info(f"Distributed training enabled with {self._accelerator.num_processes} processes")
            effective_batch_size = (
                self._config.optimization.batch_size
                * self._accelerator.num_processes
                * self._config.optimization.gradient_accumulation_steps
            )
            logger.info(f"Global batch size: {effective_batch_size}")

    def _init_wandb(self) -> None:
        if not self._config.wandb.enabled or not IS_MAIN_PROCESS:
            self._wandb_run = None
            return
        wandb_config = self._config.wandb
        run = wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity,
            name=Path(self._config.output_dir).name,
            tags=wandb_config.tags,
            config=self._config.model_dump(),
        )
        self._wandb_run = run

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)

    @staticmethod
    def _log_training_stats(stats: TrainingStats) -> None:
        stats_str = (
            "Training Statistics:\n"
            f" - Total time: {stats.total_time_seconds / 60:.1f} minutes\n"
            f" - Training time: {stats.training_time / 60:.1f} minutes\n"
            f" - Training speed: {stats.steps_per_second:.2f} steps/second\n"
            f" - Samples/second: {stats.samples_per_second:.2f}\n"
            f" - Peak GPU memory: {stats.peak_gpu_memory_gb:.2f} GB"
        )
        if stats.compilation_time_seconds is not None:
            stats_str += f"\n - Compilation time: {stats.compilation_time_seconds:.1f} seconds"
        if stats.num_processes > 1:
            stats_str += f"\n - Number of processes: {stats.num_processes}"
            stats_str += f"\n - Global batch size: {stats.global_batch_size}"
        logger.info(stats_str)

    @staticmethod
    def _print_config(config: BaseModel) -> None:
        if not IS_MAIN_PROCESS:
            return

        from rich.table import Table

        table = Table(title="Training Configuration", show_header=True, header_style="bold green")
        table.add_column("Parameter", style="bold white")
        table.add_column("Value", style="bold cyan")

        def flatten_config(cfg: BaseModel, prefix: str = "") -> list[tuple[str, str]]:
            rows = []
            for field, value in cfg:
                full_field = f"{prefix}.{field}" if prefix else field
                if isinstance(value, BaseModel):
                    rows.extend(flatten_config(value, full_field))
                elif isinstance(value, (list, tuple, set)):
                    value_str = ", ".join(str(item) for item in value)
                    if len(value_str) > 70:
                        value_str = value_str[:70] + "..."
                    rows.append((full_field, value_str))
                else:
                    value_str = str(value)
                    if len(value_str) > 70:
                        value_str = value_str[:70] + "..."
                    rows.append((full_field, value_str))
            return rows

        skip_prefixes = []
        if hasattr(config, "scalar_conditioning") and not config.scalar_conditioning.enabled:
            skip_prefixes.append("scalar_conditioning.")

        for param, value in flatten_config(config):
            if any(param.startswith(p) for p in skip_prefixes):
                continue
            table.add_row(param, value)

        rich.print(table)

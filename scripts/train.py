#!/usr/bin/env python
# Copied from LTX-Video-Trainer (https://github.com/Lightricks/LTX-Video-Trainer).

"""
Train the DiT on precomputed VAE latents (see scripts/preprocess_dataset.py).

Usage:
    python scripts/train.py configs/dit/train.yaml
"""

from pathlib import Path

import typer
import yaml
from rich.console import Console

from windinet.config import LtxvTrainerConfig
from windinet.training.dit_trainer import LtxvTrainer

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Train LTXV models using configuration from YAML files.",
)


@app.command()
def main(config_path: str = typer.Argument(..., help="Path to YAML configuration file")) -> None:
    """Train the model using the provided configuration file."""
    # Load the configuration from the YAML file
    config_path = Path(config_path)
    if not config_path.exists():
        typer.echo(f"Error: Configuration file {config_path} does not exist.")
        raise typer.Exit(code=1)

    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    # Convert the loaded data to the LtxvTrainerConfig object
    try:
        trainer_config = LtxvTrainerConfig(**config_data)
    except Exception as e:
        typer.echo(f"Error: Invalid configuration data: {e}")
        raise typer.Exit(code=1) from e

    # Initialize the training process
    trainer = LtxvTrainer(trainer_config)
    trainer.train()


if __name__ == "__main__":
    app()

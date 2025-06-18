import os
from pathlib import Path
from typing import Any

import torch
import yaml
from typeguard import typechecked

from daart.model.base import Segmenter


@typechecked
class Model:
    """High-level API wrapper for daart models.

    This class manages both the model and the training/inference processes.
    """

    def __init__(
        self,
        model: Segmenter,
        config: dict[str, Any],
        model_dir: str | Path | None = None
    ) -> None:
        """Initialize with model and config."""
        self.model = model
        self.config = config
        self.model_dir = Path(model_dir) if model_dir is not None else None

    @classmethod
    def from_dir(cls, model_dir: str | Path):
        """Load a model from a directory.

        Parameters
        ----------
        model_dir: Path to directory containing model checkpoint and config

        Returns
        -------
        Initialized model wrapper

        """

        model_dir = Path(model_dir)

        config_path = model_dir / 'config.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)

        model = Segmenter(config)

        # Load best weights
        checkpoint_path = list(model_dir.rglob('*best.ckpt'))[0]
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict['state_dict'])
        print(f'Loaded model weights from {checkpoint_path}')

        return cls(model, config, model_dir)

    @classmethod
    def from_config(cls, config_path: str | Path | dict):
        """Create a new model from a config file.

        Parameters
        ----------
        config_path: Path to config file or config dict

        Returns
        -------
        Initialized model wrapper

        """
        if not isinstance(config_path, dict):
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            config = config_path

        model = Segmenter(config)

        return cls(model, config, model_dir=None)

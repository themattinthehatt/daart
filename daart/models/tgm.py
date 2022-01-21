"""Temporal Gaussian Mixture model implemented in PyTorch."""

import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from daart.models.base import BaseModel

# to ignore imports for sphix-autoapidoc
__all__ = []


class TGM(BaseModel):

    def __init__(self, hparams):
        pass

    def build_model(self):
        """Construct the model using hparams."""
        pass

    def forward(self, x, **kwargs):
        """Process input data.

        Parameters
        ----------
        x : torch.Tensor object
            input data

        Returns
        -------
        dict
            - 'labels' (torch.Tensor): model classification
            - 'prediction' (torch.Tensor): one-step-ahead prediction
            - 'embedding' (torch.Tensor): behavioral embedding used for classification/prediction

        """
        pass

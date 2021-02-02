"""Supervised models implemented in PyTorch."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
import behavenet.fitting.losses as losses
from behavenet.models.base import BaseModule, BaseModel

# to ignore imports for sphix-autoapidoc
__all__ = ['TemporalMLP', 'TemporalConv', 'LSTM', 'TGM']


class TemporalMLP(BaseModel):

    def __init__(self, hparams):
        pass

    def __str__(self):
        """Pretty print the model architecture."""
        pass

    def build_model(self):
        """Construct the model using hparams."""
        pass

    def forward(self, x, **kwargs):
        """Process input data.

        Parameters
        ----------
        x : :obj:`torch.Tensor` object
            input data

        Returns
        -------

        """
        pass

    def loss(self, data, accumulate_grad=True):
        """Calculate loss for model.

        Parameters
        ----------
        data : dict
            batch of data; keys should include 'images' and 'masks', if necessary
        accumulate_grad : bool, optional
            accumulate gradient for training step

        Returns
        -------
        :obj:`dict`
            - 'loss' (:obj:`float`): mse loss

        """
        pass


class LSTM(BaseModel):

    def __init__(self, hparams):
        pass

    def __str__(self):
        """Pretty print the model architecture."""
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

        """
        pass

    def loss(self, data, accumulate_grad=True):
        """Calculate loss for model.

        Parameters
        ----------
        data : dict
            batch of data; keys should include 'images' and 'masks', if necessary
        accumulate_grad : bool, optional
            accumulate gradient for training step

        Returns
        -------
        dict
            - 'loss' (float): mse loss

        """
        pass


class TGM(BaseModel):

    def __init__(self, hparams):
        pass

    def __str__(self):
        """Pretty print the model architecture."""
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

        """
        pass

    def loss(self, data, accumulate_grad=True):
        """Calculate loss for model.

        Parameters
        ----------
        data : dict
            batch of data; keys should include 'images' and 'masks', if necessary
        accumulate_grad : bool, optional
            accumulate gradient for training step

        Returns
        -------
        dict
            - 'loss' (float): mse loss

        """
        pass

"""Temporal MLP model implemented in PyTorch."""

import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from daart.models.base import BaseModel, get_activation_func_from_str

# to ignore imports for sphix-autoapidoc
__all__ = ['TemporalMLP']


class TemporalMLP(BaseModel):
    """MLP network with initial 1D convolution layer."""

    def __init__(self, hparams, type='encoder', in_size=None, hid_size=None, out_size=None):

        super().__init__()
        self.hparams = hparams

        self.activation_func = get_activation_func_from_str(self.hparams['activation'])

        self.model = nn.ModuleList()
        if type == 'encoder':
            in_size_ = hparams['input_size'] if in_size is None else in_size
            hid_size_ = hparams['n_hid_units'] if hid_size is None else hid_size
            out_size_ = hparams['n_hid_units'] if out_size is None else out_size
            self.build_encoder(in_size=in_size_, hid_size=hid_size_, out_size=out_size_)
        else:
            in_size_ = hparams['n_hid_units'] if in_size is None else in_size
            hid_size_ = hparams['n_hid_units'] if hid_size is None else hid_size
            out_size_ = hparams['input_size'] if out_size is None else out_size
            self.build_decoder(in_size=in_size_, hid_size=hid_size_, out_size=out_size_)

    def build_encoder(self, in_size, hid_size, out_size):
        """Construct the encoder using hparams."""

        global_layer_num = 0

        # -------------------------------------------------------------
        # first layer is 1d conv for incorporating past/future activity
        # -------------------------------------------------------------
        layer = nn.Conv1d(
            in_channels=in_size,
            out_channels=hid_size,
            kernel_size=self.hparams['n_lags'] * 2 + 1,  # window around t
            padding=self.hparams['n_lags'])  # same output
        name = str('conv1d_layer_%02i' % global_layer_num)
        self.model.add_module(name, layer)

        # add activation
        if self.hparams['n_hid_layers'] == 0:
            activation = None  # cross entropy loss handles this
        else:
            activation = self.activation_func
        if activation:
            name = '%s_%02i' % (self.hparams['activation'], global_layer_num)
            self.model.add_module(name, activation)

        # update layer info
        global_layer_num += 1

        # -------------------------------------------------------------
        # remaining layers
        # -------------------------------------------------------------
        # loop over hidden layers (0 layers <-> linear model)
        for i_layer in range(self.hparams['n_hid_layers']):

            if i_layer == self.hparams['n_hid_layers'] - 1:
                activation = None  # cross entropy loss handles this
                out_size_ = out_size
            else:
                activation = self.activation_func
                out_size_ = hid_size

            # add layer
            layer = nn.Linear(in_features=hid_size, out_features=out_size_)
            name = str('dense_layer_%02i' % global_layer_num)
            self.model.add_module(name, layer)

            # add activation
            if activation:
                name = '%s_%02i' % (self.hparams['activation'], global_layer_num)
                self.model.add_module(name, activation)

            # update layer info
            global_layer_num += 1

        return global_layer_num

    def build_decoder(self, in_size, hid_size, out_size):
        """Construct the vanilla MLP decoder using hparams."""

        global_layer_num = 0

        in_size_ = in_size

        # loop over hidden layers (0 layers <-> linear model)
        for i_layer in range(self.hparams['n_hid_layers'] + 1):

            if i_layer == self.hparams['n_hid_layers']:
                out_size_ = out_size
            else:
                out_size_ = hid_size

            # add layer
            layer = nn.Linear(in_features=in_size_, out_features=out_size_)
            name = str('dense_layer_%02i' % global_layer_num)
            self.model.add_module(name, layer)

            # add activation
            if i_layer == self.hparams['n_hid_layers']:
                # no activation for final layer
                activation = None
            else:
                activation = self.activation_func
            if activation:
                name = '%s_%02i' % (self.hparams['activation'], global_layer_num)
                self.model.add_module(name, activation)

            # update layer info
            global_layer_num += 1
            in_size_ = out_size_

        return global_layer_num

    def forward(self, x, **kwargs):
        """Process input data.

        Parameters
        ----------
        x : torch.Tensor object
            input data of shape (n_sequences, sequence_length, n_markers)

        Returns
        -------
        torch.Tensor
            shape (n_sequences, sequence_length, n) where n is the embedding dimension if an
            encoder, or n_markers if a decoder/predictor

        """

        # push data through encoder to get latent embedding
        for name, layer in self.model.named_children():

            if name == 'conv1d_layer_00':
                # conv1d layer input is batch x in_channels x time
                # conv1d layer output is batch x out_channels x time

                # x = B x T x N (e.g. B = 2, T = 500, N = 16)
                # x.transpose(1, 2) -> x = B x N x T
                # x = layer(x) -> x = B x M x T
                # x.transpose(1, 2) -> x = B x T x M

                x = layer(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = layer(x)

        return x

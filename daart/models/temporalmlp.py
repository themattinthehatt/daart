"""Temporal MLP model implemented in PyTorch."""

import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from daart.models.base import BaseModel

# to ignore imports for sphix-autoapidoc
__all__ = ['TemporalMLP']


class TemporalMLP(BaseModel):
    """MLP network with initial 1D convolution layer."""

    def __init__(self, hparams, type='encoder'):
        super().__init__()
        self.hparams = hparams
        self.model = nn.ModuleList()
        if type == 'encoder':
            self.build_encoder()
        else:
            self.build_decoder()

    def build_encoder(self):
        """Construct the encoder using hparams."""

        global_layer_num = 0

        # -------------------------------------------------------------
        # first layer is 1d conv for incorporating past/future activity
        # -------------------------------------------------------------
        in_size = self.hparams['input_size']
        out_size = self.hparams['n_hid_units']
        layer = nn.Conv1d(
            in_channels=in_size,
            out_channels=out_size,
            kernel_size=self.hparams['n_lags'] * 2 + 1,  # window around t
            padding=self.hparams['n_lags'])  # same output
        name = str('conv1d_layer_%02i' % global_layer_num)
        self.model.add_module(name, layer)

        # add activation
        if self.hparams['n_hid_layers'] == 0:
            activation = None  # cross entropy loss handles this
        else:
            if self.hparams['activation'] == 'linear':
                activation = None
            elif self.hparams['activation'] == 'relu':
                activation = nn.ReLU()
            elif self.hparams['activation'] == 'lrelu':
                activation = nn.LeakyReLU(0.05)
            elif self.hparams['activation'] == 'sigmoid':
                activation = nn.Sigmoid()
            elif self.hparams['activation'] == 'tanh':
                activation = nn.Tanh()
            else:
                raise ValueError(
                    '"%s" is an invalid activation function' % self.hparams['activation'])

        if activation:
            name = '%s_%02i' % (self.hparams['activation'], global_layer_num)
            self.model.add_module(name, activation)

        # update layer info
        global_layer_num += 1
        in_size = out_size

        # -------------------------------------------------------------
        # remaining layers
        # -------------------------------------------------------------
        # loop over hidden layers (0 layers <-> linear model)
        for i_layer in range(self.hparams['n_hid_layers']):

            # add layer
            layer = nn.Linear(in_features=in_size, out_features=out_size)
            name = str('dense_layer_%02i' % global_layer_num)
            self.model.add_module(name, layer)

            # add activation
            if i_layer == self.hparams['n_hid_layers'] - 1:
                activation = None  # cross entropy loss handles this
            else:
                if self.hparams['activation'] == 'linear':
                    activation = None
                elif self.hparams['activation'] == 'relu':
                    activation = nn.ReLU()
                elif self.hparams['activation'] == 'lrelu':
                    activation = nn.LeakyReLU(0.05)
                elif self.hparams['activation'] == 'sigmoid':
                    activation = nn.Sigmoid()
                elif self.hparams['activation'] == 'tanh':
                    activation = nn.Tanh()
                else:
                    raise ValueError(
                        '"%s" is an invalid activation function' % self.hparams['activation'])

            if activation:
                name = '%s_%02i' % (self.hparams['activation'], global_layer_num)
                self.model.add_module(name, activation)

            # update layer info
            global_layer_num += 1
            in_size = out_size

        return global_layer_num

    def build_decoder(self):
        """Construct the vanilla MLP decoder using hparams."""

        global_layer_num = 0

        in_size = self.hparams['n_hid_units']

        # loop over hidden layers (0 layers <-> linear model)
        for i_layer in range(self.hparams['n_hid_layers'] + 1):

            if i_layer == self.hparams['n_hid_layers']:
                out_size = self.hparams['input_size']
            else:
                out_size = self.hparams['n_hid_units']

            # add layer
            layer = nn.Linear(in_features=in_size, out_features=out_size)
            name = str('dense_layer_%02i' % global_layer_num)
            self.model.add_module(name, layer)

            # add activation
            if i_layer == self.hparams['n_hid_layers']:
                # no activation for final layer
                activation = None
            else:
                if self.hparams['activation'] == 'linear':
                    activation = None
                elif self.hparams['activation'] == 'relu':
                    activation = nn.ReLU()
                elif self.hparams['activation'] == 'lrelu':
                    activation = nn.LeakyReLU(0.05)
                elif self.hparams['activation'] == 'sigmoid':
                    activation = nn.Sigmoid()
                elif self.hparams['activation'] == 'tanh':
                    activation = nn.Tanh()
                else:
                    raise ValueError(
                        '"%s" is an invalid activation function' % self.hparams['activation'])

            if activation:
                name = '%s_%02i' % (self.hparams['activation'], global_layer_num)
                self.model.add_module(name, activation)

            # update layer info
            global_layer_num += 1
            in_size = out_size

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

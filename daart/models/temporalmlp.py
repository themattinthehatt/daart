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

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.classifier = None
        self.predictor = None
        self.build_model()

    def __str__(self):
        """Pretty print model architecture."""
        format_str = '\nTemporalMLP architecture\n'
        format_str += '------------------------\n'
        for i, module in enumerate(self.classifier):
            format_str += str('    {}: {}\n'.format(i, module))
        if self.predictor is not None:
            format_str += '\nPredictor architecture\n'
            format_str += '------------------------\n'
            for i, module in enumerate(self.predictor):
                format_str += str('    {}: {}\n'.format(i, module))
        return format_str

    def build_model(self):
        """Construct the model using hparams."""

        self.classifier = nn.ModuleList()

        global_layer_num = 0

        in_size = self.hparams['input_size']

        # -------------------------------------------------------------
        # first layer is 1d conv for incorporating past/future activity
        # -------------------------------------------------------------
        if self.hparams['n_hid_layers'] == 0:
            out_size = self.hparams['output_size']
        else:
            out_size = self.hparams['n_hid_units']

        layer = nn.Conv1d(
            in_channels=in_size,
            out_channels=out_size,
            kernel_size=self.hparams['n_lags'] * 2 + 1,  # window around t
            padding=self.hparams['n_lags'])  # same output
        name = str('conv1d_layer_%02i' % global_layer_num)
        self.classifier.add_module(name, layer)

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
            self.classifier.add_module(name, activation)

        # update layer info
        global_layer_num += 1
        in_size = out_size

        # -------------------------------------------------------------
        # remaining layers
        # -------------------------------------------------------------
        # loop over hidden layers (0 layers <-> linear model)
        for i_layer in range(self.hparams['n_hid_layers']):

            if i_layer == self.hparams['n_hid_layers'] - 1:
                out_size = self.hparams['output_size']
            else:
                out_size = self.hparams['n_hid_units']

            # add layer
            layer = nn.Linear(in_features=in_size, out_features=out_size)
            name = str('dense_layer_%02i' % global_layer_num)
            self.classifier.add_module(name, layer)

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
                self.classifier.add_module(name, activation)

            # update layer info
            global_layer_num += 1
            in_size = out_size

        # -------------------------------------------------------------
        # decoding layers for next step prediction
        # -------------------------------------------------------------
        if self.hparams.get('lambda_pred', 0) > 0:

            self.predictor = nn.ModuleList()

            # loop over hidden layers (0 layers <-> linear model)
            for i_layer in range(self.hparams['n_hid_layers'] + 1):

                if i_layer == self.hparams['n_hid_layers']:
                    out_size = self.hparams['input_size']
                else:
                    out_size = self.hparams['n_hid_units']

                # add layer
                layer = nn.Linear(in_features=in_size, out_features=out_size)
                name = str('dense_layer_%02i' % global_layer_num)
                self.predictor.add_module(name, layer)

                # add activation
                if i_layer == self.hparams['n_hid_layers']:
                    activation = None  # linear
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
                    self.predictor.add_module(name, activation)

                # update layer info
                global_layer_num += 1
                in_size = out_size

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

        """
        for name, layer in self.classifier.named_children():

            if name == 'conv1d_layer_00':
                # input is batch x in_channels x time
                # output is batch x out_channels x time

                # x = T x N (T = 500, N = 16)
                # x.transpose(1, 0) -> x = N x T
                # x.unsqueeze(0) -> x = 1 x N x T
                # x = layer(x) -> x = 1 x M x T
                # x.squeeze() -> x = M x T
                # x.transpose(1, 0) -> x = T x M

                x = layer(x.transpose(1, 0).unsqueeze(0)).squeeze().transpose(1, 0)
            else:
                x = layer(x)

        if self.hparams.get('lambda_pred', 0) > 0:
            y = x
            for name, layer in self.predictor.named_children():
                y = layer(y)
        else:
            y = None

        return {'labels': x, 'prediction': y}

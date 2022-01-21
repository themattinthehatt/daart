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
        self.encoder = None
        self.classifier = None
        self.classifier_weak = None
        self.predictor = None
        self.task_predictor = None
        self.build_model()

    def build_model(self):
        """Construct the model using hparams."""

        self.encoder = nn.ModuleList()

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
        self.encoder.add_module(name, layer)

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
            self.encoder.add_module(name, activation)

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
            self.encoder.add_module(name, layer)

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
                self.encoder.add_module(name, activation)

            # update layer info
            global_layer_num += 1
            in_size = out_size

        final_encoder_size = out_size

        # -------------------------------------------------------------
        # classifier: single linear layer
        # -------------------------------------------------------------
        # linear classifier (hand labels)
        if self.hparams.get('lambda_strong') > 0:
            self.classifier = self._build_linear(
                global_layer_num=global_layer_num, name='classification',
                in_size=self.hparams['n_hid_units'], out_size=self.hparams['output_size'])

        # linear classifier (heuristic labels)
        if self.hparams.get('lambda_weak') > 0:
            self.classifier_weak = self._build_linear(
                global_layer_num=global_layer_num, name='classification',
                in_size=self.hparams['n_hid_units'], out_size=self.hparams['output_size'])

        # update layer info
        global_layer_num += 1

        # -------------------------------------------------------------
        # task regression: single linear layer
        # -------------------------------------------------------------
        if self.hparams.get('lambda_task') > 0:
            self.task_predictor = self._build_linear(
                global_layer_num=global_layer_num, name='regression',
                in_size=self.hparams['n_hid_units'], out_size=self.hparams['task_size'])

        # update layer info
        global_layer_num += 1

        # -------------------------------------------------------------
        # decoding layers for next step prediction
        # -------------------------------------------------------------
        in_size = final_encoder_size
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
                    self.predictor.add_module(name, activation)

                # update layer info
                global_layer_num += 1
                in_size = out_size

    def forward(self, x, **kwargs):
        """Process input data.

        Parameters
        ----------
        x : torch.Tensor object
            input data of shape (n_sequences, sequence_length, n_markers)

        Returns
        -------
        dict
            - 'labels' (torch.Tensor): model classification
            - 'labels' (torch.Tensor): model classification of weak/pseudo labels
            - 'prediction' (torch.Tensor): one-step-ahead prediction
            - 'task_prediction' (torch.Tensor): prediction of regression tasks
            - 'embedding' (torch.Tensor): behavioral embedding used for classification/prediction

        """

        # push data through encoder to get latent embedding
        for name, layer in self.encoder.named_children():

            if name == 'conv1d_layer_00':
                # input is batch x in_channels x time
                # output is batch x out_channels x time

                # x = B x T x N (e.g. B = 2, T = 500, N = 16)
                # x.transpose(1, 2) -> x = B x N x T
                # x = layer(x) -> x = B x M x T
                # x.transpose(1, 2) -> x = B x T x M

                x = layer(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = layer(x)

        # push embedding through classifier to get labels
        if self.hparams.get('lambda_strong', 0) > 0:
            z = self.classifier(x)
        else:
            z = None

        if self.hparams.get('lambda_weak', 0) > 0:
            z_weak = self.classifier_weak(x)
        else:
            z_weak = None

        # push embedding through linear layer to get task predictions
        if self.hparams.get('lambda_task', 0) > 0:
            w = self.task_predictor(x)
        else:
            w = None

        # push embedding through predictor network to get data at subsequent time points
        if self.hparams.get('lambda_pred', 0) > 0:
            y = x
            for name, layer in self.predictor.named_children():
                y = layer(y)
        else:
            y = None

        return {
            'labels': z,  # (n_sequences, sequence_length, n_classes)
            'labels_weak': z_weak,  # (n_sequences, sequence_length, n_classes)
            'prediction': y,  # (n_sequences, sequence_length, n_markers)
            'task_prediction': w,  # (n_sequences, sequence_length, n_tasks)
            'embedding': x  # (n_sequences, sequence_length, embedding_dim)
        }

"""Supervised models implemented in PyTorch."""

import numpy as np
import torch
from torch import nn
from daart.models.base import BaseModule, BaseModel

# to ignore imports for sphix-autoapidoc
__all__ = ['HardSegmenter', 'TemporalMLP', 'TemporalConv', 'LSTM', 'TGM']


class HardSegmenter(BaseModel):
    """General wrapper class for models without soft labels."""

    def __init__(self, hparams):
        """

        Parameters
        ----------
        hparams : dict
            - model_type (str): 'temporal-mlp' | 'temporal-conv' | 'lstm' | 'tgm'
            - input_size (int): number of input channels
            - output_size (int): number of classes
            - n_hid_layers (int): hidden layers of mlp/lstm network
            - n_hid_units (int): hidden units per layer
            - n_lags (int): number of lags in input data to use for temporal convolution
            - activation (str): 'linear' | 'relu' | 'lrelu' | 'sigmoid' | 'tanh'

        """
        super().__init__()
        self.hparams = hparams
        self.model = None
        self.build_model()

        # label loss based on cross entropy
        self._loss = nn.CrossEntropyLoss()

    def __str__(self):
        """Pretty print model architecture."""
        return self.model.__str__()

    def build_model(self):
        """Construct the model using hparams."""

        if self.hparams['model_type'] == 'temporal-mlp':
            self.model = TemporalMLP(self.hparams)
        elif self.hparams['model_type'] == 'temporal-conv':
            raise NotImplementedError
            # self.model = TemporalConv(self.hparams)
        elif self.hparams['model_type'] == 'lstm':
            raise NotImplementedError
            # self.model = LSTM(self.hparams)
        elif self.hparams['model_type'] == 'tgm':
            raise NotImplementedError
            # self.model = TGM(self.hparams)
        else:
            raise ValueError('"%s" is not a valid model type' % self.hparams['model_type'])

    def forward(self, x):
        """Process input data."""
        return self.model(x)

    def loss(self, data, accumulate_grad=True, **kwargs):
        """Calculate negative log-likelihood loss for supervised models.

        The batch is split into chunks if larger than a hard-coded `chunk_size` to keep memory
        requirements low; gradients are accumulated across all chunks before a gradient step is
        taken.

        Parameters
        ----------
        data : dict
            signals are of shape (1, time, n_channels)
        accumulate_grad : bool, optional
            accumulate gradient for training step

        Returns
        -------
        dict
            - 'loss' (float): total loss (negative log-like under specified noise dist)
            - 'fc' (float): fraction correct

        """

        predictors = data['markers'][0]
        targets = data['labels'][0]

        # push data through model
        outputs = self.model(predictors)

        # compute loss on allowed window of data
        loss = self._loss(outputs, targets)

        if accumulate_grad:
            loss.backward()

        # get loss value (weighted by batch size)
        loss_val = loss.item()

        outputs_val = outputs.cpu().detach().numpy()

        fc = accuracy_score(targets.cpu().detach().numpy(), np.argmax(outputs_val, axis=1))

        return {'loss': loss_val, 'fc': fc}


class TemporalMLP(BaseModel):
    """MLP network with initial 1D convolution layer."""

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.decoder = None
        self.build_model()

    def __str__(self):
        """Pretty print model architecture."""
        format_str = '\nNN architecture\n'
        format_str += '---------------\n'
        for i, module in enumerate(self.decoder):
            format_str += str('    {}: {}\n'.format(i, module))
        return format_str

    def build_model(self):
        """Construct the model using hparams."""

        self.decoder = nn.ModuleList()

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
        self.decoder.add_module(name, layer)

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
            self.decoder.add_module(name, activation)

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
            self.decoder.add_module(name, layer)

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
                self.decoder.add_module(name, activation)

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
        torch.Tensor
            mean prediction of model

        """
        for name, layer in self.decoder.named_children():

            if name == 'conv1d_layer_00':
                # input is batch x in_channels x time
                # output is batch x out_channels x time
                x = layer(x.transpose(1, 0).unsqueeze(0)).squeeze().transpose(1, 0)
            else:
                x = layer(x)

        return x


class TemporalConv(BaseModel):

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
        torch.Tensor
            mean prediction of model

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
        torch.Tensor
            mean prediction of model

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
        torch.Tensor
            mean prediction of model

        """
        pass

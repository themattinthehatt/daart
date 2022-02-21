"""Temporal Convolution model implemented in PyTorch."""

import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from daart.models.base import BaseModel, get_activation_func_from_str

# to ignore imports for sphix-autoapidoc
__all__ = ['DilatedTCN']


class DilatedTCN(BaseModel):
    """Temporal Convolutional Model with dilated convolutions and no temporal downsampling.

    Code adapted from: https://www.kaggle.com/ceshine/pytorch-temporal-convolutional-networks
    """

    def __init__(self, hparams, type='encoder', in_size=None, hid_size=None, out_size=None):
        super().__init__()
        self.hparams = hparams
        self.model = nn.Sequential()
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
        """Construct encoder model using hparams."""

        global_layer_num = 0

        for i_layer in range(self.hparams['n_hid_layers']):

            dilation = 2 ** i_layer
            in_size_ = in_size if i_layer == 0 else hid_size
            hid_size_ = hid_size
            if i_layer == (self.hparams['n_hid_layers'] - 1):
                # final layer
                out_size_ = out_size
            else:
                # intermediate layer
                out_size_ = hid_size

            # conv -> activation -> dropout (+ residual)
            tcn_block = DilationBlock(
                input_size=in_size_, int_size=hid_size_, output_size=out_size_,
                kernel_size=self.hparams['n_lags'], stride=1, dilation=dilation,
                activation=self.hparams['activation'], dropout=self.hparams.get('dropout', 0.2))
            name = 'tcn_block_%02i' % global_layer_num
            self.model.add_module(name, tcn_block)

            # update layer info
            global_layer_num += 1

        return global_layer_num

    def build_decoder(self, in_size, hid_size, out_size):
        """Construct the decoder using hparams."""

        global_layer_num = 0

        out_size_ = in_size  # set "output size" of the layer that feeds into this module
        for i_layer in range(self.hparams['n_hid_layers']):

            dilation = 2 ** (self.hparams['n_hid_layers'] - i_layer - 1)  # down by powers of 2
            in_size_ = out_size_  # input is output size of previous block
            hid_size_ = hid_size
            if i_layer == (self.hparams['n_hid_layers'] - 1):
                # final layer
                # out_size = self.hparams['input_size']
                # final_activation = 'linear'
                # predictor_block = True
                out_size_ = out_size
                final_activation = self.hparams['activation']
                predictor_block = False
            else:
                # intermediate layer
                out_size_ = hid_size
                final_activation = self.hparams['activation']
                predictor_block = False

            # conv -> activation -> dropout (+ residual)
            tcn_block = DilationBlock(
                input_size=in_size_, int_size=hid_size_, output_size=out_size_,
                kernel_size=self.hparams['n_lags'], stride=1, dilation=dilation,
                activation=self.hparams['activation'], final_activation=final_activation,
                dropout=self.hparams.get('dropout', 0.2), predictor_block=predictor_block)
            name = 'tcn_block_%02i' % global_layer_num
            self.model.add_module(name, tcn_block)

            # update layer info
            global_layer_num += 1

        # add final fully-connected layer
        dense = nn.Conv1d(
            in_channels=out_size,
            out_channels=out_size,
            kernel_size=1)  # kernel_size=1 <=> dense, fully connected layer
        self.model.add_module('final_dense_%02i' % global_layer_num, dense)

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
        # x = B x T x N (e.g. B = 2, T = 500, N = 16)
        # x.transpose(1, 2) -> x = B x N x T
        # x = layer(x) -> x = B x M x T
        # x.transpose(1, 2) -> x = B x T x M
        return self.model(x.transpose(1, 2)).transpose(1, 2)


class DilationBlock(nn.Module):
    """Residual Temporal Block module for use with DilatedTCN class."""

    def __init__(
            self, input_size, int_size, output_size, kernel_size, stride=1, dilation=2,
            activation='relu', dropout=0.2, final_activation=None, predictor_block=False):

        super(DilationBlock, self).__init__()

        self.conv0 = nn.utils.weight_norm(nn.Conv1d(
            in_channels=input_size,
            out_channels=int_size,
            stride=stride,
            dilation=dilation,
            kernel_size=kernel_size * 2 + 1,  # window around t
            padding=kernel_size * dilation))  # same output

        self.conv1 = nn.utils.weight_norm(nn.Conv1d(
            in_channels=int_size,
            out_channels=output_size,
            stride=stride,
            dilation=dilation,
            kernel_size=kernel_size * 2 + 1,  # window around t
            padding=kernel_size * dilation))  # same output

        # intermediate activations
        self.activation = get_activation_func_from_str(activation)

        # final activation
        if final_activation is None:
            final_activation = activation
        self.final_activation = get_activation_func_from_str(final_activation)

        # no Dropout1D in pytorch API, but Dropout2D does what what we want:
        # takes an input of shape (N, C, L) and drops out entire features in the `C` dimension
        self.dropout = nn.Dropout2d(dropout)

        # build net
        self.block = nn.Sequential()
        # conv -> relu -> dropout block # 0
        self.block.add_module('conv1d_layer_0', self.conv0)
        self.block.add_module('%s_0' % activation, self.activation)
        self.block.add_module('dropout_0', self.dropout)
        # conv -> relu -> dropout block # 1
        self.block.add_module('conv1d_layer_1', self.conv1)
        if not predictor_block:
            self.block.add_module('%s_1' % activation, self.activation)
            self.block.add_module('dropout_1', self.dropout)

        # for downsampling residual connection
        if input_size != output_size:
            self.downsample = nn.Conv1d(input_size, output_size, kernel_size=1)
        else:
            self.downsample = None

        self.init_weights()

    def __str__(self):
        format_str = 'DilationBlock\n'
        for i, module in enumerate(self.block):
            format_str += '        {}: {}\n'.format(i, module)
        format_str += '        {}: residual connection\n'.format(i + 1)
        format_str += '        {}: {}\n'.format(i + 2, self.final_activation)
        return format_str

    def init_weights(self):
        self.conv0.weight.data.normal_(0, 0.01)
        self.conv1.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.block(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_activation(out + res)

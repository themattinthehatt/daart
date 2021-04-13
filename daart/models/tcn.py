"""Temporal Convolution model implemented in PyTorch."""

import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from daart.models.base import BaseModel

# to ignore imports for sphix-autoapidoc
__all__ = ['TCN']


class TCN(BaseModel):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.encoder = None
        self.classifier = None
        self.predictor = None
        self.build_model()

    def __str__(self):
        """Pretty print model architecture."""
        format_str = '\nTCN architecture\n'
        format_str += '------------------------\n'
        format_str += 'Encoder:\n'
        for i, module in enumerate(self.encoder):
            format_str += str('    {}: {}\n'.format(i, module))
        format_str += 'Classifier:\n'
        for i, module in enumerate(self.classifier):
            format_str += str('    {}: {}\n'.format(i, module))
        if self.predictor is not None:
            format_str += 'Predictor:\n'
            for i, module in enumerate(self.predictor):
                format_str += str('    {}: {}\n'.format(i, module))
        return format_str

    def build_model(self):
        """Construct the model using hparams."""

        global_layer_num = 0

        # ------------------------------
        # encoder TCN
        # ------------------------------

        self.encoder = nn.ModuleList()

        in_size = self.hparams['input_size']
        out_size = self.hparams['n_hid_units']

        t_sizes = [self.hparams['batch_size']]
        for i_layer in range(self.hparams['n_hid_layers']):

            # conv -> activation -> maxpool
            self._build_tcn_encoder_block(
                module_list=self.encoder, input_size=in_size, output_size=out_size,
                global_layer_num=global_layer_num, t_sizes=t_sizes)

            # update input/output dims
            in_size = out_size
            out_size = self.hparams['n_hid_units']

            # update layer info
            global_layer_num += 1

        # ----------------------------
        # Classifier
        # ----------------------------

        self.classifier = nn.ModuleList()

        in_size = self.hparams['n_hid_units']
        out_size = self.hparams['n_hid_units']

        for i_layer in range(self.hparams['n_hid_layers']):

            # upsample -> conv -> activation
            self._build_tcn_decoder_block(
                module_list=self.classifier, input_size=in_size, output_size=out_size,
                global_layer_num=global_layer_num, t_size=t_sizes[-i_layer-2])

            # update input/output dims
            in_size = out_size
            out_size = self.hparams['n_hid_units']

            # update layer info
            global_layer_num += 1

        out_size = self.hparams['output_size']

        # add layer (cross entropy loss handles activation)
        layer = nn.Linear(in_features=in_size, out_features=out_size)
        name = str('dense(classification)_layer_%02i' % global_layer_num)
        self.classifier.add_module(name, layer)

        global_layer_num += 1

        # ----------------------------
        # Predictor
        # ----------------------------

        if self.hparams.get('lambda_pred', 0) > 0:

            self.predictor = nn.ModuleList()

            in_size = self.hparams['n_hid_units']
            out_size = self.hparams['n_hid_units']

            for i_layer in range(self.hparams['n_hid_layers']):

                # upsample -> conv -> activation
                self._build_tcn_decoder_block(
                    module_list=self.predictor, input_size=in_size, output_size=out_size,
                    global_layer_num=global_layer_num, t_size=t_sizes[-i_layer-2])

                # update input/output dims
                in_size = out_size
                out_size = self.hparams['n_hid_units']

                # update layer info
                global_layer_num += 1

            out_size = self.hparams['input_size']

            # add layer (linear activation)
            layer = nn.Linear(in_features=in_size, out_features=out_size)
            name = str('dense(prediction)_layer_%02i' % global_layer_num)
            self.predictor.add_module(name, layer)

            global_layer_num += 1

    def _build_tcn_encoder_block(
            self, module_list, input_size, output_size, global_layer_num, t_sizes=None):
        """Encoder TCN block as described in Lea et al CVPR 2017."""

        # conv layer
        layer = nn.Conv1d(
            in_channels=input_size,
            out_channels=output_size,
            kernel_size=self.hparams['n_lags'] * 2 + 1,  # window around t
            padding=self.hparams['n_lags'])  # same output
        name = str('conv1d_layer_%02i' % global_layer_num)
        module_list.add_module(name, layer)

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
            module_list.add_module(name, activation)

        # downsample: add max pooling across time
        kernel_size = 2
        stride = kernel_size
        padding = 0
        dilation = 1
        layer = nn.MaxPool1d(
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        name = str('maxpool1d_layer_%02i' % global_layer_num)
        module_list.add_module(name, layer)

        # save temporal dim
        if t_sizes is not None:
            t_size = (t_sizes[-1] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
            t_sizes.append(int(np.floor(t_size)))

    def _build_tcn_decoder_block(
            self, module_list, input_size, output_size, global_layer_num, t_size=None):
        """Decoder TCN block as described in Lea et al CVPR 2017."""

        # upsample: undo max pooling across time
        if t_size is None:
            layer = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            layer = nn.Upsample(size=t_size, mode='nearest')
        name = str('maxpool1d_layer_%02i' % global_layer_num)
        module_list.add_module(name, layer)

        # conv layer
        layer = nn.Conv1d(
            in_channels=input_size,
            out_channels=output_size,
            kernel_size=self.hparams['n_lags'] * 2 + 1,  # window around t
            padding=self.hparams['n_lags'])  # same output
        name = str('conv1d_layer_%02i' % global_layer_num)
        module_list.add_module(name, layer)

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
            module_list.add_module(name, activation)

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

        # push data through encoder to get latent embedding
        for name, layer in self.encoder.named_children():
            if name == 'conv1d_layer_00':
                # input is batch x in_channels x time
                # output is batch x out_channels x time

                # x = T x N (T = 500, N = 16)
                # x.transpose(1, 0) -> x = N x T
                # x.unsqueeze(0) -> x = 1 x N x T
                # x = layer(x) -> x = 1 x M x T
                # x.squeeze() -> x = M x T
                # x.transpose(1, 0) -> x = T x M

                x = layer(x.transpose(1, 0).unsqueeze(0))
            else:
                x = layer(x)

        # push embedding through classifier to get labels
        z = x
        for name, layer in self.classifier.named_children():
            if name[:5] == 'dense':
                # reshape for linear layer
                z = layer(z.squeeze().transpose(1, 0))
            else:
                z = layer(z)

        # push embedding through predictor network to get data at subsequent time points
        if self.hparams.get('lambda_pred', 0) > 0:
            y = x
            for name, layer in self.predictor.named_children():
                if name[:5] == 'dense':
                    # reshape for linear layer
                    y = layer(y.squeeze().transpose(1, 0))
                else:
                    y = layer(y)
        else:
            y = None

        return {'labels': z, 'prediction': y, 'embedding': x.squeeze().transpose(1, 0)}

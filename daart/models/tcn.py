"""Temporal Convolution model implemented in PyTorch."""

import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from daart.models.base import BaseModel

# to ignore imports for sphix-autoapidoc
__all__ = ['TCN', 'DilatedTCN']


class TCN(BaseModel):
    """Basic Temporal Convolutional Model with temporal downsampling."""

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.encoder = None
        self.classifier = None
        self.predictor = None
        self.build_model()

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


class DilatedTCN(BaseModel):
    """Temporal Convolutional Model with dilated convolutions and no temporal downsampling.

    Code adapted from: https://www.kaggle.com/ceshine/pytorch-temporal-convolutional-networks
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.hparams['pred_final_linear_layer'] = self.hparams.get('pred_final_linear_layer', True)

        self.encoder = None
        self.classifier = None
        self.classifier_weak = None
        self.predictor = None
        self.task_predictor = None
        self.build_model()

    def build_model(self):
        """Construct the model using hparams."""

        global_layer_num = 0

        # encoder TCN
        global_layer_num = self._build_encoder(global_layer_num=global_layer_num)

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

        # predictor TCN
        if self.hparams.get('lambda_pred', 0) > 0:
            self._build_predictor(global_layer_num=global_layer_num)

    def _build_encoder(self, global_layer_num):

        self.encoder = nn.Sequential()

        for i_layer in range(self.hparams['n_hid_layers']):

            dilation = 2 ** i_layer
            in_size = self.hparams['input_size'] if i_layer == 0 else self.hparams['n_hid_units']
            out_size = self.hparams['n_hid_units']

            # conv -> activation -> dropout (+ residual)
            tcn_block = DilationBlock(
                input_size=in_size, int_size=out_size, output_size=out_size,
                kernel_size=self.hparams['n_lags'], stride=1, dilation=dilation,
                activation=self.hparams['activation'], dropout=self.hparams.get('dropout', 0.2))
            name = 'tcn_block_%02i' % global_layer_num
            self.encoder.add_module(name, tcn_block)

            # update layer info
            global_layer_num += 1

        return global_layer_num

    def _build_predictor(self, global_layer_num):

        self.predictor = nn.Sequential()

        for i_layer in range(self.hparams['n_hid_layers']):

            dilation = 2 ** (self.hparams['n_hid_layers'] - i_layer - 1)  # down by powers of 2
            in_size = self.hparams['n_hid_units']
            if i_layer == (self.hparams['n_hid_layers'] - 1):
                # final layer
                # out_size = self.hparams['input_size']
                # final_activation = 'linear'
                # predictor_block = True
                out_size = self.hparams['n_hid_units']
                final_activation = self.hparams['activation']
                predictor_block = False
            else:
                # intermediate layer
                out_size = self.hparams['n_hid_units']
                final_activation = self.hparams['activation']
                predictor_block = False

            # conv -> activation -> dropout (+ residual)
            tcn_block = DilationBlock(
                input_size=in_size, int_size=in_size, output_size=out_size,
                kernel_size=self.hparams['n_lags'], stride=1, dilation=dilation,
                activation=self.hparams['activation'], final_activation=final_activation,
                dropout=self.hparams.get('dropout', 0.2), predictor_block=predictor_block)
            name = 'tcn_block_%02i' % global_layer_num
            self.predictor.add_module(name, tcn_block)

            # update layer info
            global_layer_num += 1

        # add final fully-connected layer
        if self.hparams['pred_final_linear_layer']:
            dense = nn.Conv1d(
                in_channels=out_size,
                out_channels=self.hparams['input_size'],
                kernel_size=1)  # kernel_size=1 <=> dense, fully connected layer
            self.predictor.add_module('final_dense_%02i' % global_layer_num, dense)

        return global_layer_num

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
        # x = B x T x N (e.g. B = 2, T = 500, N = 16)
        # x.transpose(1, 2) -> x = B x N x T
        # x = layer(x) -> x = B x M x T
        # x.transpose(1, 2) -> x = B x T x M
        x = self.encoder(x.transpose(1, 2))
        xt = x.transpose(1, 2)

        # push embedding through classifiers to get labels
        if self.hparams.get('lambda_strong', 0) > 0:
            z = self.classifier(xt)
        else:
            z = None

        if self.hparams.get('lambda_weak', 0) > 0:
            z_weak = self.classifier_weak(xt)
        else:
            z_weak = None

        # push embedding through linear layer to get task predictions
        if self.hparams.get('lambda_task', 0) > 0:
            w = self.task_predictor(xt)
        else:
            w = None

        # push embedding through predictor network to get data at subsequent time points
        if self.hparams.get('lambda_pred', 0) > 0:
            y = self.predictor(x).transpose(1, 2)
        else:
            y = None

        return {
            'labels': z,  # (n_sequences, sequence_length, n_classes)
            'labels_weak': z_weak,  # (n_sequences, sequence_length, n_classes)
            'prediction': y,  # (n_sequences, sequence_length, n_markers)
            'task_prediction': w,  # (n_sequences, sequence_length, n_tasks)
            'embedding': xt  # (n_sequences, sequence_length, embedding_dim)
        }


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
        if activation == 'linear':
            self.activation = nn.Identity()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.05)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('"%s" is an invalid activation function' % activation)

        # final activation
        if final_activation is None:
            final_activation = activation
        if final_activation == 'linear':
            self.final_activation = nn.Identity()
        elif final_activation == 'relu':
            self.final_activation = nn.ReLU()
        elif final_activation == 'lrelu':
            self.final_activation = nn.LeakyReLU(0.05)
        elif final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif final_activation == 'tanh':
            self.final_activation = nn.Tanh()
        else:
            raise ValueError('"%s" is an invalid activation function' % final_activation)

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

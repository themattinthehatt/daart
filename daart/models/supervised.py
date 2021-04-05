"""Supervised models implemented in PyTorch."""

import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from daart.models.base import BaseModule, BaseModel
from daart.models.lstm import LSTM

# to ignore imports for sphix-autoapidoc
__all__ = ['Segmenter', 'TemporalMLP', 'TemporalConv', 'LSTM', 'TGM']


class Segmenter(BaseModel):
    """General wrapper class for behavioral segmentation models."""

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
            - lambda_weak (float): hyperparam on weak label classification
            - lambda_strong (float): hyperparam on srong label classification
            - lambda_pred (float): hyperparam on next step prediction

        """
        super().__init__()
        self.hparams = hparams
        self.model = None
        self.build_model()

        # label loss based on cross entropy; don't compute gradient when target = 0
        self.class_loss = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
        self.pred_loss = nn.MSELoss(reduction='mean')

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
            self.model = LSTM(self.hparams)
            # self.model = LSTM(self.hparams)
        elif self.hparams['model_type'] == 'tgm':
            raise NotImplementedError
            # self.model = TGM(self.hparams)
        else:
            raise ValueError('"%s" is not a valid model type' % self.hparams['model_type'])

    def forward(self, x):
        """Process input data."""
        return self.model(x)

    def predict_labels(self, data_generator):
        """

        Parameters
        ----------
        data_generator : DataGenerator object
            data generator to serve data batches

        Returns
        -------
        dict
            - 'predictions' (list of lists): first list is over datasets; second list is over
              batches in the dataset; each element is a numpy array of the label probability
              distribution
            - 'weak_labels' (list of lists): corresponding weak labels
            - 'labels' (list of lists): corresponding labels

        """
        self.eval()

        softmax = nn.Softmax(dim=1)

        # initialize container for labels
        labels = [[] for _ in range(data_generator.n_datasets)]
        for sess, dataset in enumerate(data_generator.datasets):
            labels[sess] = [np.array([]) for _ in range(dataset.n_trials)]

        # partially fill container (gap trials will be included as nans)
        dtypes = ['train', 'val', 'test']
        for dtype in dtypes:
            data_generator.reset_iterators(dtype)
            for i in range(data_generator.n_tot_batches[dtype]):
                data, sess = data_generator.next_batch(dtype)
                predictors = data['markers'][0]
                # targets = data['labels'][0]
                outputs_dict = self.model(predictors)
                # push through log-softmax, since this is included in the loss and not model
                labels[sess][data['batch_idx'].item()] = \
                    softmax(outputs_dict['labels']).cpu().detach().numpy()

        return {'labels': labels}

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

        # define hyperparams
        lambda_weak = self.hparams.get('lambda_weak', 0)
        lambda_strong = self.hparams.get('lambda_strong', 0)
        lambda_pred = self.hparams.get('lambda_pred', 0)

        # push data through model
        markers = data['markers'][0]
        outputs_dict = self.model(markers)

        # get masks that define where strong labels are
        if lambda_strong > 0:
            labels_strong = data['labels_strong'][0]
        else:
            labels_strong = None

        # initialize loss to zero
        loss = 0

        # ------------------------------------
        # compute loss on weak labels
        # ------------------------------------
        if lambda_weak > 0:
            labels_weak = data['labels_weak'][0]
            # only compute loss where strong labels do not exist [indicated by a zero]
            if labels_strong is not None:
                loss_weak = self.class_loss(
                    outputs_dict['labels'][labels_strong == 0], labels_weak[labels_strong == 0])
            else:
                loss_weak = self.class_loss(outputs_dict['labels'], labels_weak)
            loss += lambda_weak * loss_weak
            loss_weak_val = loss_weak.item()
            # compute fraction correct on weak labels
            outputs_val = outputs_dict['labels'].cpu().detach().numpy()
            fc = accuracy_score(labels_weak.cpu().detach().numpy(), np.argmax(outputs_val, axis=1))
        else:
            fc = np.nan
            loss_weak_val = 0

        # ------------------------------------
        # compute loss on strong labels
        # ------------------------------------
        if lambda_strong > 0:
            loss_strong = self.class_loss(outputs_dict['labels'], labels_strong)
            loss += lambda_strong * loss_strong
            loss_strong_val = loss_strong.item()
        else:
            loss_strong_val = 0

        # ------------------------------------
        # compute loss on one-step predictions
        # ------------------------------------
        if lambda_pred > 0:
            loss_pred = self.pred_loss(markers[1:], outputs_dict['prediction'][:-1])
            loss += lambda_pred * loss_pred
            loss_pred_val = loss_pred.item()
        else:
            loss_pred_val = 0

        if accumulate_grad:
            loss.backward()

        # collect loss vals
        loss_dict = {
            'loss': loss.item(),
            'loss_weak': loss_weak_val,
            'loss_strong': loss_strong_val,
            'loss_pred': loss_pred_val,
            'fc': fc,
        }

        return loss_dict


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

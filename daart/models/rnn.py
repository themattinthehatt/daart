import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from daart.models.base import BaseModel

# to ignore imports for sphix-autoapidoc
__all__ = ['RNN']


class RNN(BaseModel):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model_type = hparams.get('model_type', 'lstm').lower()
        self.encoder = None
        self.classifier = None
        self.classifier_weak = None
        self.predictor = None
        self.build_model()

    def __str__(self):
        """Pretty print model architecture."""
        format_str = '\n%s architecture\n' % self.model_type.upper()
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

        self.encoder = nn.ModuleList()

        global_layer_num = 0

        # ------------------------------
        # trainable hidden 0 for RNN
        # ------------------------------

        in_size = self.hparams['input_size']
        if self.model_type == 'lstm':
            layer = nn.LSTM(
                input_size=in_size,
                hidden_size=self.hparams['n_hid_units'],
                num_layers=self.hparams['n_hid_layers'],
                batch_first=True,
                bidirectional=self.hparams['bidirectional'])
            name = str('LSTM_layer_%02i' % global_layer_num)
        elif self.hparams['model_type'] == 'gru':
            layer = nn.GRU(
                input_size=in_size,
                hidden_size=self.hparams['n_hid_units'],
                num_layers=self.hparams['n_hid_layers'],
                batch_first=True,
                bidirectional=self.hparams['bidirectional'])
            name = str('GRU_layer_%02i' % global_layer_num)
        else:
            raise NotImplementedError(
                'Invalid model type "%s"; must choose "lstm" or "gru"' % self.model_type)
        self.encoder.add_module(name, layer)

        # update layer info
        global_layer_num += 1
        final_encoder_size = (int(self.hparams['bidirectional']) + 1) * self.hparams['n_hid_units']

        # ----------------------------
        # Classifiers
        # ----------------------------

        # linear classifier (hand labels)
        self.classifier = self._build_classifier(
            in_size=final_encoder_size, global_layer_num=global_layer_num)

        # linear classifier (heuristic labels)
        self.classifier_weak = self._build_classifier(
            in_size=final_encoder_size, global_layer_num=global_layer_num)

        global_layer_num += 1

        # ----------------------------
        # Predictor
        # ----------------------------

        if self.hparams.get('lambda_pred', 0) > 0:

            self.predictor = nn.ModuleList()

            # rnn decoder
            if self.model_type == 'lstm':
                layer = nn.LSTM(
                    input_size=final_encoder_size,
                    hidden_size=self.hparams['n_hid_units'],
                    num_layers=self.hparams['n_hid_layers'],
                    batch_first=True,
                    bidirectional=self.hparams['bidirectional'])
                name = str('LSTM(prediction)_layer_%02i' % global_layer_num)
            elif self.hparams['model_type'] == 'gru':
                layer = nn.GRU(
                    input_size=final_encoder_size,
                    hidden_size=self.hparams['n_hid_units'],
                    num_layers=self.hparams['n_hid_layers'],
                    batch_first=True,
                    bidirectional=self.hparams['bidirectional'])
                name = str('GRU(prediction)_layer_%02i' % global_layer_num)
            else:
                raise NotImplementedError(
                    'Invalid model type "%s"; must choose "lstm" or "gru"' % self.model_type)

            self.predictor.add_module(name, layer)

            global_layer_num += 1

            # final linear layer
            in_size = (int(self.hparams['bidirectional']) + 1) * self.hparams['n_hid_units']
            layer = nn.Linear(in_features=in_size, out_features=self.hparams['input_size'])

            name = str('dense(prediction)_layer_%02i' % global_layer_num)
            self.predictor.add_module(name, layer)

            global_layer_num += 1

    def _build_classifier(self, in_size, global_layer_num):

        classifier = nn.Sequential()

        out_size = self.hparams['output_size']

        # add layer (cross entropy loss handles activation)
        layer = nn.Linear(in_features=in_size, out_features=out_size)
        name = str('dense(classification)_layer_%02i' % global_layer_num)
        classifier.add_module(name, layer)

        return classifier

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

        # push data through encoder to get latent embedding
        # x is of shape (T, input_size)
        # unsqueeze adds new dim in front; now shape (1, T, input_size)
        x = x.unsqueeze(0)
        for name, layer in self.encoder.named_children():
            x, _ = layer(x)

        # push embedding through classifier to get labels
        xt = x.squeeze()
        z = self.classifier(xt)
        if self.hparams.get('lambda_weak', 0) > 0:
            z_weak = self.classifier_weak(xt)
        else:
            z_weak = None

        # push embedding through predictor network to get data at subsequent time points
        if self.hparams.get('lambda_pred', 0) > 0:
            y = x
            for name, layer in self.predictor.named_children():
                if name[:4] == 'LSTM' or name[:3] == 'GRU':
                    y, _ = layer(y)
                else:
                    y = layer(y.squeeze())
        else:
            y = None

        return {'labels': z, 'labels_weak': z_weak, 'prediction': y, 'embedding': xt}

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
        self.task_predictor = None
        self.build_model()

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
        if self.hparams.get('lambda_strong') > 0:
            self.classifier = self._build_linear(
                global_layer_num=global_layer_num, name='classification',
                in_size=final_encoder_size, out_size=self.hparams['output_size'])

        # linear classifier (heuristic labels)
        if self.hparams.get('lambda_weak') > 0:
            self.classifier_weak = self._build_linear(
                global_layer_num=global_layer_num, name='classification',
                in_size=final_encoder_size, out_size=self.hparams['output_size'])

        global_layer_num += 1

        # -------------------------------------------------------------
        # task regression: single linear layer
        # -------------------------------------------------------------
        if self.hparams.get('lambda_task') > 0:
            self.task_predictor = self._build_linear(
                global_layer_num=global_layer_num, name='regression',
                in_size=final_encoder_size, out_size=self.hparams['task_size'])

        # update layer info
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
        for name, layer in self.encoder.named_children():
            x, _ = layer(x)

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
                if name[:4] == 'LSTM' or name[:3] == 'GRU':
                    y, _ = layer(y)
                else:
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

import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from daart.models.base import BaseModel

# to ignore imports for sphix-autoapidoc
__all__ = ['RNN']


class RNN(BaseModel):

    def __init__(self, hparams, type='encoder'):
        super().__init__()
        self.hparams = hparams
        self.model_type = hparams.get('model_type', 'lstm').lower()
        self.model = nn.ModuleList()
        if type == 'encoder':
            self.build_encoder()
        else:
            self.build_decoder()

    def _build_rnn(self, in_size, out_size, global_layer_num=0):

        # RNN layers
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
        self.model.add_module(name, layer)

        # update layer info
        global_layer_num += 1

        # final linear layer
        final_encoder_size = (int(self.hparams['bidirectional']) + 1) * self.hparams['n_hid_units']
        layer = nn.Linear(in_features=final_encoder_size, out_features=out_size)
        name = str('dense(prediction)_layer_%02i' % global_layer_num)
        self.model.add_module(name, layer)

        # update layer info
        global_layer_num += 1

        return global_layer_num

    def build_encoder(self):
        """Construct the encoder using hparams."""
        global_layer_num = self._build_rnn(
            in_size=self.hparams['input_size'], out_size=self.hparams['n_hid_units'],
            global_layer_num=0,
        )
        return global_layer_num

    def build_decoder(self):
        """Construct the decoder using hparams."""
        global_layer_num = self._build_rnn(
            in_size=self.hparams['n_hid_units'], out_size=self.hparams['input_size'],
            global_layer_num=0,
        )
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
        for name, layer in self.model.named_children():
            if name[:4] == 'LSTM' or name[:3] == 'GRU':
                x, _ = layer(x)
            else:
                x = layer(x)

        return x

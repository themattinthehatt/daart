import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from daart.models.base import BaseModule, BaseModel



class LSTM_classifier_only(BaseModel):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.decoder = None
        self.build_model()

    def __str__(self):
        """Pretty print model architecture."""
        format_str = '\LSTM model, classifier only - architecture\n'
        format_str += '------------------------\n'
        for i, module in enumerate(self.decoder):
            format_str += str('    {}: {}\n'.format(i, module))
        return format_str

    def build_model(self):
        """Construct the model using hparams."""

        self.decoder = nn.ModuleList()

        global_layer_num = 0

        in_size = self.hparams['input_size']

        # -------------------------------------------------------------
        # first layer is LSTM for incorporating past/future(biderectional) activity
        # -------------------------------------------------------------
        if self.hparams['n_linear_layers'] == 0:
            out_size = self.hparams['output_size']
        else:
            out_size = self.hparams['n_hid_size']
        
         # Do we want to train hidden 0, default false
        if self.hparams['train_h0'] :
            self.LSTM_hidden = (torch.zeros(1,self.hparams['batch_size'],self.hparams['n_hid_size']),
                                torch.zeros(1,self.hparams['batch_size'],self.hparams['n_hid_size']))
        else
            self.LSTM_hidden = None
            
        #   1, seq, feature    
        layer = nn.LSTM(
            input_size = in_size,
            hidden_size = out_size,
            num_layers = self.hparams['n_hid_layers'],
            bidirectional = self.hparams['bidirectional'])  
        
        
        name = str('LSTM_layer_%02i' % global_layer_num)
        self.decoder.add_module(name, layer)
    
        # -------------------------------------------------------------
        # Conv layer for model with single sequence input
        # -------------------------------------------------------------
            # Have not been implemented YET
        
        
        
        # -------------------------------------------------------------
        # Linear layer for model with windowed input
        # -------------------------------------------------------------
        global_layer_num += 1
        in_size = out_size * (self.hparams['bidirectional'] + 1) * self.hparams['n_LSTM_layers']
        
        for i_layer in range(self.hparams['n_linear_layers']):
            if i_layer == self.hparams['n_linear_layers'] - 1:
                out_size = self.hparams['output_size']
            else:
                out_size = self.hparams['linear_hid_units']

            # add layer
            layer = nn.Linear(in_features=in_size, out_features=out_size)
            name = str('dense_layer_%02i' % global_layer_num)
            self.decoder.add_module(name, layer)

            # add activation
            if i_layer == self.hparams['n_linear_layers'] - 1:
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
            batch x seq x in_features
        Returns
        -------
        torch.Tensor
            mean prediction of model

        """
        for name, layer in self.decoder.named_children():
            
            if name == 'LSTM_layer_00':
                # input is batch x in_channels x time
                # output is batch x out_channels x time
                
                if not self.LSTM_hidden:
                    output, _ = layer(x.unsqueeze(0))
                else:
                    output, _ = layer(x.unsqueeze(0), self.LSTM_hidden)
                x = output.squeeze()
            else:
                x = layer(x)
        return x
    
class LSTM(BaseModel):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.encoder = None
        self.classifier = None
        self.predictor = None
        self.build_model()

    def __str__(self):
        """Pretty print the model architecture."""
        
        format_str = '\nLSTM architecture\n'
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
        
        self.encoder = nn.ModuleList()

        global_layer_num = 0

        in_size = self.hparams['input_size']

        

        # ------------------------------
        # trainable hidden 0 for LSTM 
        # ------------------------------
        
#         if self.hparams['train_h0'] :
#             self.LSTM_hidden = (torch.zeros(1,self.hparams['batch_size'],self.hparams['n_hid_size']),
#                                 torch.zeros(1,self.hparams['batch_size'],self.hparams['n_hid_size']))
#         else
#             self.LSTM_hidden = None
        
        
        #   1, seq, feature    
        layer = nn.LSTM(
            input_size = in_size,
            hidden_size = self.hparams['n_hid_size'],
            num_layers = self.hparams['n_hid_layers'],
            bidirectional = self.hparams['bidirectional'])  
        
        name = str('LSTM_layer_%02i' % global_layer_num)
        
        self.encoder.add_module(name, layer)

        # update layer info
        global_layer_num += 1
        in_size =  (int(self.hparams['bidirectional']) + 1) * self.hparams['n_hid_size']
        
        
        # Second LSTM layer
        layer = nn.LSTM(
            input_size = in_size,
            hidden_size = self.hparams['n_hid_size'],
            num_layers = self.hparams['n_hid_layers'],
            bidirectional = self.hparams['bidirectional'])  
        
        name = str('LSTM_layer_%02i' % global_layer_num)
        self.encoder.add_module(name, layer)
        
        global_layer_num += 1    
        
        
        #----------------------------
        #  Classifier
        #----------------------------
        
        self.classifier = nn.ModuleList()
        
        layer = nn.LSTM(
            input_size = in_size,
            hidden_size = self.hparams['n_hid_size'],
            num_layers = self.hparams['n_hid_layers'],
            bidirectional = self.hparams['bidirectional'])  
        
        name = str('LSTM(classification)_layer_%02i' % global_layer_num)
        self.classifier.add_module(name, layer)
        
        global_layer_num += 1    
        
        # Last layer
        self.hidden_to_output = nn.Linear(self.in_size, self.hparams['output_size'])
        
        name = str('dense(classification)_layer_%02i' % global_layer_num)
        self.classifier.add_module(name, layer)
        
        global_layer_num += 1
        
        
        #----------------------------
        #  Predictor
        #----------------------------
        
        
        if self.hparams.get('lambda_pred', 0) > 0:
            self.predictor = nn.ModuleList()
            
            layer = nn.LSTM(
            input_size = in_size,
            hidden_size = self.hparams['n_hid_size'],
            num_layers = self.hparams['n_hid_layers'],
            bidirectional = self.hparams['bidirectional'])  
        
            name = str('LSTM(prediction)_layer_%02i' % global_layer_num)
            self.predictor.add_module(name, layer)

            global_layer_num += 1    

            # Last layer
            self.hidden_to_output = nn.Linear(self.in_size, self.hparams['input_size'])

            name = str('dense(prediction)_layer_%02i' % global_layer_num)
            self.classifier.add_module(name, layer)

            global_layer_num += 1
        
        
        
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
        #x = x.unsqueeze(0)
        for name, layer in self.encoder.named_children():
            x, _ = layer(x)
            
        if self.hparams.get('lambda_pred', 0) > 0:
            y = x
            for name, layer in self.predictor.named_children():
                if name[:4]=="LSTM":
                    y, _ = layer(y)
                else:
                    y = layer(y)
        else:
            y = None
            
        for name, layer in self.classifier.named_children():
            if name[:4]=="LSTM":
                x, _ = layer(x)
            else:
                x = layer(x)
    
        return {'labels': x, 'prediction': y}
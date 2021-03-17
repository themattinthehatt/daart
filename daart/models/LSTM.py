import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from daart.models.base import BaseModule, BaseModel


    
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
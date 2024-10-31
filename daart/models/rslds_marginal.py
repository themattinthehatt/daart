"""rSLDS models/modules in PyTorch."""

import math
import numpy as np
import os
import pickle
from scipy.special import softmax as scipy_softmax
from scipy.stats import entropy
import torch
import copy
from sklearn.metrics import accuracy_score, r2_score
from torch import nn, save

from daart import losses
from daart.losses import FocalLoss
from daart.models.base import BaseModel, BaseInference, BaseGenerative, reparameterize_gaussian, get_activation_func_from_str
from daart.models.rsldsm import RSLDSMInference, RSLDSMGenerative
from daart.transforms import MakeOneHot
    
from torch.distributions import Categorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence



class RSLDSM(BaseModel):
    """rSLDS Model.
    
    [insert arxiv link here]
    """

    def __init__(self, hparams):
        """
        
        Parameters
        ----------
        hparams : dict
            - backbone (str): 'temporal-mlp' | 'dtcn' | 'lstm' | 'gru'
            - rng_seed_model (int): random seed to control weight initialization
            - input_size (int): number of input channels
            - output_size (int): number of observed classes
            - n_aug_classes (int): number of additional classes without labels
            - sequence_pad (int): padding needed to account for convolutions
            - n_hid_layers (int): hidden layers of network architecture
            - n_hid_units (int): hidden units per layer
            - n_latents (int): dimension of latent space
            - n_lags (int): number of lags in input data to use for temporal convolution
            - activation (str): 'linear' | 'relu' | 'lrelu' | 'sigmoid' | 'tanh'
            - lambda_strong (float): hyperparam on strong label classification 
              (alpha in original paper)
            
        """
        super().__init__()
        if len(hparams['label_names']) == 1:
            hparams['label_names'] = hparams['label_names'][0]
            
        self.hparams = hparams
        
        self.keys = ['qy_x_probs', 'qz_xy_mean', 'py_logits', 'py_probs', 'reconstruction']
        
#         self.keys = ['qy_x_probs','qz_xy_mean','qz_xy_logvar'
#                 ,'pz_mean','pz_logvar','reconstruction']

        # model dict will contain some or all of the following components:
        # - classifier: q(y|x) [weighted by hparams['lambda_strong'] on labeled data]
        # - encoder: q(z|x,y)
        # - decoder: p(x|z)
        # - latent_generator: p(z|y)

        self.model = nn.ModuleDict()
        self.inference = RSLDSMInference(hparams, self.model)
        self.generative = RSLDSMGenerative(hparams, self.model)
        
        self.build_model()

        # label loss based on cross entropy; don't compute gradient when target = 0
        ignore_index = hparams.get('ignore_class', 0)
        weight = hparams.get('alpha', None)
        if weight is not None:
            weight = torch.tensor(weight)
        focal_loss = self.hparams.get('focal_loss', False)
        if focal_loss:
            self.class_loss = FocalLoss(gamma=self.hparams['gamma'], alpha=weight, ignore_index=ignore_index)
        else:
            self.class_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='mean')
        
        # MSE loss for reconstruction
        self.recon_loss = nn.MSELoss(reduction='mean')
        
    def __str__(self):
        """Pretty print model architecture."""

        format_str = '\n\nInference model\n'
        format_str += self.inference.__str__()

        format_str += '------------------------\n'

        format_str += '\n\nGenerative model\n'
        format_str += self.generative.__str__()

        return format_str
    
    def predict_argmax_labels(self, x, y):
        
        '''
        return {
        STAR    'py_logits': py_logits, # (n_total_classes, n_sequences, sequence_length, n_total_classes)
        STAR    'pz_mean': pz_mean,  # (n_total_classes, n_sequences, sequence_length, n_total_classes, embedding_dim)
        STAR    'pz_logvar': pz_logvar,  # (n_sequences, sequence_length, n_total_classes, embedding_dim)
        STAR    'reconstruction': x_hat,  # (n_total_classes, n_sequences, sequence_length, n_markers)
        }
        
        return {
            'qy_x_probs': qy_x_probs,  # (n_sequences, sequence_length, n_classes)
            'qy_e_probs': qy_e_probs, # (n_sequences, sequence_length, n_classes)
            'qy_x_logits': qy_x_logits,  # (n_sequences, sequence_length, n_classes)
       STAR     'qz_xy_mean': qz_xy_mean,  # (n_classes, n_sequences, sequence_length, embedding_dim)
       STAR     'qz_xy_logvar': qz_xy_logvar,  # (n_classes, n_sequences, sequence_length, embedding_dim)
       STAR     'z_xy_sample': z_xy_sample, # (n_classes, n_sequences, sequence_length, embedding_dim)
            'idxs_labeled': idxs_labeled,  # (n_sequences, sequence_length)
        }
        '''
        
        
        output_dict = self.forward(x, y)
        
        probs = torch.argmax(output_dict['qy_x_probs'], dim=2)
        recon = output_dict['reconstruction']
        py_logits = output_dict['py_logits']
        qz_mean = output_dict['qz_xy_mean']
        
        recon_new = torch.zeros((recon.shape[1:])) # (n_sequences, sequence_length, n_markers)
        qz_mean_new = torch.zeros((qz_mean.shape[1:])) # (n_sequences, sequence_length, embedding_dim)
        py_logits_new = torch.zeros((py_logits.shape[1:])) # (n_sequences, sequence_length, n_total_classes)
    
        for s in range(recon.shape[1]):
            for i in range(recon.shape[2]):
                recon_new[s, i] = recon[probs[s, i], s, i, ...]
                qz_mean_new[s, i] = qz_mean[probs[s, i], s, i, ...]
                py_logits_new[s, i] = py_logits[probs[s, i], s, i, ...]      
            
        output_dict['py_logits'] = py_logits_new
        output_dict['reconstruction'] = recon_new
        output_dict['qz_xy_mean'] = qz_mean_new
        output_dict['py_probs'] = nn.functional.softmax(py_logits_new, dim=2)
        
        return output_dict
        
    
    def predict_labels(self, data_generator, return_scores=False, remove_pad=True, mode='eval'):
        """
        Parameters
        ----------
        data_generator : DataGenerator object
            data generator to serve data batches
        return_scores : bool
            return scores before they've been passed through softmax
        remove_pad : bool
            remove batch padding from model outputs before returning
        Returns
        -------
        dict
            - 'predictions' (list of lists): first list is over datasets; second list is over
              batches in the dataset; each element is a numpy array of the label probability
              distribution
            - 'weak_labels' (list of lists): corresponding weak labels
            - 'labels' (list of lists): corresponding labels
        """
        if mode == 'eval':
            self.eval()
        elif mode == 'train':
            self.train()
        else:
            raise NotImplementedError(
                'must choose mode="eval" or mode="train", not mode="%s"' % mode)

        pad = self.hparams.get('sequence_pad', 0)
        softmax = nn.Softmax(dim=1)

        # initialize outputs dict
        keys = self.keys
        
        results_dict = {}
        
        results_dict['markers'] = [[] for _ in range(data_generator.n_datasets)]
        for sess, dataset in enumerate(data_generator.datasets):
                results_dict['markers'][sess] = [np.array([]) for _ in range(dataset.n_sequences)]
                
        results_dict['labels_strong'] = [[] for _ in range(data_generator.n_datasets)]
        for sess, dataset in enumerate(data_generator.datasets):
                results_dict['labels_strong'][sess] = [np.array([]) for _ in range(dataset.n_sequences)]

        
        for key in keys:
            results_dict[key] = [[] for _ in range(data_generator.n_datasets)]

            for sess, dataset in enumerate(data_generator.datasets):
                results_dict[key][sess] = [np.array([]) for _ in range(dataset.n_sequences)]
                

        # partially fill container (gap trials will be included as nans)
        dtypes = ['train', 'val', 'test']
        y_dim = self.hparams['n_total_classes']
        for dtype in dtypes:
            data_generator.reset_iterators(dtype)
            for i in range(data_generator.n_tot_batches[dtype]):
                data, sess_list = data_generator.next_batch(dtype)  
                    
                outputs_dict = self.predict_argmax_labels(data['markers'], data['labels_strong'])
                #outputs_dict = self.forward(data['markers'], data['labels_strong'])
                # remove padding if necessary
                if pad > 0 and remove_pad:
                    #markers = markers_wpad[:, pad:-pad, ...]
                    # remove padding from model output
                    for key, val in outputs_dict.items():
                        #print('key: ', key)
                        #print('val', val.shape)
                        #print(key, outputs_dict[key].shape)
                        #print('val.shape', val.shape)
                        if key in ['pz_logvar', 'qy_x_probs', 'qy_e_probs', 'qy_x_logits', 'idxs_labeled', 'qz_xy_mean']:
                            outputs_dict[key] = val[:, pad:-pad, ...] if val is not None else None
                        else:
                            outputs_dict[key] = val[:, :, pad:-pad, ...] if val is not None else None
#                         if val.shape[0] != y_dim:
#                             outputs_dict[key] = val[:, pad:-pad, ...] if val is not None else None
#                         else:
#                             outputs_dict[key] = val[:, :, pad:-pad, ...] if val is not None else None
                        #print(key, outputs_dict[key].shape)
                    data['markers'] = data['markers'][:, pad:-pad] 
                    data['labels_strong'] = data['labels_strong'][:, pad:-pad]

                # loop over sequences in batch
                for s, sess in enumerate(sess_list):
                    batch_idx = data['batch_idx'][s].item()
                    
                    results_dict['markers'][sess][batch_idx] = \
                    data['markers'][s].cpu().detach().numpy()
                    
                    results_dict['labels_strong'][sess][batch_idx] = \
                    data['labels_strong'][s].cpu().detach().numpy()
                    
                  
                    for key in keys:   
                        results_dict[key][sess][batch_idx] = \
                        outputs_dict[key][s].cpu().detach().numpy()
 
        return results_dict
    
    def sampler(self, num_samples=1000):
        
        y_samples = []
        z_samples = []
        x_samples = []
        #self.hparams['py_1_probs'] = [.25,.25,.25,.25]
        self.hparams['py_1_probs'] = [0, -1.09861, -1.09861, -1.09861, -1.09861]
        y_dim = self.hparams['n_total_classes']
        #self.hparams['n_total_classes'] = 5
        
        ##################
        # for t = 1
        ##################
        
        # sample y_1
        py_1_probs = torch.tensor(self.hparams['py_1_probs'])
        
        py_1 = Categorical(logits=py_1_probs)
        y_1 = py_1.sample()
        y_1 = MakeOneHot()(y_1.unsqueeze(-1), self.hparams['n_total_classes']).type(torch.LongTensor)
        y_samples.append(torch.squeeze(y_1))
        
        # sample z_1
        z_1 = torch.normal(torch.tensor([0,1]).type(torch.FloatTensor), torch.ones(self.hparams['n_latents']) *1.5)
        z_samples.append(z_1)
        
        # sample x_1 | z_1
        
        ##################
        # for t = 2 to T
        ##################
        
        
        for t in range(1, num_samples):
            
            # sample y_t | y_(t-1), z_(t-1)
            py_t_probs = self.model['py_t_probs'](z_samples[t-1])
            
            #print('py p', py_t_probs, py_t_probs.shape)
            py_t = Categorical(logits=py_t_probs)
            #print('py_t_probs', py_t_probs.shape, py_t_probs)
            y_t = py_t.sample()
            
            y_t = MakeOneHot()(y_t.unsqueeze(-1), self.hparams['n_total_classes']).type(torch.LongTensor)
            y_t = torch.squeeze(y_t).type(torch.LongTensor)
            y_samples.append(y_t)

            # sample z_t | y_t, z_(t-1)
            

            pz_t_mean = self.model['pz_t_mean_{}'.format(torch.argmax(y_samples[t]))](z_samples[t-1])       
         
       
            # use fix var for test
            pz_t_logvar = torch.ones_like(pz_t_mean) * 1e-4#self.model['pz_t_logvar'](y_samples[t].type(torch.FloatTensor))
            
            z_t = torch.normal(pz_t_mean, pz_t_logvar)
            z_samples.append(z_t)
        
        
        # sample x_t | z_t
        for t in range(0, num_samples):
            
            x_sample = self.model['decoder'](z_samples[t])
            x_samples.append(x_sample)

        return y_samples, z_samples, x_samples    
        
    def build_model(self):
        """Construct the model using hparams."""

        # set random seeds for control over model initialization
        rng_seed_model = self.hparams.get('rng_seed_model', 0)
        torch.manual_seed(rng_seed_model)
        np.random.seed(rng_seed_model)
        
        n_total_classes = self.hparams['n_observed_classes'] + self.hparams['n_aug_classes']
        self.hparams['n_total_classes'] = n_total_classes

        self.inference.build_model()
        self.generative.build_model()
    
    def forward(self, x, y):
        """Process input data.
        
        Parameters
        ----------
        x : torch.Tensor
            observation data of shape (n_sequences, sequence_length, n_markers)
        y : torch.Tensor
            label data of shape (n_sequences, sequence_length)

        Returns
        -------
        dict of model outputs/internals as torch tensors
            - 'y_probs' (torch.Tensor): model classification
               shape of (n_sequences, sequence_length, n_classes)
            - 'y_sample' (torch.Tensor): sample from concrete distribution
              shape of (n_sequences, sequence_length, n_classes)
            - 'z_mean' (torch.Tensor): mean of appx posterior of latents in variational models
              shape of (n_sequences, sequence_length, embedding_dim)
            - 'z_logvar' (torch.Tensor): logvar of appx posterior of latents in variational models
              shape of (n_sequences, sequence_length, embedding_dim)
            - 'reconstruction' (torch.Tensor): input decoder prediction
              shape of (n_sequences, sequence_length, n_markers)

        """
        
        inf_outputs = self.inference(x, y)
        
        gen_inputs = {
           'y_probs': inf_outputs['qy_x_probs'],
           'z_sample': inf_outputs['z_xy_sample'], 
           'idxs_labeled': inf_outputs['idxs_labeled']
        }
        idxs_labeled = inf_outputs['idxs_labeled']
        #print('idxs_labeled', idxs_labeled, idxs_labeled.shape, idxs_labeled.sum().item())
        gen_outputs = self.generative(**gen_inputs)

        # merge the two outputs
        output_dict = {**inf_outputs, **gen_outputs}

        return output_dict
    
    def get_expectation(self, values, probs, axis=1):
        expectation = torch.sum(values*probs, axis=axis)
        return expectation
    
    def training_step(self, data, accumulate_grad=True, **kwargs):
        """Calculate negative log-likelihood loss for supervised models.
        The batch is split into chunks if larger than a hard-coded `chunk_size` to keep memory
        requirements low; gradients are accumulated across all chunks before a gradient step is
        taken.
        Parameters
        ----------
        data : dict
            signals are of shape (n_sequences, sequence_length, n_channels)
        accumulate_grad : bool, optional
            accumulate gradient for training step
        Returns
        -------
        dict
            - 'loss' (float): total loss (negative log-like under specified noise dist)
            - other loss terms depending on model hyperparameters
        """

        # define hyperparams
        device = self.hparams['device']
        kl_y_weight = self.hparams.get('kl_y_weight', 100)
        lambda_strong = self.hparams.get('lambda_strong', 1)
        ignore_class = self.hparams.get('ignore_class', 0)
        kl_z_weight = self.hparams.get('kl_z_weight', 1)
        ann_weight = self.hparams.get('ann_weight', 1)
        recon_weight = self.hparams.get('recon_weight', 1)
        
        mask_unlabeled = self.hparams.get('mask_unlabeled', False)

        # index padding for convolutions
        pad = self.hparams.get('sequence_pad', 0)

        # push data through model
        markers_wpad = data['markers']
        labels_wpad = data['labels_strong']


        outputs_dict = self.forward(markers_wpad, labels_wpad)

        # remove padding from supplied data
        if pad > 0:
            labels_strong = data['labels_strong'][:, pad:-pad, ...]
        else:
            labels_strong = data['labels_strong']
        
        # remove padding from model output
        y_dim = self.hparams['n_total_classes']
        z_dim = self.hparams['n_latents']
        
        if pad > 0:
            markers = markers_wpad[:, pad:-pad, ...]
            # remove padding from model output
            for key, val in outputs_dict.items():
                #print('key', key, 'val ', val.shape)
                if key in ['pz_logvar', 'qy_x_probs', 'qy_e_probs', 'qy_x_logits', 'idxs_labeled']:
                    #val.shape[0] != y_dim or key == 'idxs_labeled':
                    outputs_dict[key] = val[:, pad:-pad, ...] if val is not None else None
                else:
                    outputs_dict[key] = val[:, :, pad:-pad, ...] if val is not None else None
        else:
            markers = markers_wpad
            
        # reshape everything to be (n_sequences * sequence_length, ...)
        N = markers.shape[0] * markers.shape[1]
        markers_rs = torch.reshape(markers, (N, markers.shape[-1]))
        labels_rs = torch.reshape(labels_strong, (N,))
        
        outputs_dict_rs = {}
      
        for key, val in outputs_dict.items():
            
            if isinstance(val, torch.Tensor):
                if key not in ['pz_logvar', 'qy_x_probs', 'qy_e_probs', 'qy_x_logits', 'idxs_labeled']:#val.shape[0] == y_dim:
                    shape = (y_dim, N) + tuple(val.shape[3:])
                    outputs_dict_rs[key] = torch.reshape(val, shape) 
                elif len(val.shape) > 2:
                    shape = (N, ) + tuple(val.shape[2:])
                    outputs_dict_rs[key] = torch.reshape(val, shape)
                else:
                    # when the input is (n_sequences, sequence_length), we want the output to be (n_sequences * sequence_length)
                    outputs_dict_rs[key] = torch.reshape(val, (N, 1))
            else:
                outputs_dict_rs[key] = val
         
        # pull out indices of labeled data for loss computation
        idxs_labeled = outputs_dict_rs['idxs_labeled'].squeeze(-1)

        # initialize loss to zero
        #torch.autograd.set_detect_anomaly(True)
        loss = 0
        loss_dict = {}
        qy_e_probs = outputs_dict_rs['qy_e_probs']
        
                
        # ----------------------------------------------
        # compute classification loss on labeled data
        # ----------------------------------------------
        if lambda_strong > 0:
            loss_strong = self.class_loss(outputs_dict_rs['qy_x_logits'], labels_rs) * lambda_strong
            loss += loss_strong
            # log
            loss_dict['loss_classifier'] = loss_strong.item()     
        
        # ------------------------------------
        # compute reconstruction loss
        # ------------------------------------ 
        reconstruction = outputs_dict_rs['reconstruction'] # (y_dim, N, n_markers)
        px_z_mean = reconstruction
        px_z_std = torch.ones_like(px_z_mean)* .5
        px_z = Normal(px_z_mean, px_z_std)
        
        # diff between log prob of adding 1d Normals and MVN log prob
        k = markers_rs.shape[1]
        mvn_scalar =  (-1) * .5 * math.log(k)
        
        # sum over marker dim
        loss_reconstruction = torch.sum(px_z.log_prob(markers_rs), axis=2) 
        
        # take transpose to get shape N * y_dim
        loss_reconstruction = torch.transpose(loss_reconstruction, 0, 1)
        
        # get expectation to get shape N
        loss_reconstruction = self.get_expectation(loss_reconstruction, qy_e_probs)
        
        if mask_unlabeled:
            loss_reconstruction = loss_reconstruction[idxs_labeled.bool()]
           
        # average over batch dim
        loss_reconstruction = (torch.mean(loss_reconstruction, axis=0) + mvn_scalar )* (-1) * recon_weight
                              
        loss += loss_reconstruction
        # log
        loss_dict['loss_reconstruction'] = loss_reconstruction.item()
        
        # -------------------------------------------------------
        # compute log p(y_t | y_(t-1), z_(t-1)) loss for labeled
        # -------------------------------------------------------    
        
        if idxs_labeled.sum() > 0:
            
            py_logits = outputs_dict_rs['py_logits']  # (n_total_classes, N, n_total_classes) first dim is y_(t-1)
            y_input = labels_rs.clone()
            y_input[labels_rs == ignore_class] = 1  # shape (N,) this is arbitray so that logprob doesn't throw errors

            logpy = []
            for k in range(y_dim):
                py = Categorical(logits=py_logits[k])

                # evaluate prob of true labels
                logpy.append(py.log_prob(y_input))
                
            logpy = torch.stack(logpy, dim=0).to(device=device) # (y_dim, N)
            
            # marginalize over conditioning var y_{t-1}
            e_py = self.get_expectation(logpy[:, 1:].transpose(0,1), qy_e_probs[:-1, :], axis=1).to(device=device) # (N-1,)
            
            # this adds logp y_1 and makes sure the vector is length N
            if labels_rs[0] != ignore_class:
                e_py = torch.cat([logpy[labels_rs[0], 0].unsqueeze(0), e_py], dim=0)              
            else:
                e_py = torch.cat([torch.zeros(1).to(device=device), e_py], dim=0)
            
            # subselect results from labeled data
            e_py_labeled = e_py[labels_rs != ignore_class] # (N_labeled,)
            
            loss_py = torch.mean(e_py_labeled, axis=0) * (-1) * ann_weight
            
            loss += loss_py

            # log
            loss_dict['loss_py'] = loss_py.item()


        # ----------------------------------------------------------------------------------
        # compute kl loss between q(y_t|x_(T_t) and p(y_t|y_(t-1), z_(t-1)) for all (unlabeled?)
        # ----------------------------------------------------------------------------------
        
         # 'py_logits': py_logits, # (n_total_classes, N, n_total_classes)
         # 'y_logits': y_logits, # (N, n_classes)
            
        # create classifier q(y_t|x_t) 
        qy_logits = outputs_dict_rs['qy_x_logits']  # (1, N, n_classes)
        qy = Categorical(logits=qy_logits.unsqueeze(0))

        # create prior p(y)
        py_logits = outputs_dict_rs['py_logits']  # (n_classes, N, n_classes)
        py = Categorical(logits=py_logits)

        loss_y_kl = torch.transpose(kl_divergence(qy, py), 0, 1) # (N, n_classes)

        # get expectation
        expectation_probs = torch.vstack((torch.ones(y_dim).to(device=device)/y_dim, qy_e_probs[:-1].to(device=device))).to(device=device)
        
        # create kl matrix
        kl_matrix = loss_y_kl.clone() # (N, n_classes)
        
        # compute expectation across all timepoints, labeled and unlabeled
        #print('eshape loss_y_kl ep' ,loss_y_kl.shape, expectation_probs.shape)
        loss_y_kl = self.get_expectation(loss_y_kl, expectation_probs) # (N,)
        

        # subselect unlabeled data, mean over batch dim
        betas = torch.tensor([1, .5, .25], device=device) * 10
        
        #if mask_unlabeled:
        #    loss_reconstruction = loss_reconstruction[idxs_labeled.bool()]
            
        loss_y_kl = torch.mean(loss_y_kl[idxs_labeled == 0], axis=0) *betas[0]
        # for D = 3, add 2 extra terms
        pad = self.hparams.get('sequence_pad', 0)
        if 'overshoot_y' in self.hparams:
            D=3
            #print('N', N)
            py_logits_old_rs = outputs_dict_rs['qy_e_probs']
            py_logits_new = outputs_dict['qy_e_probs']
            z_sample_new = outputs_dict['z_xy_sample']

            # start with pushing py_logits and z_sample (from q(z|x,y)) into the generative model
            gen_params = {'y_probs': py_logits_new, 'z_sample': z_sample_new, 'idxs_labeled': []}

            for d in range(1, D):
                # start with pushing py_logits and z_sample (from q(z|x,y)) into the generative model
                gen_new = self.generative(**gen_params)
                
                # extract the new py and pz values
                py_logits_new, pz_mean_new, pz_logvar_new = gen_new['py_logits'], gen_new['pz_mean'], gen_new['pz_logvar']
                
                # reshape the outputs
                py_logits_new_rs = py_logits_new.reshape((y_dim, N, y_dim))
                py_logits_new_rs = nn.Softmax(dim=2)(py_logits_new_rs)

                pz_mean_new = pz_mean_new.reshape((y_dim, N, y_dim, z_dim))
                pz_logvar_new = pz_logvar_new.reshape((N, y_dim, z_dim))

                pz_mean_temp = torch.zeros((N, y_dim, z_dim), device=pz_mean_new.device)
                pz_logvar_temp = torch.zeros((N, z_dim), device=pz_mean_new.device)
                pz_mean_perm = pz_mean_new.permute((1, 0, 2, 3))
                py_logits_e = torch.zeros((N, y_dim), device=pz_mean_new.device)
                py_logits_re = py_logits_new_rs.permute(1,0,2)
                
                for y_class in range(y_dim):
                    pz_mean_temp += pz_mean_perm[:,:,y_class,:] * py_logits_old_rs[:,y_class].reshape(-1, 1, 1)
                    pz_logvar_temp += pz_logvar_new[:, y_class, :] * py_logits_old_rs[:,y_class].reshape(-1, 1)
                    py_logits_e += py_logits_re[:, y_class, :] * py_logits_old_rs[:,y_class].reshape(-1, 1)

                #py_logits_e = py_logits_e.permute(1,0)
                z_sample_new = pz_mean_temp + torch.randn_like(pz_mean_temp, device=pz_mean_new.device) * pz_logvar_temp.exp().sqrt().reshape(N,-1,z_dim)
                z_sample_new = z_sample_new.permute(1,0,2)
                #print(z_sample_new.shape, 'z_sample_new ')
                loss_y_kl_new = betas[d] * self.get_expectation(kl_matrix[d:, :], py_logits_e[:-d, :])
                loss_y_kl += torch.mean(loss_y_kl_new, axis=0) 
                
                batch_size = self.hparams['batch_size']
                seq_len = self.hparams['sequence_length']
                gen_params['y_probs'] = torch.reshape(py_logits_e, (batch_size,seq_len,y_dim))
                gen_params['z_sample'] = torch.reshape(z_sample_new, (y_dim,batch_size,seq_len,z_dim))
                py_logits_old_rs = py_logits_e
        
        if not mask_unlabeled:
            
            loss += loss_y_kl * kl_y_weight * ann_weight
            loss_dict['loss_y_kl'] = loss_y_kl.item()
        
        # ----------------------------------------------------------------------------------
        # compute kl loss between q(y_t|x_(T_t) and p(y_1) for all (uniform kl)
        # ----------------------------------------------------------------------------------
        if self.hparams['kl_y_weight_uniform'] > 0:
            py_logits =  torch.cat((torch.tensor([.001]), torch.ones((y_dim-1))/(y_dim-1))).to(device=outputs_dict_rs['qy_x_logits'].device)
            
            #print('py', py_logits.shape,py_logits)
            py = Categorical(py_logits) 

            qy_logits = outputs_dict_rs['qy_x_logits'].mean(dim=0) # (N, n_classes)

            qy = Categorical(logits=qy_logits)

            loss_y_kl_uniform = torch.mean(kl_divergence(qy, py), axis=0) 
            loss_y_kl_uniform = loss_y_kl_uniform * self.hparams['kl_y_weight_uniform']
            #print("self.hparams['kl_y_weight_uniform']", self.hparams['kl_y_weight_uniform'])

            loss += loss_y_kl_uniform
            loss_dict['loss_y_kl_uniform'] = loss_y_kl_uniform.item()
   
        
        # ----------------------------------------------------------------------------------
        # compute entropy loss on q(y_t|x_(T_t)  for all
        # ----------------------------------------------------------------------------------
        
        
#         qy_logits = outputs_dict_rs['qy_x_logits']# (N, n_classes)
#         qy = Categorical(logits=qy_logits)
        
#         #print('qy', qy_logits.shape, qy_logits)
#         #print('ent', qy.entropy(), qy.entropy().shape)
#         loss_entropy = torch.mean(qy.entropy(), axis=0) 
#         loss_entropy = loss_entropy *  self.hparams['entropy_weight'] #* .01
#         #print('lo', loss_entropy)
     
        
#         loss += loss_entropy

#         loss_dict['loss_entropy'] = loss_entropy.item()
#         #print('kl y loss', loss_y_kl)
        
        # ----------------------------------------
        # compute kl divergence b/t qz_xy and pz_y
        # ----------------------------------------   
        
        # build MVN q(z|x,y)
        qz_mean = outputs_dict_rs['qz_xy_mean']  # qz_mean shape (y_dim, N, n_latents)
        qz_std = outputs_dict_rs['qz_xy_logvar'].exp().pow(0.5)  
        qz = Normal(qz_mean, qz_std)
        
        # first dim is y_(t-1) 3rd dim is y_t
        pz_mean = outputs_dict_rs['pz_mean'] # (n_total_classes, N, n_total_classes, embedding_dim) 
        
        pz_std = torch.transpose(outputs_dict_rs['pz_logvar'], 1, 0).exp().pow(0.5) # (n_total_classes, N, embedding_dim)

        loss_z_kl = 0
        for k_prime in range(y_dim): # k_prime loops over y_(t-1)

            # build MVN p(z|y)
            # pz_mean shape (y_dim, N, n_latents))
            pz_mean_temp = torch.transpose(pz_mean[k_prime], 1, 0)
            pz = Normal(pz_mean_temp, pz_std)

            # compute kl; final shape (N, y_dim)
            kl_temp = torch.transpose(torch.sum(kl_divergence(qz, pz), axis=2), 1, 0)

            # expectation over y_t
            loss_z_kl_temp = torch.sum(
                outputs_dict_rs['qy_e_probs'][1:] * kl_temp[1:], axis=1)
            # multiply by probability for y_{t-1}
            loss_z_kl_temp *= outputs_dict_rs['qy_e_probs'][:-1, k_prime]

            # mean over batch
            if mask_unlabeled:
                loss_z_kl_temp = loss_z_kl_temp[idxs_labeled[1:].bool()]
            loss_z_kl += torch.mean(loss_z_kl_temp, axis=0)


        loss += kl_z_weight * (loss_z_kl * ann_weight)
        # log
        loss_dict['loss_z_kl'] = kl_z_weight * (loss_z_kl.item() * ann_weight)
       
        if accumulate_grad:
            loss.backward()

        # collect loss vals
        loss_dict['loss'] = loss.item()

        return loss_dict
    

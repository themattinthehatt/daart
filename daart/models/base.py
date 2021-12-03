"""Base models/modules in PyTorch."""

import math
import numpy as np
import os
import pickle
from scipy.special import softmax as scipy_softmax
from scipy.stats import entropy
import torch
from sklearn.metrics import accuracy_score, r2_score
from torch import nn, save

# to ignore imports for sphix-autoapidoc
__all__ = ['BaseModel', 'Segmenter', 'Ensembler']


class BaseModel(nn.Module):
    """Template for PyTorch models."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __str__(self):
        """Pretty print model architecture."""
        raise NotImplementedError

    def build_model(self):
        """Build model from hparams."""
        raise NotImplementedError

    @staticmethod
    def _build_linear(global_layer_num, name, in_size, out_size):

        linear_layer = nn.Sequential()

        # add layer (cross entropy loss handles activation)
        layer = nn.Linear(in_features=in_size, out_features=out_size)
        layer_name = str('dense(%s)_layer_%02i' % (name, global_layer_num))
        linear_layer.add_module(layer_name, layer)

        return linear_layer

    def forward(self, *args, **kwargs):
        """Push data through model."""
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        """Compute loss."""
        raise NotImplementedError

    def save(self, filepath):
        """Save model parameters."""
        save(self.state_dict(), filepath)

    def get_parameters(self):
        """Get all model parameters that have gradient updates turned on."""
        return filter(lambda p: p.requires_grad, self.parameters())

    def load_parameters_from_file(self, filepath):
        """Load parameters from .pt file."""
        self.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))


class Segmenter(BaseModel):
    """General wrapper class for behavioral segmentation models."""

    def __init__(self, hparams):
        """

        Parameters
        ----------
        hparams : dict
            - model_type (str): 'temporal-mlp' | 'dtcn' | 'lstm' | 'gru' | 'tgm'
            - rng_seed_model (int): random seed to control weight initialization
            - input_size (int): number of input channels
            - output_size (int): number of classes
            - task_size (int): number of regression tasks
            - batch_pad (int): padding needed to account for convolutions
            - n_hid_layers (int): hidden layers of network architecture
            - n_hid_units (int): hidden units per layer
            - n_lags (int): number of lags in input data to use for temporal convolution
            - activation (str): 'linear' | 'relu' | 'lrelu' | 'sigmoid' | 'tanh'
            - lambda_weak (float): hyperparam on weak label classification
            - lambda_strong (float): hyperparam on srong label classification
            - lambda_pred (float): hyperparam on next step prediction
            - lambda_task (float): hyperparam on task regression

        """
        super().__init__()
        self.hparams = hparams
        self.model = None
        self.build_model()

        # label loss based on cross entropy; don't compute gradient when target = 0
        self.class_loss = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
        self.pred_loss = nn.MSELoss(reduction='mean')
        self.task_loss = nn.MSELoss(reduction='mean')

    def __str__(self):
        """Pretty print model architecture."""
        return self.model.__str__()

    def build_model(self):
        """Construct the model using hparams."""

        # set random seeds for control over model initialization
        rng_seed_model = self.hparams.get('rng_seed_model', 0)
        torch.manual_seed(rng_seed_model)
        np.random.seed(rng_seed_model)

        if self.hparams['model_type'].lower() == 'temporal-mlp':
            from daart.models.temporalmlp import TemporalMLP
            self.model = TemporalMLP(self.hparams)
        elif self.hparams['model_type'].lower() == 'tcn':
            raise NotImplementedError('Split classifiers have not been implemented for TCN')
            # from daart.models.tcn import TCN
            # self.model = TCN(self.hparams)
        elif self.hparams['model_type'].lower() == 'dtcn':
            from daart.models.tcn import DilatedTCN
            self.model = DilatedTCN(self.hparams)
        elif self.hparams['model_type'].lower() in ['lstm', 'gru']:
            from daart.models.rnn import RNN
            self.model = RNN(self.hparams)
        elif self.hparams['model_type'].lower() == 'tgm':
            raise NotImplementedError
            # from daart.models.tgm import TGM
            # self.model = TGM(self.hparams)
        else:
            raise ValueError('"%s" is not a valid model type' % self.hparams['model_type'])

    def forward(self, x):
        """Process input data."""
        return self.model(x)

    def predict_labels(self, data_generator, return_scores=False, remove_pad=True):
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
        self.eval()

        pad = self.hparams.get('batch_pad', 0)

        softmax = nn.Softmax(dim=1)

        # initialize containers

        # softmax outputs
        labels = [[] for _ in range(data_generator.n_datasets)]
        # logits
        scores = [[] for _ in range(data_generator.n_datasets)]
        # latent representation
        embedding = [[] for _ in range(data_generator.n_datasets)]
        # predictions on regression task
        task_predictions = [[] for _ in range(data_generator.n_datasets)]
        for sess, dataset in enumerate(data_generator.datasets):
            labels[sess] = [np.array([]) for _ in range(dataset.n_trials)]
            scores[sess] = [np.array([]) for _ in range(dataset.n_trials)]
            embedding[sess] = [np.array([]) for _ in range(dataset.n_trials)]
            task_predictions[sess] = [np.array([]) for _ in range(dataset.n_trials)]

        # partially fill container (gap trials will be included as nans)
        dtypes = ['train', 'val', 'test']
        for dtype in dtypes:
            data_generator.reset_iterators(dtype)
            for i in range(data_generator.n_tot_batches[dtype]):
                data, sess = data_generator.next_batch(dtype)
                predictors = data['markers'][0]
                # targets = data['labels'][0]
                batch_idx = data['batch_idx'].item()
                outputs_dict = self.model(predictors)
                # remove padding if necessary
                if pad > 0 and remove_pad:
                    for key, val in outputs_dict.items():
                        outputs_dict[key] = val[pad:-pad] if val is not None else None
                # push through log-softmax, since this is included in the loss and not model
                labels[sess][batch_idx] = \
                    softmax(outputs_dict['labels']).cpu().detach().numpy()
                embedding[sess][batch_idx] = \
                    outputs_dict['embedding'].cpu().detach().numpy()
                if return_scores:
                    scores[sess][batch_idx] = \
                        outputs_dict['labels'].cpu().detach().numpy()
                if outputs_dict.get('task_prediction', None) is not None:
                    task_predictions[sess][batch_idx] = \
                        outputs_dict['labels'].cpu().detach().numpy()

        return {
            'labels': labels,
            'scores': scores,
            'embedding': embedding,
            'task_predictions': task_predictions
        }

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
            - other loss terms depending on model hyperparameters

        """

        # define hyperparams
        lambda_weak = self.hparams.get('lambda_weak', 0)
        lambda_strong = self.hparams.get('lambda_strong', 0)
        lambda_pred = self.hparams.get('lambda_pred', 0)
        lambda_task = self.hparams.get('lambda_task', 0)

        # index padding for convolutions
        pad = self.hparams.get('batch_pad', 0)

        # push data through model
        markers_wpad = data['markers'][0]
        outputs_dict = self.model(markers_wpad)

        # get masks that define where strong labels are
        if lambda_strong > 0:
            if pad > 0:
                labels_strong = data['labels_strong'][0][pad:-pad]
            else:
                labels_strong = data['labels_strong'][0]
        else:
            labels_strong = None

        if lambda_weak > 0:
            if pad > 0:
                labels_weak = data['labels_weak'][0][pad:-pad]
            else:
                labels_weak = data['labels_weak'][0]
        else:
            labels_weak = None

        if lambda_task > 0:
            if pad > 0:
                tasks = data['tasks'][0][pad:-pad]
            else:
                tasks = data['tasks'][0]
        else:
            tasks = None

        # remove padding from other tensors
        if pad > 0:
            markers = markers_wpad[pad:-pad]
            # remove padding from model output
            for key, val in outputs_dict.items():
                outputs_dict[key] = val[pad:-pad] if val is not None else None
        else:
            markers = markers_wpad

        # initialize loss to zero
        loss = 0
        loss_dict = {}

        # ------------------------------------
        # compute loss on weak labels
        # ------------------------------------
        if lambda_weak > 0:
            # only compute loss where strong labels do not exist [indicated by a zero]
            if labels_strong is not None:
                try:
                    loss_weak = self.class_loss(
                        outputs_dict['labels_weak'][labels_strong == 0],
                        labels_weak[labels_strong == 0])
                except:
                    print("num strong labels: {}".format(torch.sum(labels_strong)))
                    print("gt: {}".format(torch.sum(outputs_dict['labels_weak'], dim=0)))
                    print("pred: {}".format(torch.sum(labels_weak, dim=0)))
                    print(outputs_dict['labels_weak'][labels_strong == 0])
                    print(labels_weak[labels_strong == 0])
            else:
                loss_weak = self.class_loss(outputs_dict['labels_weak'], labels_weak)
            loss += lambda_weak * loss_weak
            loss_weak_val = loss_weak.item()
            # compute fraction correct on weak labels
            fc = accuracy_score(
                labels_weak.cpu().detach().numpy(),
                np.argmax(outputs_dict['labels'].cpu().detach().numpy(), axis=1)
            )
            # log
            loss_dict['loss_weak'] = loss_weak_val
            loss_dict['fc'] = fc

        # ------------------------------------
        # compute loss on strong labels
        # ------------------------------------
        if lambda_strong > 0:
            loss_strong = self.class_loss(outputs_dict['labels'], labels_strong)
            loss += lambda_strong * loss_strong
            loss_strong_val = loss_strong.item()
            # log
            loss_dict['loss_strong'] = loss_strong_val

        # ------------------------------------
        # compute loss on one-step predictions
        # ------------------------------------
        if lambda_pred > 0:
            loss_pred = self.pred_loss(markers[1:], outputs_dict['prediction'][:-1])
            loss += lambda_pred * loss_pred
            loss_pred_val = loss_pred.item()
            # log
            loss_dict['loss_pred'] = loss_pred_val

        # ------------------------------------
        # compute regression loss on tasks
        # ------------------------------------
        if lambda_task > 0:
            loss_task = self.task_loss(tasks, outputs_dict['task_prediction'])
            loss += lambda_task * loss_task
            loss_task_val = loss_task.item()
            r2 = r2_score(
                tasks.cpu().detach().numpy(),
                outputs_dict['task_prediction'].cpu().detach().numpy()
            )
            # log
            loss_dict['loss_task'] = loss_task_val
            loss_dict['task_r2'] = r2

        if accumulate_grad:
            loss.backward()

        # collect loss vals
        loss_dict['loss'] = loss.item()

        return loss_dict


class Ensembler(object):
    """Ensemble of models."""

    def __init__(self, models):
        self.models = models
        self.n_models = len(models)

    def predict_labels(self, data_generator, combine_before_softmax=False, weights=None):
        """Combine class predictions from multiple models by averaging before softmax.

        Parameters
        ----------
        data_generator : DataGenerator object
            data generator to serve data batches
        combine_before_softmax : bool, optional
            True to combine logits across models before taking softmax; False to take softmax for
            each model then combine probabilities

        Returns
        -------
        dict
            - 'labels' (list of lists): corresponding labels

        """

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
                labels_curr = []
                for model in self.models:
                    outputs_dict = model(predictors)
                    # remove padding if necessary
                    if model.hparams.get('batch_pad', 0) > 0:
                        for key, val in outputs_dict.items():
                            outputs_dict[key] = val[pad:-pad] if val is not None else None
                    labels_tmp = outputs_dict['labels'].cpu().detach().numpy()
                    if combine_before_softmax:
                        labels_curr.append(labels_tmp[None, ...])
                    else:
                        labels_curr.append(scipy_softmax(labels_tmp, axis=1)[None, ...])

                # combine predictions across models
                if weights is None:
                    # simple average across models
                    labels_curr = np.mean(np.vstack(labels_curr), axis=0)
                elif isinstance(weights, str) and weights == 'entropy':
                    # weight each model at each time point by inverse entropy of distribution so
                    # that more confident models have a higher weight
                    labels_tmp = np.vstack(labels_curr)
                    # compute entropy across labels
                    ent = entropy(labels_tmp, axis=2)
                    # low entropy = high confidence, weight these more
                    w = 1.0 / ent
                    # normalize over models
                    w /= np.sum(w, axis=0)
                    labels_curr = np.mean(labels_tmp * w[:, :, None], axis=0)
                elif isinstance(weights, (list, tuple, np.ndarray)):
                    # weight each model according to user-supplied weights
                    labels_curr = np.average(np.vstack(labels_curr), axis=0, weights=weights)

                # store predictions
                if combine_before_softmax:
                    # push through softmax
                    labels[sess][data['batch_idx'].item()] = scipy_softmax(labels_curr, axis=1)
                else:
                    labels[sess][data['batch_idx'].item()] = labels_curr

        return {'labels': labels}

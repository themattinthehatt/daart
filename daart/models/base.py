"""Base models/modules in PyTorch."""

import math
import numpy as np
import os
import pickle
from scipy.special import softmax as scipy_softmax
from scipy.stats import entropy
import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim, save, Tensor
from tqdm import tqdm

from daart.io import export_hparams
from daart.train import EarlyStopping, Logger

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

    def fit(self, data_generator, save_path=None, **kwargs):
        """Fit pytorch models with stochastic gradient descent and early stopping.

        Training parameters such as min epochs, max epochs, and early stopping hyperparameters are
        specified in the class attribute `hparams` or specified in the kwargs.

        For more information on how early stopping is implemented, see the class
        :class:`EarlyStopping`.

        Training progess is monitored by calculating the model loss on both training data and
        validation data. The training loss is calculated each epoch, and the validation loss is
        calculated according to the `hparams` key `'val_check_interval'`. For example, if
        `val_check_interval=5` then the validation loss is calculated every 5 epochs. If
        `val_check_interval=0.5` then the validation loss is calculated twice per epoch - after
        the first half of the batches have been processed, then again after all batches have been
        processed.

        Monitored metrics are saved in a csv file in the model directory. This logging is handled
        by the class :class:`Logger`.

        Parameters
        ----------
        data_generator : ConcatSessionsGenerator object
            data generator to serve data batches
        save_path : str, optional
            absolute path to store model and training results
        kwargs : key-val pairs
            - 'learning_rate' (float): adam learning rate
            - 'l2_reg' (float): general l2 reg on parameters
            - 'min_epochs' (int): minimum number of training epochs
            - 'max_epochs' (int): maximum number of training epochs
            - 'val_check_interal' (int): frequency with which to log performance on val data
            - 'rng_seed_train' (int): control order in which data are served to model
            - 'enable_early_stop' (bool): True to use early stopping; False will use max_epochs
            - 'early_stop_history' (int): epochs over which to average early stopping metric
            - 'save_last_model' (bool): true to save out last (as well as best) model

        """

        def get_param(key, default):
            return kwargs.get(key, self.hparams.get(key, default))

        # -----------------------------------
        # get training params
        # -----------------------------------
        # adam learning rate
        learn_rate = get_param('learning_rate', 1e-4)
        self.hparams['learning_rate'] = learn_rate

        # l2 regularization
        l2_reg = get_param('l2_reg', 0)
        self.hparams['l2_reg'] = l2_reg

        # min number of training epochs
        min_epochs = get_param('min_epochs', 10)
        self.hparams['min_epochs'] = min_epochs

        # max number of training epochs
        max_epochs = get_param('max_epochs', 200)
        self.hparams['max_epochs'] = max_epochs

        # how frequently to run model on validation data
        val_check_interval = get_param('val_check_interval', 1)
        self.hparams['val_check_interval'] = val_check_interval

        # rng seed that determines training data order
        rng_seed_train = int(get_param('rng_seed_train', np.random.randint(0, 10000)))
        self.hparams['rng_seed_train'] = rng_seed_train

        # early stopping
        # account for val check interval > 1; for example, if val_check_interval=5 and
        # early_stop_history=20, then we only need the val loss to increase on 20 / 5 = 4
        # successive checks (rather than 20) to terminate training
        patience = get_param('early_stop_history', 10) // val_check_interval
        self.hparams['early_stop_history'] = patience

        # -----------------------------------
        # set up training
        # -----------------------------------
        # optimizer setup
        optimizer = torch.optim.Adam(
            self.get_parameters(), lr=learn_rate, weight_decay=l2_reg, amsgrad=True)

        # logging setup
        logger = Logger(n_datasets=data_generator.n_datasets, save_path=save_path)

        # early stopping setup
        if get_param('enable_early_stop', False):
            early_stop = EarlyStopping(patience=patience, min_epochs=min_epochs)
        else:
            early_stop = None
        self.hparams['enable_early_stop'] = True

        # enumerate batches on which validation metrics should be recorded
        best_val_loss = np.inf
        best_val_epoch = None
        n_train_batches = data_generator.n_tot_batches['train']
        val_check_batch = np.append(
            val_check_interval * n_train_batches *
            np.arange(1, int((max_epochs + 1) / val_check_interval)),
            [n_train_batches * max_epochs, n_train_batches * (max_epochs + 1)]).astype('int')

        # set random seeds for training
        torch.manual_seed(rng_seed_train)
        np.random.seed(rng_seed_train)

        # -----------------------------------
        # train loop
        # -----------------------------------
        i_epoch = 0
        best_model_saved = False
        for i_epoch in tqdm(range(max_epochs + 1)):
            # Note: the 0th epoch has no training (randomly initialized model is evaluated) so we
            # cycle through `max_epochs` training epochs

            # print_epoch(i_epoch, max_epochs)

            # control how data is batched to that models can be restarted from a particular epoch
            torch.manual_seed(rng_seed_train + i_epoch)  # order of batches within datasets
            np.random.seed(rng_seed_train + i_epoch)  # order of datasets

            logger.reset_metrics('train')
            data_generator.reset_iterators('train')

            i_batch = 0
            for i_batch in range(data_generator.n_tot_batches['train']):

                # -----------------------------------
                # train step
                # -----------------------------------

                # model in train mode
                self.train()

                # zero out gradients; don't want gradients from previous iterations
                optimizer.zero_grad()

                # get next minibatch and put it on the device
                data, dataset = data_generator.next_batch('train')

                # call the appropriate loss function
                loss_dict = self.loss(data, dataset=dataset, accumulate_grad=True)
                logger.update_metrics('train', loss_dict, dataset=dataset)

                # step (evaluate untrained network on epoch 0)
                if i_epoch > 0:
                    optimizer.step()

                # --------------------------------------
                # check validation according to schedule
                # --------------------------------------
                curr_batch = (i_batch + 1) + i_epoch * data_generator.n_tot_batches['train']
                if np.any(curr_batch == val_check_batch):

                    logger.reset_metrics('val')
                    data_generator.reset_iterators('val')
                    self.eval()

                    for i_val in range(data_generator.n_tot_batches['val']):
                        # get next minibatch and put it on the device
                        data, dataset = data_generator.next_batch('val')

                        # call the appropriate loss function
                        loss_dict = self.loss(data, dataset=dataset, accumulate_grad=False)
                        logger.update_metrics('val', loss_dict, dataset=dataset)

                    # save best val model
                    if logger.get_loss('val') < best_val_loss:
                        best_val_loss = logger.get_loss('val')
                        self.save(os.path.join(save_path, 'best_val_model.pt'))
                        best_model_saved = True
                        best_val_epoch = i_epoch

                    # export aggregated metrics on val data
                    logger.create_metric_row(
                        dtype='val', epoch=i_epoch, batch=i_batch, dataset=-1, trial=-1,
                        by_dataset=False, best_epoch=best_val_epoch)
                    # export individual dataset metrics on val data
                    if data_generator.n_datasets > 1:
                        for dataset in range(data_generator.n_datasets):
                            logger.create_metric_row(
                                dtype='val', epoch=i_epoch, batch=i_batch, dataset=dataset,
                                trial=-1, by_dataset=True, best_epoch=best_val_epoch)

            # ---------------------------------------
            # export training metrics at end of epoch
            # ---------------------------------------
            # export aggregated metrics on train data
            logger.create_metric_row(
                dtype='train', epoch=i_epoch, batch=i_batch, dataset=-1, trial=-1,
                by_dataset=False, best_epoch=best_val_epoch)
            # export individual dataset metrics on train/val data
            if data_generator.n_datasets > 1:
                for dataset in range(data_generator.n_datasets):
                    logger.create_metric_row(
                        dtype='train', epoch=i_epoch, batch=i_batch, dataset=dataset, trial=-1,
                        by_dataset=True, best_epoch=best_val_epoch)

            # ---------------------------------------
            # check for early stopping
            # ---------------------------------------
            curr_batch = (i_batch + 1) + i_epoch * data_generator.n_tot_batches['train']
            if early_stop is not None and np.any(curr_batch == val_check_batch):
                early_stop.on_val_check(i_epoch, logger.get_loss('val'))
                if early_stop.should_stop:
                    break

        # ---------------------------------------
        # wrap up with final save/eval
        # ---------------------------------------

        # save out last model as best model if no best model saved
        if not best_model_saved:
            self.save(os.path.join(save_path, 'best_val_model.pt'))

        # save out last model
        if get_param('save_last_model', False):
            self.save(os.path.join(save_path, 'last_model.pt'))

        # load weights of best model if not current
        if best_model_saved:
            self.load_state_dict(torch.load(
                os.path.join(save_path, 'best_val_model.pt'),
                map_location=lambda storage, loc: storage))

        # compute test loss
        logger.reset_metrics('test')
        data_generator.reset_iterators('test')
        self.eval()
        for i_test in range(data_generator.n_tot_batches['test']):
            # get next minibatch and put it on the device
            data, dataset = data_generator.next_batch('test')

            # call the appropriate loss function
            logger.reset_metrics('test')
            loss_dict = self.loss(data, dataset=dataset, accumulate_grad=False)
            logger.update_metrics('test', loss_dict, dataset=dataset)

            # calculate metrics for each *batch* (rather than whole dataset)
            logger.create_metric_row(
                'test', i_epoch, i_test, dataset, trial=data['batch_idx'].item(),
                by_dataset=True)

        # save out hparams
        if save_path is not None:
            from daart.io import export_hparams
            export_hparams(self.hparams, filename=os.path.join(save_path, 'hparams.pkl'))


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

    def predict_labels(self, data_generator, return_scores=False):
        """

        Parameters
        ----------
        data_generator : DataGenerator object
            data generator to serve data batches
        return_scores : bool
            return scores before they've been passed through softmax

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
        scores = [[] for _ in range(data_generator.n_datasets)]
        embedding = [[] for _ in range(data_generator.n_datasets)]
        for sess, dataset in enumerate(data_generator.datasets):
            labels[sess] = [np.array([]) for _ in range(dataset.n_trials)]
            scores[sess] = [np.array([]) for _ in range(dataset.n_trials)]
            embedding[sess] = [np.array([]) for _ in range(dataset.n_trials)]

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
                embedding[sess][data['batch_idx'].item()] = \
                    outputs_dict['embedding'].cpu().detach().numpy()
                if return_scores:
                    scores[sess][data['batch_idx'].item()] = \
                        outputs_dict['labels'].cpu().detach().numpy()

        return {'labels': labels, 'scores': scores, 'embedding': embedding}

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
                    outputs_dict['labels_weak'][labels_strong == 0],
                    labels_weak[labels_strong == 0])
            else:
                loss_weak = self.class_loss(outputs_dict['labels_weak'], labels_weak)
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


def print_epoch(curr, total):
    """Pretty print epoch number."""
    if total < 10:
        print('epoch %i/%i' % (curr, total))
    elif total < 100:
        print('epoch %02i/%02i' % (curr, total))
    elif total < 1000:
        print('epoch %03i/%03i' % (curr, total))
    elif total < 10000:
        print('epoch %04i/%04i' % (curr, total))
    elif total < 100000:
        print('epoch %05i/%05i' % (curr, total))
    else:
        print('epoch %i/%i' % (curr, total))

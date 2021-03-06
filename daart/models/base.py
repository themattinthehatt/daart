"""Base models/modules in PyTorch."""

import math
import numpy as np
import os
import pickle
from scipy.special import softmax
import torch
from torch import nn, optim, save, Tensor
from tqdm import tqdm

from daart.io import export_hparams
from daart.train import EarlyStopping, Logger

# to ignore imports for sphix-autoapidoc
__all__ = ['BaseModule', 'BaseModel']


class BaseModule(nn.Module):
    """Template for PyTorch modules."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __str__(self):
        """Pretty print module architecture."""
        raise NotImplementedError

    def build_model(self):
        """Build model from hparams."""
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """Push data through module."""
        raise NotImplementedError

    def freeze(self):
        """Prevent updates to module parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Force updates to module parameters."""
        for param in self.parameters():
            param.requires_grad = True


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
        patience = get_param('early_stop_history', 10)
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

            i_train = 0
            for i_train in range(data_generator.n_tot_batches['train']):

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
                curr_batch = (i_train + 1) + i_epoch * data_generator.n_tot_batches['train']
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
                        'val', i_epoch, i_train, -1, trial=-1,
                        by_dataset=False, best_epoch=best_val_epoch)
                    # export individual dataset metrics on val data
                    if data_generator.n_datasets > 1:
                        logger.create_metric_row(
                            'val', i_epoch, i_train, dataset, trial=-1,
                            by_dataset=True, best_epoch=best_val_epoch)

            # ---------------------------------------
            # export training metrics at end of epoch
            # ---------------------------------------
            # export aggregated metrics on train data
            logger.create_metric_row(
                'train', i_epoch, i_train, -1, trial=-1,
                by_dataset=False, best_epoch=best_val_epoch)
            # export individual dataset metrics on train/val data
            if data_generator.n_datasets > 1:
                for dataset in range(data_generator.n_datasets):
                    logger.create_metric_row(
                        'train', i_epoch, i_train, dataset, trial=-1,
                        by_dataset=True, best_epoch=best_val_epoch)

            # ---------------------------------------
            # check for early stopping
            # ---------------------------------------
            if early_stop is not None:
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


class Ensembler(object):
    """Ensemble of models."""

    def __init__(self, models):
        self.models = models
        self.n_models = len(models)

    def predict_labels(self, data_generator):
        """Combine class predictions from multiple models by averaging before softmax.

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
                    labels_curr.append(outputs_dict['labels'].cpu().detach().numpy()[None, ...])
                labels_curr = np.mean(np.vstack(labels_curr), axis=0)
                # push through softmax, since this is included in the loss and not model
                labels[sess][data['batch_idx'].item()] = softmax(labels_curr, axis=1)

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

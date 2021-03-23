"""Helper functions for model training."""

import copy
import os
import numpy as np
import pandas as pd
import torch

from daart.io import make_dir_if_not_exists

# to ignore imports for sphix-autoapidoc
__all__ = ['Logger', 'EarlyStopping']


class Logger(object):
    """Base class for logging loss metrics.

    Loss metrics are tracked for the aggregate dataset (potentially spanning multiple datasets) as
    well as dataset-specific metrics for easier downstream plotting.
    """

    def __init__(self, n_datasets=1, save_path=None):
        """

        Parameters
        ----------
        n_datasets : int
            total number of datasets served by data generator
        save_path : str
            absolute path to directory where logged values are saved

        """
        if save_path is not None:
            self.save_file = os.path.join(save_path, 'metrics.csv')
            make_dir_if_not_exists(self.save_file)
        else:
            self.save_file = None

        self.metrics = {}
        self.n_datasets = n_datasets
        dtype_strs = ['train', 'val', 'test', 'curr']

        # aggregate metrics over all datasets
        for dtype in dtype_strs:
            self.metrics[dtype] = {}

        # separate metrics by dataset
        self.metrics_by_dataset = []
        if self.n_datasets > 1:
            for dataset in range(self.n_datasets):
                self.metrics_by_dataset.append({})
                for dtype in dtype_strs:
                    self.metrics_by_dataset[dataset][dtype] = {}

        # store all metrics in a list for easy saving
        self.all_metrics_list = []

    def reset_metrics(self, dtype):
        """Reset all metrics.

        Parameters
        ----------
        dtype : str
            datatype to reset metrics for (e.g. 'train', 'val', 'test')

        """
        # reset aggregate metrics
        for key in self.metrics[dtype].keys():
            self.metrics[dtype][key] = 0
        # reset separated metrics
        for m in self.metrics_by_dataset:
            for key in m[dtype].keys():
                m[dtype][key] = 0

    def update_metrics(self, dtype, loss_dict, dataset=None):
        """Update metrics for a specific dtype/dataset.

        Parameters
        ----------
        dtype : str
            dataset type to update metrics for (e.g. 'train', 'val', 'test')
        loss_dict : dict
            key-value pairs correspond to all quantities that should be logged throughout training;
            dictionary returned by `loss` attribute of models
        dataset : int or NoneType, optional
            if NoneType, updates the aggregated metrics; if `int`, updates the associated dataset

        """
        metrics = {**loss_dict, 'batches': 1}  # append `batches` to loss_dict

        for key, val in metrics.items():

            # define metric for the first time if necessary
            if key not in self.metrics[dtype]:
                self.metrics[dtype][key] = 0

            # update aggregate methods
            self.metrics[dtype][key] += val

            # update separated metrics
            if dataset is not None and self.n_datasets > 1:
                if key not in self.metrics_by_dataset[dataset][dtype]:
                    self.metrics_by_dataset[dataset][dtype][key] = 0
                self.metrics_by_dataset[dataset][dtype][key] += val

    def create_metric_row(
            self, dtype, epoch, batch, dataset, trial, best_epoch=None, by_dataset=False):
        """Export metrics and other data (e.g. epoch) for logging train progress.

        Parameters
        ----------
        dtype : str
            'train' | 'val' | 'test'
        epoch : int
            current training epoch
        batch : int
            current training batch
        dataset : int
            dataset id for current batch
        trial : int or NoneType
            trial id within the current dataset
        best_epoch : int, optional
            best current training epoch
        by_dataset : bool, optional
            `True` to return metrics for a specific dataset, `False` to return metrics aggregated
            over multiple datasets

        Returns
        -------
        dict
            aggregated metrics for current epoch/batch

        """

        if dtype == 'train':
            prefix = 'tr'
        elif dtype == 'val':
            prefix = 'val'
        elif dtype == 'test':
            prefix = 'test'
        else:
            raise ValueError("%s is an invalid data type" % dtype)

        metric_row = {
            'epoch': epoch,
            'batch': batch,
            'trial': trial}

        if dtype == 'val':
            metric_row['best_val_epoch'] = best_epoch

        if by_dataset and self.n_datasets > 1:
            norm = self.metrics_by_dataset[dataset][dtype]['batches']
            for key, val in self.metrics_by_dataset[dataset][dtype].items():
                if key == 'batches':
                    continue
                metric_row['%s_%s' % (prefix, key)] = val / norm
        else:
            dataset = -1
            norm = self.metrics[dtype]['batches']
            for key, val in self.metrics[dtype].items():
                if key == 'batches':
                    continue
                metric_row['%s_%s' % (prefix, key)] = val / norm

        metric_row['dataset'] = dataset

        self.all_metrics_list.append(metric_row)

        if self.save_file is not None:
            # save the metrics data
            df = pd.DataFrame(self.all_metrics_list)
            df.to_csv(self.save_file, index=False)

        return metric_row

    def get_loss(self, dtype):
        """Return loss aggregated over all datasets.

        Parameters
        ----------
        dtype : str
            datatype to calculate loss for (e.g. 'train', 'val', 'test')

        """
        return self.metrics[dtype]['loss'] / self.metrics[dtype]['batches']


class EarlyStopping(object):
    """Stop training when a monitored quantity has stopped improving.

    Adapted from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    def __init__(self, patience=10, min_epochs=10, delta=0):
        """

        Parameters
        ----------
        patience : int, optional
            number of previous checks to average over when checking for increase in loss
        min_epochs : int, optional
            minimum number of epochs for training
        delta : float, optional
            minimum change in monitored quantity to qualify as an improvement

        """

        self.patience = patience
        self.min_epochs = min_epochs
        self.delta = delta

        self.counter = 0
        self.best_epoch = 0
        self.best_loss = np.inf
        self.stopped_epoch = 0
        self.should_stop = False

    def on_val_check(self, epoch, curr_loss):
        """Check to see if loss has begun to increase on validation data for current epoch.

        Rather than returning the results of the check, this method updates the class attribute
        :obj:`should_stop`, which is checked externally by the fitting function.

        Parameters
        ----------
        epoch : int
            current epoch
        curr_loss : float
            current loss

        """

        # update best loss and epoch that it happened at
        if curr_loss < self.best_loss - self.delta:
            self.best_loss = curr_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        # check if smoothed loss is starting to increase; exit training if so
        if epoch > self.min_epochs and self.counter >= self.patience:
            print('\n== early stopping criteria met; exiting train loop ==')
            print('training epochs: %d' % epoch)
            print('end cost: %04f' % curr_loss)
            print('best epoch: %i' % self.best_epoch)
            print('best cost: %04f\n' % self.best_loss)
            self.stopped_epoch = epoch
            self.should_stop = True

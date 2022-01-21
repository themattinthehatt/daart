"""Helper functions for model training."""

import copy
import os
import numpy as np
import pandas as pd
import torch
from torch import optim
from tqdm import tqdm
from typing import List, Optional, Union
from typeguard import typechecked

from daart.io import export_hparams
from daart.io import make_dir_if_not_exists

# to ignore imports for sphix-autoapidoc
__all__ = ['Logger', 'Trainer']


class Logger(object):
    """Base class for logging loss metrics.

    Loss metrics are tracked for the aggregate dataset (potentially spanning multiple datasets) as
    well as dataset-specific metrics for easier downstream plotting.
    """

    @typechecked
    def __init__(self, n_datasets: int = 1, save_path: Optional[str] = None) -> None:
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

    @typechecked
    def reset_metrics(self, dtype: str):
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

    @typechecked
    def update_metrics(
            self,
            dtype: str,
            loss_dict: dict,
            dataset: Union[int, np.int64, list, None] = None
    ) -> None:
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

            # update separated metrics if dataset exists and is integer or list of length 1
            if dataset is not None and self.n_datasets > 1:
                if isinstance(dataset, list) and len(dataset) > 1:
                    continue
                elif isinstance(dataset, list) and len(dataset) == 1:
                    dataset_ = dataset[0]
                else:
                    dataset_ = dataset
                if key not in self.metrics_by_dataset[dataset_][dtype]:
                    self.metrics_by_dataset[dataset_][dtype][key] = 0
                self.metrics_by_dataset[dataset_][dtype][key] += val

    @typechecked
    def create_metric_row(
            self,
            dtype: str,
            epoch: Union[int, np.int64],
            batch: Union[int, np.int64],
            dataset: Union[int, np.int64],
            trial: Union[int, np.int64, None],
            best_epoch: Optional[Union[int, np.int64]] = None,
            by_dataset: bool = False
    ) -> dict:
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
            'trial': trial
        }

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

    @typechecked
    def get_loss(self, dtype: str) -> float:
        """Return loss aggregated over all datasets.

        Parameters
        ----------
        dtype : str
            datatype to calculate loss for (e.g. 'train', 'val', 'test')

        Returns
        -------
        float

        """
        return self.metrics[dtype]['loss'] / self.metrics[dtype]['batches']


class Trainer(object):

    @typechecked
    def __init__(
            self,
            learning_rate: float = 1e-4,
            l2_reg: float = 0.0,
            min_epochs: int = 10,
            max_epochs: int = 200,
            val_check_interval: int = 10,
            rng_seed_train: int = 0,
            save_last_model: bool = False,
            callbacks: list = [],
            **kwargs
    ) -> None:
        """Initialize trainer object with hyperparameters.

        Parameters
        ----------
        learning_rate: float, optional
            adam learning rate
        l2_reg: float, optional
            general l2 reg on parameters
        min_epochs: int, optional
            minimum number of training epochs
        max_epochs: int, optional
            maximum number of training epochs
        val_check_interval: int, optional
            frequency with which to log performance on val data
        rng_seed_train: int, optional
            control order in which data are served to model
        save_last_model: bool, optional
            True to save out last (as well as best) model
        callbacks: list, optional
            list of callback objects

        """

        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.val_check_interval = val_check_interval
        self.rng_seed_train = rng_seed_train
        self.save_last_model = save_last_model
        self.callbacks = callbacks

        # monitor training progress
        self.curr_epoch = 0
        self.curr_batch = 0
        self.val_check_batch = None
        self.should_halt = False

    def fit(self, model, data_generator, save_path):
        """Fit pytorch models with stochastic gradient descent and early stopping.

        Training parameters such as min/max epochs are specified in the class constructor.

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
        model : Segmenter object
            daart model to train
        data_generator : ConcatSessionsGenerator object
            data generator to serve data batches
        save_path : str, optional
            absolute path to store model and training results

        """

        # -----------------------------------
        # update training params in model
        # -----------------------------------
        model.hparams['learning_rate'] = self.learning_rate
        model.hparams['l2_reg'] = self.l2_reg
        model.hparams['min_epochs'] = self.min_epochs
        model.hparams['max_epochs'] = self.max_epochs
        model.hparams['val_check_interval'] = self.val_check_interval
        model.hparams['rng_seed_train'] = self.rng_seed_train

        # -----------------------------------
        # set up training
        # -----------------------------------
        # optimizer setup
        optimizer = torch.optim.Adam(
            model.get_parameters(), lr=self.learning_rate, weight_decay=self.l2_reg, amsgrad=True)

        # logging setup
        logger = Logger(n_datasets=data_generator.n_datasets, save_path=save_path)

        # enumerate batches on which validation metrics should be recorded
        best_model_saved = False
        best_val_loss = np.inf
        best_val_epoch = None
        n_train_batches = data_generator.n_tot_batches['train']
        self.val_check_batch = np.append(
            self.val_check_interval * n_train_batches *
            np.arange(1, int((self.max_epochs + 1) / self.val_check_interval)),
            [n_train_batches * self.max_epochs,
             n_train_batches * (self.max_epochs + 1)]).astype('int')

        # set random seeds for training
        torch.manual_seed(self.rng_seed_train)
        np.random.seed(self.rng_seed_train)

        # -----------------------------------
        # train loop
        # -----------------------------------
        for i_epoch in tqdm(range(self.max_epochs + 1)):
            # Note: the 0th epoch has no training (randomly initialized model is evaluated) so we
            # cycle through `max_epochs` training epochs
            self.curr_epoch = i_epoch

            # control how data is batched to that models can be restarted from a particular epoch
            torch.manual_seed(self.rng_seed_train + i_epoch)  # order of batches within datasets
            np.random.seed(self.rng_seed_train + i_epoch)  # order of datasets

            logger.reset_metrics('train')
            data_generator.reset_iterators('train')

            i_batch = 0
            for i_batch in range(data_generator.n_tot_batches['train']):

                if i_epoch > 0:
                    self.curr_batch += 1

                # -----------------------------------
                # train step
                # -----------------------------------

                # model in train mode
                model.train()

                # zero out gradients; don't want gradients from previous iterations
                optimizer.zero_grad()

                # get next minibatch and put it on the device
                data, datasets = data_generator.next_batch('train')

                # call the appropriate loss function
                loss_dict = model.training_step(data, accumulate_grad=True)
                logger.update_metrics('train', loss_dict, dataset=datasets)

                # step (evaluate untrained network on epoch 0)
                if i_epoch > 0:
                    optimizer.step()

                # --------------------------------------
                # check validation according to schedule
                # --------------------------------------
                if np.any(self.curr_batch == self.val_check_batch):

                    logger.reset_metrics('val')
                    data_generator.reset_iterators('val')
                    model.eval()

                    for i_val in range(data_generator.n_tot_batches['val']):
                        # get next minibatch and put it on the device
                        data, datasets = data_generator.next_batch('val')

                        # call the appropriate loss function
                        loss_dict = model.training_step(data, accumulate_grad=False)
                        logger.update_metrics('val', loss_dict, dataset=datasets)

                    # save best val model
                    if logger.get_loss('val') < best_val_loss:
                        best_val_loss = logger.get_loss('val')
                        model.save(os.path.join(save_path, 'best_val_model.pt'))
                        best_model_saved = True
                        best_val_epoch = i_epoch

                    # export aggregated metrics on val data
                    logger.create_metric_row(
                        dtype='val', epoch=i_epoch, batch=i_batch, dataset=-1, trial=-1,
                        by_dataset=False, best_epoch=best_val_epoch)
                    # export individual dataset metrics on val data if possible
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
            if data_generator.n_datasets > 1 and data_generator.batch_size == 1:
                for dataset in range(data_generator.n_datasets):
                    logger.create_metric_row(
                        dtype='train', epoch=i_epoch, batch=i_batch, dataset=dataset,
                        trial=-1, by_dataset=True, best_epoch=best_val_epoch)

            # ---------------------------------------
            # run end-of-epoch callbacks
            # ---------------------------------------
            for callback in self.callbacks:
                callback.on_epoch_end(
                    data_generator=data_generator, model=model, trainer=self, logger=logger)

            if self.should_halt:
                # break out of training loop; trainer.should_halt is modified by callbacks
                break

        # ---------------------------------------
        # wrap up with final save/eval
        # ---------------------------------------

        # save out last model as best model if no best model saved
        if not best_model_saved:
            model.save(os.path.join(save_path, 'best_val_model.pt'))

        # save out last model
        if self.save_last_model:
            model.save(os.path.join(save_path, 'last_model.pt'))

        # load weights of best model if not current
        if best_model_saved:
            model.load_parameters_from_file(os.path.join(save_path, 'best_val_model.pt'))

        # compute test loss
        logger.reset_metrics('test')
        data_generator.reset_iterators('test')
        model.eval()
        for i_test in range(data_generator.n_tot_batches['test']):
            # get next minibatch and put it on the device
            data, datasets = data_generator.next_batch('test')

            # call the appropriate loss function
            logger.reset_metrics('test')
            loss_dict = model.training_step(data, dataset=datasets, accumulate_grad=False)
            logger.update_metrics('test', loss_dict, dataset=datasets)

            # calculate metrics for each *batch* (rather than whole dataset)
            logger.create_metric_row(
                'test', i_epoch, i_test, datasets[0], trial=data['batch_idx'][0].item(),
                by_dataset=True)

        # save out hparams
        if save_path is not None:
            from daart.io import export_hparams
            model.hparams['best_val_epoch'] = best_val_epoch
            export_hparams(model.hparams, filename=os.path.join(save_path, 'hparams.yaml'))

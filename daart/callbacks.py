"""Callback classes to control training."""

import logging
import numpy as np

# to ignore imports for sphix-autoapidoc
__all__ = ['BaseCallback', 'EarlyStopping', 'AnnealHparam', 'PseudoLabels', 'UPS']


class BaseCallback(object):
    """Abstract base class for callbacks."""

    def on_epoch_end(self, data_generator, model, trainer, **kwargs):
        raise NotImplementedError


class EarlyStopping(BaseCallback):
    """Stop training when a monitored quantity has stopped improving.

    Adapted from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    def __init__(self, patience=10, delta=0.0):
        """

        Note: It must be noted that the patience parameter counts the number of validation checks
        with no improvement, and not the number of training epochs. Therefore, with parameters
        `check_val_interval=10` and `patience=3`, the trainer will perform at least 40 training
        epochs before being stopped.

        Parameters
        ----------
        patience : int, optional
            number of previous checks to average over when checking for increase in loss
        delta : float, optional
            minimum change in monitored quantity to qualify as an improvement

        """

        self.patience = patience
        self.delta = delta

        self.counter = 0
        self.best_epoch = 0
        self.best_loss = np.inf

    def on_epoch_end(self, data_generator, model, trainer, logger=None, **kwargs):

        # skip if this is not a validation epoch
        if ~np.any(trainer.curr_batch == trainer.val_check_batch):
            return

        # use overall validation loss for early stopping
        loss = logger.get_loss('val')

        # update best loss and epoch that it occurred
        if loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.best_epoch = trainer.curr_epoch
            self.counter = 0
        else:
            self.counter += 1

        # check if smoothed loss is starting to increase; exit training if so
        if (trainer.curr_epoch > trainer.min_epochs) and (self.counter >= self.patience):
            trainer.should_halt = True
            print_str = '\n== early stopping criteria met; exiting train loop ==\n'
            print_str += 'training epochs: %d\n' % trainer.curr_epoch
            print_str += 'end cost: %04f\n' % loss
            print_str += 'best epoch: %i\n' % self.best_epoch
            print_str += 'best cost: %04f\n' % self.best_loss
            logging.info(print_str)


class AnnealHparam(BaseCallback):
    """Linearly increase value in an hparam dict."""

    def __init__(self, hparams, key, epoch_start, epoch_end, val_start=0):
        """

        Parameters
        ----------
        hparams : dict
            hparam dict that is an attribute of a daart model
        key : str
            key to value to anneal
        epoch_start : int
            keep value at `val_start` until this epoch
        epoch_end : int
            linearly increase value from `epoch_start` to `epoch_end`
        val_start : int, optional

        """

        # basic error checking
        assert epoch_start <= epoch_end
        assert key in hparams.keys()

        # store data
        self.hparams = hparams
        self.key = key
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end
        self.val_start = val_start
        self.val_end = self.hparams[self.key]

    def on_epoch_end(self, data_generator, model, trainer, **kwargs):

        if trainer.curr_epoch < self.epoch_start:
            self.hparams[self.key] = self.val_start
        elif trainer.curr_epoch > self.epoch_end:
            self.hparams[self.key] = self.val_end
        else:
            frac = (trainer.curr_epoch - self.epoch_start) / (self.epoch_end - self.epoch_start)
            self.hparams[self.key] = self.val_end * frac


class PseudoLabels(BaseCallback):
    """Implement PseudoLabels algorithm."""

    def __init__(self, prob_threshold=0.95, epoch_start=10):
        self.prob_threshold = prob_threshold
        self.epoch_start = epoch_start

    def on_epoch_end(self, data_generator, model, trainer, **kwargs):

        if trainer.curr_epoch < self.epoch_start:
            return

        # push training data through model; collect output probabilities
        outputs_dict = model.predict_labels(data_generator, remove_pad=False)

        # outputs_dict['labels']  # list of list of numpy arrays
        # threshold the probabilities to produce pseudo-labels
        pseudo_labels = []
        for dataset in outputs_dict['labels']:
            # `dataset` is a list of numpy arrays
            pseudo_labels_data = []
            for batch in dataset:
                if batch.shape[0] > 0:
                    # batch is a numpy array
                    # set all probabilities > threshold to 1
                    batch[batch >= self.prob_threshold] = 1
                    # set all other probabilities to 0
                    batch[batch < 1] = 0
                    # update background class
                    batch[np.sum(batch, axis=1) == 0, 0] = 1
                    # turn into a one-hot vector
                    batch = np.argmax(batch, axis=1)
                pseudo_labels_data.append(batch.astype(int))
            pseudo_labels.append(pseudo_labels_data)

        # total_new_pseudos = \
        #     np.sum([np.sum([np.sum(b[:, 1:]) for b in data]) for data in pseudo_labels])
        # print(total_new_pseudos)

        # update the data generator with the new psuedo-labels
        for dataset, labels in zip(data_generator.datasets, pseudo_labels):
            dataset.data['labels_weak'] = labels


class UPS(BaseCallback):
    """Implement uncertainty-aware pseudo-labels algorithm.

    See details in: https://arxiv.org/pdf/2101.06329.pdf

    """

    def __init__(self, prob_threshold=0.95, variance_threshold=0.05, epoch_start=10):
        self.prob_threshold = prob_threshold
        self.variance_threshold = variance_threshold
        self.epoch_start = epoch_start

    def on_epoch_end(self, data_generator, model, trainer, **kwargs):

        if trainer.curr_epoch < self.epoch_start:
            return

        # push training data through model 10 times to get a sense of variability in output
        # probabilities; collect output probabilities
        n_passes = 10
        n_datasets = 0
        outputs_list = []  # list (over passes) of lists (over datasets) of lists (over batches)
        for n in range(n_passes):
            # outputs_dict['labels']  # list of list of numpy arrays
            outputs_dict = model.predict_labels(data_generator, remove_pad=False, mode='train')
            outputs_list.append(outputs_dict['labels'])
            n_datasets = len(outputs_dict['labels'])

        # threshold the probabilities and the variances across passes to produce pseudo-labels
        pseudo_labels = []
        for dataset in range(n_datasets):
            n_batches = len(outputs_list[0][dataset])
            pseudo_labels_data = []
            for batch in range(n_batches):
                # compute medians and variances of probabilities across passes
                # batch_data will be of shape (n_t, n_classes, n_passes)
                batch_data = np.concatenate(
                    [o[dataset][batch][:, :, None] for o in outputs_list], axis=2)
                new_batch = np.zeros((batch_data.shape[0], batch_data.shape[1]))
                if batch_data.shape[0] > 0:
                    batch_medians = np.median(batch_data, axis=2)  # shape (n_t, n_classes)
                    batch_vars = np.variance(batch_data, axis=2)  # shape (n_t, n_classes)
                    # set all probabilities > threshold to 1
                    new_batch[
                        (batch_medians >= self.prob_threshold)
                        & (batch_vars <= self.variance_threshold)
                    ] = 1
                    # update background class
                    new_batch[np.sum(new_batch, axis=1) == 0, 0] = 1
                    # turn into a one-hot vector
                    new_batch = np.argmax(new_batch, axis=1)
                pseudo_labels_data.append(new_batch.astype(int))
            pseudo_labels.append(pseudo_labels_data)

        # update the data generator with the new psuedo-labels
        for dataset, labels in zip(data_generator.datasets, pseudo_labels):
            dataset.data['labels_weak'] = labels

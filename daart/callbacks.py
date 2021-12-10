"""Callback classes to control training."""


class BaseCallback(object):
    """Abstract base class for callbacks."""

    def on_epoch_end(self, curr_batch, curr_epoch, model, data_generator, **kwargs):
        raise NotImplementedError


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

    def on_epoch_end(self, curr_batch, curr_epoch, model, data_generator, **kwargs):

        if curr_epoch < self.epoch_start:
            self.hparams[self.key] = self.val_start
        elif curr_epoch > self.epoch_end:
            self.hparams[self.key] = self.val_end
        else:
            frac = (curr_epoch - self.epoch_start) / (self.epoch_end - self.epoch_start)
            self.hparams[self.key] = self.val_end * frac


class PseudoLabels(BaseCallback):
    """Implement PseudoLabels algorithm."""

    def __init__(self, prob_threshold=0.95, epoch_start=10):
        self.prob_threshold = prob_threshold
        self.epoch_start = epoch_start

    def on_epoch_end(self, curr_batch, curr_epoch, model, data_generator, **kwargs):

        if curr_epoch < self.epoch_start:
            return

        # push training data through model; collect output probabilities
        outputs_dict = model.predict_labels(data_gen)

        # outputs_dict['labels']  # list of list of numpy arrays
        # threshold the probabilities to produce pseudo-labels
        pseudo_labels = []
        for dataset in outputs_dict['labels']:
            # dataset is a list of numpy arrays
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
                pseudo_labels_data.append(batch.astype(np.int))
            pseudo_labels.append(pseudo_labels_data)

        # total_new_pseudos = \
        #     np.sum([np.sum([np.sum(b[:, 1:]) for b in data]) for data in pseudo_labels])
        # print(total_new_pseudos)

        # update the data generator with the new psuedo-labels
        for dataset, labels in zip(data_gen.datasets, pseudo_labels):
            dataset.data['labels_weak'] = labels

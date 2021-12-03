"""Callback classes to control training."""


class BaseCallback(object):
    """Abstract base class for callbacks."""

    def on_epoch_end(self, curr_batch, curr_epoch):
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

    def on_epoch_end(self, curr_batch, curr_epoch):

        if curr_epoch < self.epoch_start:
            self.hparams[self.key] = self.val_start
        elif curr_epoch > self.epoch_end:
            self.hparams[self.key] = self.val_end
        else:
            frac = (curr_epoch - self.epoch_start) / (self.epoch_end - self.epoch_start)
            self.hparams[self.key] = self.val_end * frac


class PseudoLabels(BaseCallback):
    """Implements the Pseudo Labels algorithm.

    Dong-hyun Lee, Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep
        Neural Networks

    """

    def __init__(self):
        pass

    def on_epoch_end(self, curr_batch, curr_epoch):
        pass

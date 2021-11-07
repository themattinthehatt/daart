"""Tranform classes to process data.

Data generator objects can apply these transforms to data upon loading.
"""

import numpy as np


class Compose(object):
    """Composes several transforms together.

    Adapted from pytorch source code:
    https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#Compose

    Example
    -------
    .. code-block:: python

        >> Compose([
        >>     daart.transforms.ZScore(),
        >>     daart.transforms.MotionEnergy(),
        >> ])

    Parameters
    ----------
    transforms : list of transform objects
        list of transforms to compose

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, signal):
        for t in self.transforms:
            signal = t(signal)
        return signal

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '{0}, '.format(t)
        format_string += '\b\b)'
        return format_string


class Transform(object):
    """Abstract base class for transforms."""

    def __call__(self, *args):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class BlockShuffle(Transform):
    """Shuffle blocks of contiguous discrete states within each trial."""

    def __init__(self, rng_seed):
        """

        Parameters
        ----------
        rng_seed : int
            to control random number generator

        """
        self.rng_seed = rng_seed

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample : np.ndarray
            dense representation of shape (time)

        Returns
        -------
        np.ndarray
            output shape is (time)

        """

        np.random.seed(self.rng_seed)
        n_time = len(sample)
        if not any(np.isnan(sample)):
            # mark first time point of state change with a nonzero number
            state_change = np.where(np.concatenate([[0], np.diff(sample)], axis=0) != 0)[0]
            # collect runs
            runs = []
            prev_beg = 0
            for curr_beg in state_change:
                runs.append(np.arange(prev_beg, curr_beg))
                prev_beg = curr_beg
            runs.append(np.arange(prev_beg, n_time))
            # shuffle runs
            rand_perm = np.random.permutation(len(runs))
            runs_shuff = [runs[idx] for idx in rand_perm]
            # index back into original labels with shuffled indices
            sample_shuff = sample[np.concatenate(runs_shuff)]
        else:
            sample_shuff = np.full(n_time, fill_value=np.nan)
        return sample_shuff

    def __repr__(self):
        return str('BlockShuffle(rng_seed=%i)' % self.rng_seed)


class MakeOneHot(Transform):
    """Turn a categorical vector into a one-hot vector."""

    def __init__(self):
        pass

    def __call__(self, sample):
        """Assumes that K classes are identified by the numbers 0:K-1.

        Parameters
        ----------
        sample: p.ndarray
            input shape is (time)

        Returns
        -------
        np.ndarray
            output shape is (time, K)

        """
        if len(sample.shape) == 2:  # weak test for if sample is already onehot
            onehot = sample
        else:
            n_time = len(sample)
            n_classes = int(np.nanmax(sample))
            onehot = np.zeros((n_time, n_classes + 1))
            if not any(np.isnan(sample)):
                onehot[np.arange(n_time), sample.astype('int')] = 1
            else:
                onehot[:] = np.nan

        return onehot

    def __repr__(self):
        return 'MakeOneHot()'


class MotionEnergy(Transform):
    """Compute motion energy across batch dimension."""

    def __init__(self):
        pass

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample : np.ndarray
            input shape is (time, n_channels)

        Returns
        -------
        np.ndarray
            output shape is (time, n_channels)

        """
        return np.vstack([np.zeros((1, sample.shape[1])), np.abs(np.diff(sample, axis=0))])

    def __repr__(self):
        return 'MotionEnergy()'


class Unitize(Transform):
    """Place each channel (mostly) in [0, 1]."""

    def __init__(self):
        self.mins = None
        self.maxs = None

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample : np.ndarray
            input shape is (time, n_channels)

        Returns
        -------
        np.ndarray
            output shape is (time, n_channels)

        """
        self.mins = np.quantile(sample, 0.05, axis=0)
        self.maxs = np.quantile(sample, 0.95, axis=0)
        sample = (sample - self.mins) / (self.maxs - self.mins)
        return sample

    def __repr__(self):
        return 'Unitize()'


class ZScore(Transform):
    """z-score channel activity."""

    def __init__(self):
        pass

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample : np.ndarray
            input shape is (time, n_channels)

        Returns
        -------
        np.ndarray
            output shape is (time, n_channels)

        """
        sample -= np.mean(sample, axis=0)
        std = np.std(sample, axis=0)
        sample[:, std > 0] = (sample[:, std > 0] / std[std > 0])
        return sample

    def __repr__(self):
        return 'ZScore()'

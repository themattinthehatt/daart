"""Classes for splitting and serving data to models.

The data generator classes contained in this module inherit from the
:class:`torch.utils.data.Dataset` class. The user-facing class is the
:class:`DataGenerator`, which can manage one or more datasets. Each dataset is composed
of trials, which are split into training, validation, and testing trials using the
:func:`split_trials`. The default data generator can handle the following data types:

* **markers**: i.e. DLC/DGP markers
* **labels**: discrete behavioral labels
* **soft_labels**: noisy discrete behavioral labels

"""

from collections import OrderedDict
import h5py
import numpy as np
import os
import pandas as pd
import pickle
import torch
from torch.utils import data
from torch.utils.data import SubsetRandomSampler


__all__ = ['split_trials', 'compute_batches', 'SingleDataset', 'DataGenerator']


def split_trials(n_trials, rng_seed=0, train_tr=8, val_tr=1, test_tr=1, gap_tr=0):
    """Split trials into train/val/test blocks.

    The data is split into blocks that have gap trials between tr/val/test:

    `train tr | gap tr | val tr | gap tr | test tr | gap tr`

    Parameters
    ----------
    n_trials : int
        total number of trials to be split
    rng_seed : int, optional
        random seed for reproducibility
    train_tr : int, optional
        number of train trials per block
    val_tr : int, optional
        number of validation trials per block
    test_tr : int, optional
        number of test trials per block
    gap_tr : int, optional
        number of gap trials between tr/val/test; there will be a total of 3 * `gap_tr` gap trials
        per block; can be zero if no gap trials are desired.

    Returns
    -------
    dict
        Split trial indices are stored in a dict with keys `train`, `test`, and `val`

    """

    # same random seed for reproducibility
    np.random.seed(rng_seed)

    tr_per_block = train_tr + gap_tr + val_tr + gap_tr + test_tr + gap_tr

    n_blocks = int(np.floor(n_trials / tr_per_block))
    if n_blocks == 0:
        raise ValueError(
            'Not enough trials (n=%i) for the train/test/val/gap values %i/%i/%i/%i' %
            (n_trials, train_tr, val_tr, test_tr, gap_tr))

    leftover_trials = n_trials - tr_per_block * n_blocks
    idxs_block = np.random.permutation(n_blocks)

    batch_idxs = {'train': [], 'test': [], 'val': []}
    for block in idxs_block:

        curr_tr = block * tr_per_block
        batch_idxs['train'].append(np.arange(curr_tr, curr_tr + train_tr))
        curr_tr += (train_tr + gap_tr)
        batch_idxs['val'].append(np.arange(curr_tr, curr_tr + val_tr))
        curr_tr += (val_tr + gap_tr)
        batch_idxs['test'].append(np.arange(curr_tr, curr_tr + test_tr))

    # add leftover trials to train data
    if leftover_trials > 0:
        batch_idxs['train'].append(np.arange(tr_per_block * n_blocks, n_trials))

    for dtype in ['train', 'val', 'test']:
        batch_idxs[dtype] = np.concatenate(batch_idxs[dtype], axis=0)

    return batch_idxs


def compute_batches(data, batch_size, batch_pad=0):
    """Compute batches of temporally contiguous data points.

    Partial batches are not constructed; for example, if the number of time points is 24, and the
    batch size is 10, only the first 20 points will be returned (in two batches).

    Parameters
    ----------
    data : array-like
        data to batch, of shape (T, N) or (T,)
    batch_size : int
        number of continguous values along dimension 0 to include per batch
    batch_pad : int, optional
        if >0, add `batch_pad` time points to the beginning/end of each batch (to account for
        padding with convolution layers)

    Returns
    -------
    list
        batched data

    """

    if isinstance(data, list):
        # assume data has already been batched
        return data

    if len(data.shape) == 2:
        batch_dims = (batch_size + 2 * batch_pad, data.shape[1])
    else:
        batch_dims = (batch_size + 2 * batch_pad,)

    n_batches = int(np.floor(data.shape[0] / batch_size))
    batched_data = [np.zeros(batch_dims) for _ in range(n_batches)]
    for b in range(n_batches):
        idx_beg = b * batch_size
        idx_end = (b + 1) * batch_size
        if batch_pad > 0:
            if idx_beg == 0:
                # initial vals are zeros; rest are real data
                batched_data[b][batch_pad:] = data[idx_beg:idx_end + batch_pad]
            elif (idx_end + batch_pad) > data.shape[0]:
                batched_data[b][:-batch_pad] = data[idx_beg - batch_pad:idx_end]
            else:
                batched_data[b] = data[idx_beg - batch_pad:idx_end + batch_pad]
        else:
            batched_data[b] = data[idx_beg:idx_end]

    return batched_data


class SingleDataset(data.Dataset):
    """Dataset class for a single dataset."""

    def __init__(
            self, id, signals=None, transforms=None, paths=None, device='cuda', as_numpy=False,
            batch_size=100, batch_pad=0):
        """

        Parameters
        ----------
        id : str
            dataset id
        signals : list of strs
            e.g. 'markers' | 'labels' | ....
        transforms : list of transform objects
            each element corresponds to an entry in signals; for multiple transforms, chain
            together using :class:`daart.transforms.Compose` class. See
            :mod:`daart.transforms` for available transform options.
        paths : list of strs
            each element corresponds to an entry in `signals`; filename (using absolute path) of
            data
        device : str, optional
            location of data; options are
            'cpu' | 'cuda'
        batch_size : int, optional
            number of contiguous data points in each batch
        batch_pad : int, optional
            if >0, add `batch_pad` time points to the beginning/end of each batch (to account for
            padding with convolution layers)

        """

        # specify data
        self.id = id

        # get data paths
        self.signals = signals
        self.transforms = OrderedDict()
        self.paths = OrderedDict()
        self.dtypes = OrderedDict()
        self.data = OrderedDict()
        for signal, transform, path in zip(signals, transforms, paths):
            self.transforms[signal] = transform
            self.paths[signal] = path
            self.dtypes[signal] = None  # update when loading data

        self.batch_pad = batch_pad
        self.load_data(batch_size)
        self.n_trials = len(self.data[signals[0]])

        # meta data about train/test/xv splits; set by DataGenerator
        self.batch_idxs = None
        self.n_batches = None

        self.device = device
        self.as_numpy = as_numpy

    def __str__(self):
        """Pretty printing of dataset info"""
        format_str = str('%s\n' % self.id)
        format_str += str('    signals: {}\n'.format(self.signals))
        format_str += str('    transforms: {}\n'.format(self.transforms))
        format_str += str('    paths: {}\n'.format(self.paths))
        return format_str

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        """Return batch of data.

        Parameters
        ----------
        idx : int or NoneType
            trial index to load; if `NoneType`, return all data.

        Returns
        -------
        dict
            data sample

        """

        sample = OrderedDict()
        for signal in self.signals:

            # collect signal
            if idx is None:
                sample[signal] = [d for d in self.data[signal]]
            else:
                sample[signal] = [self.data[signal][idx]]

            # transform into tensor
            if not self.as_numpy:
                if self.dtypes[signal] == 'float32':
                    sample[signal] = torch.from_numpy(sample[signal][0]).float()
                else:
                    sample[signal] = torch.from_numpy(sample[signal][0]).long()

        # add batch index
        sample['batch_idx'] = idx

        return sample

    def load_data(self, batch_size):
        """Load all data into memory.

        Parameters
        ----------
        batch_size : int
            batch size of data

        Returns
        -------
        dict
            data samples

        """

        allowed_signals = ['markers', 'labels_strong', 'labels_weak']

        for signal in self.signals:

            if signal == 'markers':

                file_ext = self.paths[signal].split('.')[-1]

                if file_ext == 'csv':
                    # assume dlc/dgp format
                    from numpy import genfromtxt
                    dlc = genfromtxt(
                        self.paths[signal], delimiter=',', dtype=None, encoding=None)
                    dlc = dlc[3:, 1:].astype('float')  # get rid of headers, etc.
                    x = dlc[:, 0::3]
                    y = dlc[:, 1::3]
                    data_curr = np.hstack([x, y])
                elif file_ext == 'h5':
                    # assume dlc/dgp format
                    with h5py.File(self.paths[signal], 'r') as f:
                        t = f['df_with_missing']['table'][()]
                    dlc = np.concatenate([t[i][1][None, :] for i in range(len(t))])
                    x = dlc[:, 0::3]
                    y = dlc[:, 1::3]
                    data_curr = np.hstack([x, y])
                elif file_ext == 'npy':
                    # assume single array
                    data_curr = np.load(self.paths[signal])
                else:
                    raise ValueError('"%s" is an invalid file extension' % file_ext)

                # l = dlc[:, 2::3]
                self.dtypes[signal] = 'float32'

            elif signal == 'labels_strong':

                # assume csv output from deepethogram labeler
                labels = np.genfromtxt(
                    self.paths[signal], delimiter=',', dtype=np.int, encoding=None,
                    skip_header=1)
                data_curr = np.argmax(labels[:, 1:], axis=1)  # get rid of index column
                self.dtypes[signal] = 'int32'

            elif signal == 'labels_weak':

                file_ext = self.paths[signal].split('.')[-1]

                if file_ext == 'csv':
                    # assume same format as strong label csv files
                    labels = np.genfromtxt(
                        self.paths[signal], delimiter=',', dtype=np.int, encoding=None,
                        skip_header=1)
                    data_curr = np.argmax(labels[:, 1:], axis=1)  # get rid of index column
                elif file_ext == 'pkl':
                    # assume particular pkl format; already in dense representation
                    with open(self.paths[signal], 'rb') as f:
                        data_curr = pickle.load(f)['states']

                self.dtypes[signal] = 'int32'

            else:
                raise ValueError(
                    '"{}" is an invalid signal type; must choose from {}'.format(
                        signal, allowed_signals))

            # apply transforms to ALL data
            if self.transforms[signal]:
                data_curr = self.transforms[signal](data_curr)

            # compute batches of temporally contiguous data points
            data_curr = compute_batches(data_curr, batch_size, self.batch_pad)

            self.data[signal] = data_curr

        return None


class DataGenerator(object):
    """Dataset generator for serving pytorch models.

    This class contains a list of SingleDataset generators. It handles shuffling and iterating
    over these datasets.
    """

    _dtypes = {'train', 'val', 'test'}

    def __init__(
            self, ids_list, signals_list=None, transforms_list=None, paths_list=None,
            device='cuda', as_numpy=False, rng_seed=0, trial_splits=None, train_frac=1.0,
            batch_size=100, num_workers=0, pin_memory=False, batch_pad=0):
        """

        Parameters
        ----------
        ids_list : list of strs
            unique identifier for each dataset
        signals_list : list of lists
            list of signals for each dataset
        transforms_list : list of lists
            list of transforms for each dataset
        paths_list : list of lists
            list of paths for each dataset
        device : str, optional
            location of data; options are 'cpu' | 'cuda'
        as_numpy : bool, optional
            if True return data as a numpy array, else return as a torch tensor
        rng_seed : int, optional
            controls split of train/val/test trials
        trial_splits : dict, optional
            determines number of train/val/test trials using the keys 'train_tr', 'val_tr',
            'test_tr', and 'gap_tr'; see :func:`split_trials` for how these are used.
        train_frac : float, optional
            if `0 < train_frac < 1.0`, defines the fraction of assigned training trials to
            actually use; if `train_frac > 1.0`, defines the number of assigned training trials to
            actually use
        batch_size : int, optional
            number of contiguous data points in each batch
        num_workers : int, optional
            number of cpu cores per dataset; defaults to 0 (all data loaded in main process)
        pin_memory : bool, optional
            if True, the data loader automatically pulls fetched data Tensors in pinned memory, and
            thus enables faster transfer to CUDA-enabled GPUs
        batch_pad : int, optional
            if >0, add `batch_pad` time points to the beginning/end of each batch (to account for
            padding with convolution layers)

        """
        self.ids = ids_list
        self.as_numpy = as_numpy
        self.device = device

        self.datasets = []
        self.signals = signals_list
        self.transforms = transforms_list
        self.paths = paths_list
        for id, signals, transforms, paths in zip(
                ids_list, signals_list, transforms_list, paths_list):
            self.datasets.append(SingleDataset(
                id=id, signals=signals, transforms=transforms, paths=paths, device=device,
                as_numpy=self.as_numpy, batch_size=batch_size, batch_pad=batch_pad))

        # collect info about datasets
        self.n_datasets = len(self.datasets)

        # get train/val/test batch indices for each dataset
        if trial_splits is None:
            trial_splits = {'train_tr': 8, 'val_tr': 1, 'test_tr': 1, 'gap_tr': 0}
        elif isinstance(trial_splits, str):
            ttypes = ['train_tr', 'val_tr', 'test_tr', 'gap_tr']
            trial_splits = {
                ttype: s for ttype, s in zip(ttypes, [int(s) for s in trial_splits.split(';')])}
        else:
            pass
        self.batch_ratios = [None] * self.n_datasets
        for i, dataset in enumerate(self.datasets):
            dataset.batch_idxs = split_trials(len(dataset), rng_seed=rng_seed, **trial_splits)
            dataset.n_batches = {}
            for dtype in self._dtypes:
                if dtype == 'train':
                    # subsample training data if requested
                    if train_frac != 1.0:
                        n_batches = len(dataset.batch_idxs[dtype])
                        if train_frac < 1.0:
                            # subsample as fraction of total batches
                            n_idxs = int(np.floor(train_frac * n_batches))
                            if n_idxs <= 0:
                                print(
                                    'warning: attempting to use invalid number of training ' +
                                    'batches; defaulting to all training batches')
                                n_idxs = n_batches
                        else:
                            # subsample fixed number of batches
                            train_frac = n_batches if train_frac > n_batches else train_frac
                            n_idxs = int(train_frac)
                        idxs_rand = np.random.choice(n_batches, size=n_idxs, replace=False)
                        dataset.batch_idxs[dtype] = dataset.batch_idxs[dtype][idxs_rand]
                    self.batch_ratios[i] = len(dataset.batch_idxs[dtype])
                dataset.n_batches[dtype] = len(dataset.batch_idxs[dtype])
        self.batch_ratios = np.array(self.batch_ratios) / np.sum(self.batch_ratios)

        # find total number of batches per data type; this will be iterated over in the train loop
        self.n_tot_batches = {}
        for dtype in self._dtypes:
            self.n_tot_batches[dtype] = np.sum(
                [dataset.n_batches[dtype] for dataset in self.datasets])

        # create data loaders (will shuffle/batch/etc datasets)
        self.dataset_loaders = [None] * self.n_datasets
        for i, dataset in enumerate(self.datasets):
            self.dataset_loaders[i] = {}
            for dtype in self._dtypes:
                self.dataset_loaders[i][dtype] = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=1,
                    sampler=SubsetRandomSampler(dataset.batch_idxs[dtype]),
                    num_workers=num_workers,
                    pin_memory=pin_memory)

        # create all iterators (will iterate through data loaders)
        self.dataset_iters = [None] * self.n_datasets
        for i in range(self.n_datasets):
            self.dataset_iters[i] = {}
            for dtype in self._dtypes:
                self.dataset_iters[i][dtype] = iter(self.dataset_loaders[i][dtype])

    def __str__(self):
        """Pretty printing of dataset info"""
        format_str = str('Generator contains %i SingleDataset objects:\n' % self.n_datasets)
        for dataset in self.datasets:
            format_str += dataset.__str__()
        return format_str

    def __len__(self):
        return self.n_datasets

    def reset_iterators(self, dtype):
        """Reset iterators so that all data is available.

        Parameters
        ----------
        dtype : str
            'train' | 'val' | 'test' | 'all'

        """

        for i in range(self.n_datasets):
            if dtype == 'all':
                for dtype_ in self._dtypes:
                    self.dataset_iters[i][dtype_] = iter(self.dataset_loaders[i][dtype_])
            else:
                self.dataset_iters[i][dtype] = iter(self.dataset_loaders[i][dtype])

    def next_batch(self, dtype):
        """Return next batch of data.

        The data generator iterates randomly through datasets and trials. Once a dataset runs out
        of trials it is skipped.

        Parameters
        ----------
        dtype : str
            'train' | 'val' | 'test'

        Returns
        -------
        tuple
            - sample (dict): data batch with keys given by `signals` input to class
            - dataset (int): dataset from which data batch is drawn

        """
        while True:
            # get next dataset
            dataset = np.random.choice(np.arange(self.n_datasets), p=self.batch_ratios)

            # get this dataset data
            try:
                sample = next(self.dataset_iters[dataset][dtype])
                break
            except StopIteration:
                continue

        if self.as_numpy:
            for i, signal in enumerate(sample):
                if signal != 'batch_idx':
                    sample[signal] = [ss.cpu().detach().numpy() for ss in sample[signal]]
        else:
            if self.device == 'cuda':
                sample = {key: val.to('cuda') for key, val in sample.items()}

        return sample, dataset

"""Classes for splitting and serving data to models.

The data generator classes contained in this module inherit from the
:class:`torch.utils.data.Dataset` class. The user-facing class is the
:class:`DataGenerator`, which can manage one or more datasets. Each dataset is composed
of trials, which are split into training, validation, and testing trials using the
:func:`split_trials`. The default data generator can handle the following data types:

* **markers**: i.e. DLC/DGP markers
* **labels_strong**: discrete behavioral labels
* **labels_weak**: noisy discrete behavioral labels

"""

from collections import OrderedDict
import h5py
import logging
import numpy as np
import os
import pandas as pd
import pickle
import torch
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
from typing import List, Optional, Union
from typeguard import typechecked


__all__ = [
    'split_trials', 'compute_sequences', 'compute_sequence_pad', 'SingleDataset', 'DataGenerator',
    'load_marker_csv', 'load_feature_csv', 'load_marker_h5', 'load_label_csv', 'load_label_pkl',
]


@typechecked
def split_trials(
        n_trials: int,
        rng_seed: int = 0,
        train_tr: int = 8,
        val_tr: int = 1,
        test_tr: int = 1,
        gap_tr: int = 0
) -> dict:
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


@typechecked
def compute_sequences(
        data: Union[np.ndarray, list],
        sequence_length: int,
        sequence_pad: int = 0
) -> list:
    """Compute sequences of temporally contiguous data points.

    Partial sequences are not constructed; for example, if the number of time points is 24, and the
    batch size is 10, only the first 20 points will be returned (in two batches).

    Parameters
    ----------
    data : array-like or list
        data to batch, of shape (T, N) or (T,)
    sequence_length : int
        number of continguous values along dimension 0 to include per batch
    sequence_pad : int, optional
        if >0, add `sequence_pad` time points to the beginning/end of each sequence (to account for
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
        batch_dims = (sequence_length + 2 * sequence_pad, data.shape[1])
    else:
        batch_dims = (sequence_length + 2 * sequence_pad,)

    n_batches = int(np.floor(data.shape[0] / sequence_length))
    batched_data = [np.zeros(batch_dims) for _ in range(n_batches)]
    for b in range(n_batches):
        idx_beg = b * sequence_length
        idx_end = (b + 1) * sequence_length
        if sequence_pad > 0:
            if idx_beg == 0:
                # initial vals are zeros; rest are real data
                batched_data[b][sequence_pad:] = data[idx_beg:idx_end + sequence_pad]
            elif (idx_end + sequence_pad) > data.shape[0]:
                batched_data[b][:-sequence_pad] = data[idx_beg - sequence_pad:idx_end]
            else:
                batched_data[b] = data[idx_beg - sequence_pad:idx_end + sequence_pad]
        else:
            batched_data[b] = data[idx_beg:idx_end]

    return batched_data


@typechecked
def compute_sequence_pad(hparams: dict) -> int:
    """Compute padding needed to account for convolutions.

    Parameters
    ----------
    hparams : dict
        contains model architecture type and hyperparameter info (lags, n_hidden_layers, etc)

    Returns
    -------
    int
        amount of padding that needs to be added to beginning/end of each batch

    """

    if hparams['backbone'].lower() == 'temporal-mlp':
        pad = hparams['n_lags']
    elif hparams['backbone'].lower() == 'tcn':
        pad = (2 ** hparams['n_hid_layers']) * hparams['n_lags']
    elif hparams['backbone'].lower() == 'dtcn':
        # dilattion of each dilation block is 2 ** layer_num
        # 2 conv layers per dilation block
        pad = sum([2 * (2 ** n) * hparams['n_lags'] for n in range(hparams['n_hid_layers'])])
    elif hparams['backbone'].lower() in ['lstm', 'gru']:
        # give some warmup timesteps
        pad = 4
    elif hparams['backbone'].lower() == 'tgm':
        raise NotImplementedError
    elif hparam['backbone'].lower() == 'random-forest':
        pad = 0
    else:
        raise ValueError('"%s" is not a valid backbone network' % hparams['backbone'])

    return pad


class SingleDataset(data.Dataset):
    """Dataset class for a single dataset."""

    @typechecked
    def __init__(
            self,
            id: str,
            signals: List[str],
            transforms: list,
            paths: List[Union[str, None]],
            device: str = 'cuda',
            as_numpy: bool = False,
            sequence_length: int = 500,
            sequence_pad: int = 0,
            input_type: str = 'markers'
    ) -> None:
        """

        Parameters
        ----------
        id : str
            dataset id
        signals : list of strs
            e.g. 'markers' | 'labels_strong' | 'tasks' | ....
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
        sequence_length : int, optional
            number of contiguous data points in a sequence
        sequence_pad : int, optional
            if >0, add `sequence_pad` time points to the beginning/end of each sequence (to account
            for padding with convolution layers)
        input_type : str, optional
            'markers' | 'features'

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

        self.sequence_pad = sequence_pad
        self.sequence_length = sequence_length
        self.load_data(sequence_length, input_type)
        self.n_sequences = len(self.data[signals[0]])

        # meta data about train/test/xv splits; set by DataGenerator
        self.batch_idxs = None
        self.n_batches = None

        self.device = device
        self.as_numpy = as_numpy

    @typechecked
    def __str__(self) -> str:
        """Pretty printing of dataset info"""
        format_str = str('%s\n' % self.id)
        format_str += str('    signals: {}\n'.format(self.signals))
        format_str += str('    transforms: {}\n'.format(self.transforms))
        format_str += str('    paths: {}\n'.format(self.paths))
        return format_str

    @typechecked
    def __len__(self) -> int:
        return self.n_sequences

    @typechecked
    def __getitem__(self, idx: Union[int, np.int64, None]) -> dict:
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

    @typechecked
    def load_data(self, sequence_length: int, input_type: str) -> None:
        """Load all data into memory.

        Parameters
        ----------
        sequence_length : int
            number of contiguous data points in a sequence
        input_type : str
            'markers' | 'features'

        """

        allowed_signals = ['markers', 'labels_strong', 'labels_weak', 'tasks']

        for signal in self.signals:

            if signal == 'markers':

                file_ext = self.paths[signal].split('.')[-1]

                if file_ext == 'csv':
                    if input_type == 'markers':
                        xs, ys, ls, marker_names = load_marker_csv(self.paths[signal])
                        data_curr = np.hstack([xs, ys])
                    else:
                        vals, feature_names = load_feature_csv(self.paths[signal])
                        data_curr = vals

                elif file_ext == 'h5':
                    if input_type != 'markers':
                        raise NotImplementedError
                    xs, ys, ls, marker_names = load_marker_h5(self.paths[signal])
                    data_curr = np.hstack([xs, ys])

                elif file_ext == 'npy':
                    # assume single array
                    data_curr = np.load(self.paths[signal])

                else:
                    raise ValueError('"%s" is an invalid file extension' % file_ext)

                self.dtypes[signal] = 'float32'

            elif signal == 'tasks':

                file_ext = self.paths[signal].split('.')[-1]
                if file_ext == 'csv':
                    vals, feature_names = load_feature_csv(self.paths[signal])
                    data_curr = vals

                else:
                    raise ValueError('"%s" is an invalid file extension' % file_ext)

                self.dtypes[signal] = 'float32'

            elif signal == 'labels_strong':

                if (self.paths[signal] is None) or not os.path.exists(self.paths[signal]):
                    # if no path given, assume same size as markers and set all to background
                    if 'markers' in self.data.keys():
                        data_curr = np.zeros(
                            (len(self.data['markers']) * sequence_length,), dtype=np.int)
                    else:
                        raise FileNotFoundError(
                            'Could not load "labels_strong" from None file without markers')
                else:
                    labels, label_names = load_label_csv(self.paths[signal])
                    data_curr = np.argmax(labels, axis=1)

                self.dtypes[signal] = 'int32'

            elif signal == 'labels_weak':

                file_ext = self.paths[signal].split('.')[-1]

                if file_ext == 'csv':
                    labels, label_names = load_label_csv(self.paths[signal])
                    data_curr = np.argmax(labels, axis=1)

                elif file_ext == 'pkl':
                    labels, label_names = load_label_pkl(self.paths[signal])
                    data_curr = labels

                self.dtypes[signal] = 'int32'

            else:
                raise ValueError(
                    '"{}" is an invalid signal type; must choose from {}'.format(
                        signal, allowed_signals))

            # apply transforms to ALL data
            if self.transforms[signal]:
                data_curr = self.transforms[signal](data_curr)

            # compute batches of temporally contiguous data points
            data_curr = compute_sequences(data_curr, sequence_length, self.sequence_pad)

            self.data[signal] = data_curr


class DataGenerator(object):
    """Dataset generator for serving pytorch models.

    This class contains a list of SingleDataset generators. It handles shuffling and iterating
    over these datasets.
    """

    _dtypes = {'train', 'val', 'test'}

    @typechecked
    def __init__(
            self,
            ids_list: List[str],
            signals_list: List[List[str]],
            transforms_list: List[list],
            paths_list: List[List[Union[str, None]]],
            device: str = 'cuda',
            as_numpy: bool = False,
            rng_seed: int = 0,
            trial_splits: Union[str, dict, None] = None,
            train_frac: float = 1.0,
            sequence_length: int = 500,
            batch_size: int = 1,
            num_workers: int = 0,
            pin_memory: bool = False,
            sequence_pad: int = 0,
            input_type: str = 'markers'
    ) -> None:
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
        sequence_length : int, optional
            number of contiguous data points in a sequence
        batch_size : int, optional
            number of sequences in each batch
        num_workers : int, optional
            number of cpu cores per dataset; defaults to 0 (all data loaded in main process)
        pin_memory : bool, optional
            if True, the data loader automatically pulls fetched data Tensors in pinned memory, and
            thus enables faster transfer to CUDA-enabled GPUs
        sequence_pad : int, optional
            if >0, add `sequence_pad` time points to the beginning/end of each sequence (to account
            for padding with convolution layers)
        input_type : str, optional
            'markers' | 'features'

        """
        self.ids = ids_list
        self.batch_size = batch_size
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
                as_numpy=self.as_numpy, sequence_length=sequence_length,
                sequence_pad=sequence_pad, input_type=input_type))

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
                                print_str = (
                                    f'warning: attempting to use invalid number of training '
                                    f'batches; defaulting to all training batches'
                                )
                                logging.info(print_str)
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
        # automatically set val/test batch sizes to 1 for more fine-grained logging
        self.n_tot_batches = {}
        for dtype in self._dtypes:
            if dtype == 'train':
                self.n_tot_batches[dtype] = int(np.ceil(np.sum(
                    [dataset.n_batches[dtype] for dataset in self.datasets]) / self.batch_size))
            else:
                self.n_tot_batches[dtype] = np.sum(
                    [dataset.n_batches[dtype] for dataset in self.datasets])

        # create data loaders (will shuffle/batch/etc datasets)
        self.dataset_loaders = [None] * self.n_datasets
        for i, dataset in enumerate(self.datasets):
            self.dataset_loaders[i] = {}
            for dtype in self._dtypes:
                self.dataset_loaders[i][dtype] = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=1,  # keep 1 here so we can combine batches from multiple datasets
                    sampler=SubsetRandomSampler(dataset.batch_idxs[dtype]),
                    num_workers=num_workers,
                    pin_memory=pin_memory)

        # create all iterators (will iterate through data loaders)
        self.dataset_iters = [None] * self.n_datasets
        for i in range(self.n_datasets):
            self.dataset_iters[i] = {}
            for dtype in self._dtypes:
                self.dataset_iters[i][dtype] = iter(self.dataset_loaders[i][dtype])

    @typechecked
    def __str__(self) -> str:
        """Pretty printing of dataset info"""
        format_str = str('Generator contains %i SingleDataset objects:\n' % self.n_datasets)
        for dataset in self.datasets:
            format_str += dataset.__str__()
        return format_str

    @typechecked
    def __len__(self) -> int:
        return self.n_datasets

    @typechecked
    def reset_iterators(self, dtype: str) -> None:
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

    @typechecked
    def next_batch(self, dtype: str) -> tuple:
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
        empty_datasets = np.zeros(self.n_datasets)

        # automatically set val/test batch sizes to 1 for more fine-grained logging
        n_batches = self.batch_size if dtype == 'train' else 1

        n_sequences = 0
        sequences = []
        datasets = []

        while True:

            # get next dataset
            dataset = np.random.choice(np.arange(self.n_datasets), p=self.batch_ratios)

            # get sequence from this dataset
            try:
                sequence = next(self.dataset_iters[dataset][dtype])
                # add sequence to batch
                sequences.append(sequence)
                datasets.append(dataset)
                n_sequences += 1
                # exit loop if we have enough batches
                if n_sequences == n_batches:
                    break
            except StopIteration:
                # record dataset as being empty
                empty_datasets[dataset] = 1
                # leave loop if all datasets are empty; otherwise, continue collecting sequences
                if np.all(empty_datasets):
                    break
                else:
                    continue

        batch = OrderedDict()
        if self.as_numpy:
            for i, signal in enumerate(sequences[0]):
                if signal != 'batch_idx':
                    batch[signal] = np.row_stack(
                        [s[signal].cpu().detach().numpy() for s in sequences])
                else:
                    batch['batch_idx'] = [ss['batch_idx'] for ss in sequences]
        else:
            for i, signal in enumerate(sequences[0]):
                if signal != 'batch_idx':
                    batch[signal] = torch.vstack([s[signal] for s in sequences])
                else:
                    batch['batch_idx'] = torch.vstack([s['batch_idx'] for s in sequences])

            if self.device == 'cuda':
                batch = {key: val.to('cuda') for key, val in batch.items()}

        return batch, datasets


@typechecked
def load_marker_csv(filepath: str) -> tuple:
    """Load markers from csv file assuming DLC format.

    --------------------------------------------------------------------------------
       scorer  | <scorer_name> | <scorer_name> | <scorer_name> | <scorer_name> | ...
     bodyparts |  <part_name>  |  <part_name>  |  <part_name>  |  <part_name>  | ...
       coords  |       x       |       y       |  likelihood   |       x       | ...
    --------------------------------------------------------------------------------
         0     |     34.5      |     125.4     |     0.921     |      98.4     | ...
         .     |       .       |       .       |       .       |       .       | ...
         .     |       .       |       .       |       .       |       .       | ...
         .     |       .       |       .       |       .       |       .       | ...

    Parameters
    ----------
    filepath : str
        absolute path of csv file

    Returns
    -------
    tuple
        - x coordinates (np.ndarray): shape (n_t, n_bodyparts)
        - y coordinates (np.ndarray): shape (n_t, n_bodyparts)
        - likelihoods (np.ndarray): shape (n_t,)
        - marker names (list): name for each column of `x` and `y` matrices

    """
    # data = np.genfromtxt(filepath, delimiter=',', dtype=None, encoding=None)
    # marker_names = list(data[1, 1::3])
    # markers = data[3:, 1:].astype('float')  # get rid of headers, etc.

    # define first three rows as headers (as per DLC standard)
    # drop first column ('scorer' at level 0) which just contains frame indices
    df = pd.read_csv(filepath, header=[0, 1, 2]).drop(['scorer'], axis=1, level=0)
    # collect marker names from multiindex header
    marker_names = [c[1] for c in df.columns[::3]]
    markers = df.values
    xs = markers[:, 0::3]
    ys = markers[:, 1::3]
    ls = markers[:, 2::3]
    return xs, ys, ls, marker_names


@typechecked
def load_feature_csv(filepath: str) -> tuple:
    """Load markers from csv file assuming the following format.

    --------------------------------------------------------------------------------
        name   |     <f1>      |     <f2>      |     <f3>      |     <f4>      | ...
    --------------------------------------------------------------------------------
         0     |     34.5      |     125.4     |     0.921     |      98.4     | ...
         .     |       .       |       .       |       .       |       .       | ...
         .     |       .       |       .       |       .       |       .       | ...
         .     |       .       |       .       |       .       |       .       | ...

    Parameters
    ----------
    filepath : str
        absolute path of csv file

    Returns
    -------
    tuple
        - x coordinates (np.ndarray): shape (n_t, n_bodyparts)
        - y coordinates (np.ndarray): shape (n_t, n_bodyparts)
        - likelihoods (np.ndarray): shape (n_t,)
        - marker names (list): name for each column of `x` and `y` matrices

    """
    df = pd.read_csv(filepath)
    # drop first column if it just contains frame indices
    unnamed_col = 'Unnamed: 0'
    if unnamed_col in list(df.columns):
        df = df.drop([unnamed_col], axis=1)
    vals = df.values
    feature_names = list(df.columns)
    return vals, feature_names


@typechecked
def load_marker_h5(filepath: str) -> tuple:
    """Load markers from hdf5 file assuming DLC format.

    Parameters
    ----------
    filepath : str
        absolute path of hdf5 file

    Returns
    -------
    tuple
        - x coordinates (np.ndarray): shape (n_t, n_bodyparts)
        - y coordinates (np.ndarray): shape (n_t, n_bodyparts)
        - likelihoods (np.ndarray): shape (n_t,)
        - marker names (list): name for each column of `x` and `y` matrices

    """
    df = pd.read_hdf(filepath)
    marker_names = [d[1] for d in df.columns][0::3]
    markers = df.to_numpy()
    xs = markers[:, 0::3]
    ys = markers[:, 1::3]
    ls = markers[:, 2::3]
    return xs, ys, ls, marker_names


@typechecked
def load_label_csv(filepath: str) -> tuple:
    """Load labels from csv file assuming a standard format.

    --------------------------------
       | <class 0> | <class 1> | ...
    --------------------------------
     0 |     0     |     1     | ...
     1 |     0     |     1     | ...
     . |     .     |     .     | ...
     . |     .     |     .     | ...
     . |     .     |     .     | ...

    Parameters
    ----------
    filepath : str
        absolute path of csv file

    Returns
    -------
    tuple
        - labels (np.ndarray): shape (n_t, n_labels)
        - label names (list): name for each column in `labels` matrix

    """
    labels = np.genfromtxt(
        filepath, delimiter=',', dtype=np.int, encoding=None, skip_header=1)[:, 1:]
    label_names = list(
        np.genfromtxt(filepath, delimiter=',', dtype=None, encoding=None, max_rows=1)[1:])
    return labels, label_names


@typechecked
def load_label_pkl(filepath: str) -> tuple:
    """Load labels from pkl file assuming a standard format.

    Parameters
    ----------
    filepath : str
        absolute path of pickle file

    Returns
    -------
    tuple
        - labels (np.ndarray): shape (n_t, n_labels)
        - label names (list): name for each column in `labels` matrix

    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    labels = data['states']
    try:
        label_dict = data['state_mapping']
    except KeyError:
        label_dict = data['state_labels']
    label_names = [label_dict[i] for i in range(len(label_dict))]
    return labels, label_names

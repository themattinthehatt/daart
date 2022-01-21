import copy
import numpy as np
import pytest
import torch

from daart.data import split_trials, compute_sequences
from daart.utils import build_data_generator


def test_split_trials():

    # return correct number of trials 1
    splits = split_trials(100, 0, 8, 1, 1, 0)
    assert len(splits['train']) == 80
    assert len(splits['val']) == 10
    assert len(splits['test']) == 10

    # return correct number of trials 2
    splits = split_trials(10, 0, 8, 1, 1, 0)
    assert len(splits['train']) == 8
    assert len(splits['val']) == 1
    assert len(splits['test']) == 1

    # return correct number of trials 3
    splits = split_trials(11, 0, 8, 1, 1, 0)
    assert len(splits['train']) == 9
    assert len(splits['val']) == 1
    assert len(splits['test']) == 1

    # raise exception when not enough trials 1
    with pytest.raises(ValueError):
        split_trials(6, 0, 8, 1, 1, 0)

    # raise exception when not enough trials 2
    with pytest.raises(ValueError):
        split_trials(11, 0, 8, 1, 1, 1)

    # properly insert gap trials
    splits = split_trials(13, 0, 8, 1, 1, 1)
    assert len(splits['train']) == 8
    assert len(splits['val']) == 1
    assert len(splits['test']) == 1

    max_train = np.max(splits['train'])
    assert not np.any(splits['val'] == max_train + 1)
    assert not np.any(splits['test'] == max_train + 1)

    max_val = np.max(splits['val'])
    assert not np.any(splits['test'] == max_val + 1)


def test_compute_sequences():

    # pass already batched data (in a list) through without modification
    data = [1, 2, 3]
    batch_data = compute_sequences(data, sequence_length=10)
    assert data == batch_data

    # batch sizes and quantity are correct
    T = 10
    N = 4
    B = 5
    data = np.random.randn(T, N)
    batch_data = compute_sequences(data, sequence_length=B)
    assert len(batch_data) == T // B
    assert batch_data[0].shape == (B, N)
    assert np.all(batch_data[0] == data[:B, :])


def test_single_dataset(hparams, data_generator):

    dataset = data_generator.datasets[0]

    # make sure we accurately record number of batches
    assert 'markers' in dataset.data
    n_trials = len(dataset.data['markers'])
    assert len(dataset) == n_trials

    # make sure batch sizes are good: access batch 0 from dataset 0
    batch = dataset[0]
    assert isinstance(batch, dict)
    assert 'markers' in batch.keys()
    assert 'batch_idx' in batch.keys()
    for key, val in batch.items():
        if key != 'batch_idx':
            # make sure all data tensors have the same batch dimension
            assert val.shape[0] == batch['markers'].shape[0]

    # make sure batches are torch tensors
    assert isinstance(batch['markers'], torch.Tensor)

    # make sure batches are on the right device
    assert batch['markers'].device == torch.device('cpu')


def test_data_generator(hparams, data_generator):

    # make sure we accurately record number of datasets
    n_datasets = len(hparams['expt_ids'])
    assert len(data_generator) == n_datasets

    # make sure batch sizes are good
    batch, datasets = data_generator.next_batch('train')

    assert np.all(np.array(datasets) < n_datasets)

    assert isinstance(batch, dict)
    assert 'markers' in batch.keys()
    assert 'batch_idx' in batch.keys()
    for key, val in batch.items():
        if key != 'batch_idx':
            # make sure all data tensors have the same batch dimension
            assert val.shape[0] == batch['markers'].shape[0]
            # make sure all data tensors have the same sequence dimension
            assert val.shape[1] == batch['markers'].shape[1]

    # make sure batches are torch tensors
    assert isinstance(batch['markers'], torch.Tensor)

    # make sure batches are on the right device
    assert batch['markers'].device == torch.device('cpu')


def test_data_generator_loading(hparams):

    hp = copy.deepcopy(hparams)
    hp['expt_ids'] = hparams['expt_ids'][:1]  # only load one dataset for efficiency

    # load markers only
    hp['lambda_strong'] = 0
    hp['lambda_weak'] = 0
    hp['lambda_pred'] = 1
    hp['lambda_task'] = 0
    data_gen = build_data_generator(hp)
    dtypes = data_gen.datasets[0].data.keys()
    assert 'markers' in dtypes
    assert 'labels_strong' not in dtypes
    assert 'labels_weak' not in dtypes
    assert 'tasks' not in dtypes

    # load markers + strong labels
    hp['lambda_strong'] = 1
    hp['lambda_weak'] = 0
    hp['lambda_pred'] = 0
    hp['lambda_task'] = 0
    data_gen = build_data_generator(hp)
    dtypes = data_gen.datasets[0].data.keys()
    assert 'markers' in dtypes
    assert 'labels_strong' in dtypes
    assert 'labels_weak' not in dtypes
    assert 'tasks' not in dtypes

    # load markers + weak labels
    hp['lambda_strong'] = 0
    hp['lambda_weak'] = 1
    hp['lambda_pred'] = 0
    hp['lambda_task'] = 0
    data_gen = build_data_generator(hp)
    dtypes = data_gen.datasets[0].data.keys()
    assert 'markers' in dtypes
    assert 'labels_strong' not in dtypes
    assert 'labels_weak' in dtypes
    assert 'tasks' not in dtypes

    # TODO: tasks; need to upload task data to repo

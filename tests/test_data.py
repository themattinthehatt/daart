import pytest
import numpy as np
from daart.data import split_trials, compute_batches


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


def test_compute_batches():

    # pass already batched data (in a list) through without modification
    data = [1, 2, 3]
    batch_data = compute_batches(data, batch_size=10)
    assert data == batch_data

    # batch sizes and quantity are correct
    T = 10
    N = 4
    B = 5
    data = np.random.randn(T, N)
    batch_data = compute_batches(data, batch_size=B)
    assert len(batch_data) == T // B
    assert batch_data[0].shape == (B, N)
    assert np.all(batch_data[0] == data[:B, :])

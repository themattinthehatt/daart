import pytest
import numpy as np
from daart import transforms


def test_compose():

    # motion energy -> zscore
    t = transforms.Compose([transforms.MotionEnergy(), transforms.ZScore()])
    T = 10
    D = 4
    signal = np.random.randn(T, D)
    s = t(signal)
    assert s.shape == (T, D)
    assert np.allclose(np.mean(s, axis=0), np.zeros(D), atol=1e-3)
    assert np.allclose(np.std(s, axis=0), np.ones(D), atol=1e-3)


def test_blockshuffle():

    def get_runs(sample):

        vals = np.unique(sample)
        n_time = len(sample)

        # mark first time point of state change with a nonzero number
        change = np.where(np.concatenate([[0], np.diff(sample)], axis=0) != 0)[0]
        # collect runs
        runs = {val: [] for val in vals}
        prev_beg = 0
        for curr_beg in change:
            runs[sample[prev_beg]].append(curr_beg - prev_beg)
            prev_beg = curr_beg
        runs[sample[-1]].append(n_time - prev_beg)
        return runs

    t = transforms.BlockShuffle(0)

    # signal has changed
    signal = np.array([0, 0, 0, 1, 1, 1, 2, 2, 0, 0, 1, 1])
    s = t(signal)
    assert not np.all(signal == s)

    # frequency of values unchanged
    n_ex_og = np.array([len(np.argwhere(signal == i)) for i in range(3)])
    n_ex_sh = np.array([len(np.argwhere(s == i)) for i in range(3)])
    assert np.all(n_ex_og == n_ex_sh)

    # distribution of runs unchanged
    runs_og = get_runs(signal)
    runs_sh = get_runs(s)
    for key in runs_og.keys():
        assert np.all(np.sort(np.array(runs_og[key])) == np.sort(np.array(runs_sh[key])))


def test_makeonehot():

    t = transforms.MakeOneHot()

    # pass one hot array without modification
    signal = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
    s = t(signal)
    assert np.all(signal == s)

    # correct one-hotting
    signal = np.array([3, 3, 2, 2, 0])
    s = t(signal)
    assert np.all(
        s == np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0]]))


def test_motionenergy():

    T = 10
    D = 4
    t = transforms.MotionEnergy()
    signal = np.random.randn(T, D)
    s = t(signal)
    me = np.vstack([np.zeros((1, signal.shape[1])), np.abs(np.diff(signal, axis=0))])
    assert s.shape == (T, D)
    assert np.allclose(s, me, atol=1e-3)
    assert np.all(me >= 0)


def test_unitize():

    t = transforms.Unitize()
    T = 10
    D = 3
    signal = 10 + 0.3 * np.random.randn(T, D)
    s = t(signal)
    assert s.shape == (T, D)
    assert np.quantile(s, 0.9) < 1
    assert np.quantile(s, 0.1) > 0


def test_zscore():

    t = transforms.ZScore()
    T = 10
    D = 3
    signal = 10 + 0.3 * np.random.randn(T, D)
    s = t(signal)
    assert s.shape == (T, D)
    assert np.allclose(np.mean(s, axis=0), np.zeros(D), atol=1e-3)
    assert np.allclose(np.std(s, axis=0), np.ones(D), atol=1e-3)

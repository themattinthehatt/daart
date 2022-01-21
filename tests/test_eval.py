import numpy as np
import pytest

from daart.eval import get_precision_recall, run_lengths


def test_precision_recall():

    # basic binary test: perfect predictions
    T = 20
    true = np.zeros(T,)
    pred = np.zeros_like(true)
    pr = get_precision_recall(true, pred, background=None)
    assert pr['precision'] == 1.0
    assert pr['recall'] == 1.0
    assert np.isclose(pr['f1'][0], 1.0)  # we have an eps in the denominator of f1 calculation

    # basic binary test: imperfect predictions
    T = 20
    true = np.zeros(T, )
    true[:int(T / 2)] = 1
    pred = np.zeros_like(true)
    pr = get_precision_recall(true, pred, background=None)
    print(pr)
    assert np.all(pr['precision'] == np.array([0.5, 0.0]))
    assert np.all(pr['recall'] == np.array([1.0, 0.0]))

    # basic multiclass test: perfect predictions
    T = 40
    true = np.zeros(T, )
    true[10:20] = 1
    true[20:30] = 2
    true[30:] = 3
    pred = np.copy(true)
    pr = get_precision_recall(true, pred, background=None)
    assert np.all(pr['precision'] == np.array([1.0, 1.0, 1.0, 1.0]))
    assert np.all(pr['recall'] == np.array([1.0, 1.0, 1.0, 1.0]))
    assert np.allclose(pr['f1'], np.array([1.0, 1.0, 1.0, 1.0]))

    # don't include background
    pr = get_precision_recall(true, pred, background=0)
    assert np.all(pr['precision'] == np.array([1.0, 1.0, 1.0]))
    assert np.all(pr['recall'] == np.array([1.0, 1.0, 1.0]))
    assert np.allclose(pr['f1'], np.array([1.0, 1.0, 1.0]))

    # basic multiclass test: imperfect predictions
    T = 40
    true = np.zeros(T, )
    true[10:20] = 1
    true[20:30] = 2
    true[30:] = 3
    pred = np.zeros_like(true)
    pred[:20] = 1
    pred[20:] = 2
    pr = get_precision_recall(true, pred, background=None)
    assert np.all(pr['precision'] == np.array([0.0, 0.5, 0.5, 0.0]))
    assert np.all(pr['recall'] == np.array([0.0, 1.0, 1.0, 0.0]))

    # don't include background
    pr = get_precision_recall(true, pred, background=0)
    assert np.all(pr['precision'] == np.array([1.0, 0.5, 0.0]))
    assert np.all(pr['recall'] == np.array([1.0, 1.0, 0.0]))

    # don't compute values for any datapoint
    true = np.zeros(T, )
    pred = np.zeros_like(true)
    pr = get_precision_recall(true, pred, background=0)
    assert len(pr['precision']) == 0
    assert len(pr['recall']) == 0
    assert len(pr['f1']) == 0

    # only allow 0 background for now
    with pytest.raises(AssertionError):
        get_precision_recall(true, pred, background=1)


def test_int_over_union():
    # TODO
    pass


def test_run_lengths():

    a = np.array([1, 1, 1, 0, 0, 4, 4, 4, 4, 4, 4, 0, 1, 1, 1, 1, 4, 1, 4])
    ls = run_lengths(a)
    assert isinstance(ls, dict)
    assert np.all(ls[0] == np.array([2, 1]))
    assert np.all(ls[1] == np.array([3, 4, 1]))
    assert np.all(ls[2] == np.array([]))
    assert np.all(ls[3] == np.array([]))
    assert np.all(ls[4] == np.array([6, 1, 1]))

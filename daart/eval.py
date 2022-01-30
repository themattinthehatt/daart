"""Evaluation functions for the daart package."""

import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import recall_score, precision_score
from typeguard import typechecked
from typing import List, Optional, Union

from daart.io import make_dir_if_not_exists

# to ignore imports for sphix-autoapidoc
__all__ = ['get_precision_recall', 'int_over_union', 'run_lengths', 'plot_training_curves']


@typechecked
def get_precision_recall(
        true_classes: np.ndarray,
        pred_classes: np.ndarray,
        background: Union[int, None] = 0,
        n_classes: Optional[int] = None
) -> dict:
    """Compute precision and recall for classifier.

    Parameters
    ----------
    true_classes : array-like
        entries should be in [0, K-1] where K is the number of classes
    pred_classes : array-like
        entries should be in [0, K-1] where K is the number of classes
    background : int or NoneType
        defines the background class that identifies points with no supervised label; these time
        points are omitted from the precision and recall calculations; if NoneType, no background
        class is utilized
    n_classes : int, optional
        total number of non-background classes; if NoneType, will be inferred from true classes

    Returns
    -------
    dict:
        'precision' (array-like): precision for each class (including background class)
        'recall' (array-like): recall for each class (including background class)

    """

    assert true_classes.shape[0] == pred_classes.shape[0]

    # find all data points that are not background
    if background is not None:
        assert background == 0  # need to generalize
        obs_idxs = np.where(true_classes != background)[0]
    else:
        obs_idxs = np.arange(true_classes.shape[0])

    if n_classes is None:
        n_classes = len(np.unique(true_classes[obs_idxs]))

    # set of labels to include in metric computations
    if background is not None:
        labels = np.arange(1, n_classes + 1)
    else:
        labels = np.arange(n_classes)

    precision = precision_score(
        true_classes[obs_idxs], pred_classes[obs_idxs],
        labels=labels, average=None, zero_division=0)
    recall = recall_score(
        true_classes[obs_idxs], pred_classes[obs_idxs],
        labels=labels, average=None, zero_division=0)

    # replace 0s with NaNs for classes with no ground truth
    # for n in range(precision.shape[0]):
    #     if precision[n] == 0 and recall[n] == 0:
    #         precision[n] = np.nan
    #         recall[n] = np.nan

    # compute f1
    p = precision
    r = recall
    f1 = 2 * p * r / (p + r + 1e-10)
    return {'precision': p, 'recall': r, 'f1': f1}


@typechecked
def int_over_union(array1: np.ndarray, array2: np.ndarray) -> dict:
    """Compute intersection over union for two 1D arrays.

    Parameters
    ----------
    array1 : array-like
        integer array of shape (n,)
    array2 : array-like
        integer array of shape (n,)

    Returns
    -------
    dict
        keys are integer values in arrays, values are corresponding IoU (float)

    """
    vals = np.unique(np.concatenate([np.unique(array1), np.unique(array2)]))
    iou = {val: np.nan for val in vals}
    for val in vals:
        intersection = np.sum((array1 == val) & (array2 == val))
        union = np.sum((array1 == val) | (array2 == val))
        iou[val] = intersection / union
    return iou


@typechecked
def run_lengths(array: np.ndarray) -> dict:
    """Compute distribution of run lengths for an array with integer entries.

    Parameters
    ----------
    array : array-like
        single-dimensional array

    Returns
    -------
    dict
        keys are integer values up to max value in array, values are lists of run lengths


    Example
    -------
    >>> a = [1, 1, 1, 0, 0, 4, 4, 4, 4, 4, 4, 0, 1, 1, 1, 1]
    >>> run_lengths(a)
    {0: [2, 1], 1: [3, 4], 2: [], 3: [], 4: [6]}

    """
    seqs = {k: [] for k in np.arange(np.max(array) + 1)}
    for key, iterable in itertools.groupby(array):
        seqs[key].append(len(list(iterable)))
    return seqs


@typechecked
def plot_training_curves(
        metrics_file: str,
        dtype: str = 'val',
        expt_ids: Optional[list] = None,
        save_file: Optional[str] = None,
        format: str = 'pdf'
) -> None:
    """Create training plots for each term in the objective function.

    The `dtype` argument controls which type of trials are plotted ('train' or 'val').
    Additionally, multiple models can be plotted simultaneously by varying one (and only one) of
    the following parameters:

    TODO

    Each of these entries must be an array of length 1 except for one option, which can be an array
    of arbitrary length (corresponding to already trained models). This function generates a single
    plot with panels for each of the following terms:

    - total loss
    - weak label loss
    - strong label loss
    - prediction loss

    Parameters
    ----------
    metrics_file : str
        csv file saved during training
    dtype : str
        'train' | 'val'
    expt_ids : list, optional
        dataset names for easier parsing
    save_file : str, optional
        absolute path of save file; does not need file extension
    format : str, optional
        format of saved image; 'pdf' | 'png' | 'jpeg' | ...

    """

    metrics_list = [
        'loss', 'loss_weak', 'loss_strong', 'loss_pred', 'loss_task', 'loss_kl', 'fc'
    ]

    metrics_dfs = [load_metrics_csv_as_df(metrics_file, metrics_list, expt_ids=expt_ids)]
    metrics_df = pd.concat(metrics_dfs, sort=False)

    if isinstance(expt_ids, list) and len(expt_ids) > 1:
        hue = 'dataset'
    else:
        hue = None

    sns.set_style('white')
    sns.set_context('talk')
    data_queried = metrics_df[
        (metrics_df.epoch > 10) & ~pd.isna(metrics_df.val) & (metrics_df.dtype == dtype)]
    g = sns.FacetGrid(
        data_queried, col='loss', col_wrap=2, hue=hue, sharey=False, height=4)
    g = g.map(plt.plot, 'epoch', 'val').add_legend()

    if save_file is not None:
        make_dir_if_not_exists(save_file)
        g.savefig(save_file + '.' + format, dpi=300, format=format)

    plt.close()


@typechecked
def load_metrics_csv_as_df(
        metric_file: str,
        metrics_list: List[str],
        expt_ids: Optional[List[str]] = None,
        test: bool = False
) -> pd.DataFrame:
    """Load metrics csv file and return as a pandas dataframe for easy plotting.

    Parameters
    ----------
    metric_file : str
        csv file saved during training
    metrics_list : list
        names of metrics to pull from csv; do not prepend with 'tr', 'val', or 'test'
    expt_ids : list, optional
        dataset names for easier parsing
    test : bool, optional
        True to only return test values (computed once at end of training)

    Returns
    -------
    pandas.DataFrame object

    """

    metrics = pd.read_csv(metric_file)

    # collect data from csv file
    metrics_df = []
    for i, row in metrics.iterrows():

        if row['dataset'] == -1:
            dataset = 'all'
        elif expt_ids is not None:
            dataset = expt_ids[int(row['dataset'])]
        else:
            dataset = row['dataset']

        if test:
            test_dict = {'dataset': dataset, 'epoch': row['epoch'], 'dtype': 'test'}
            for metric in metrics_list:
                name = 'test_%s' % metric
                if name not in row.keys():
                    continue
                metrics_df.append(pd.DataFrame(
                    {**test_dict, 'loss': metric, 'val': row[name]}, index=[0]))
        else:
            # make dict for val data
            val_dict = {'dataset': dataset, 'epoch': row['epoch'], 'dtype': 'val'}
            for metric in metrics_list:
                name = 'val_%s' % metric
                if name not in row.keys():
                    continue
                metrics_df.append(pd.DataFrame(
                    {**val_dict, 'loss': metric, 'val': row[name]}, index=[0]))
            # make dict for train data
            tr_dict = {'dataset': dataset, 'epoch': row['epoch'], 'dtype': 'train'}
            for metric in metrics_list:
                name = 'tr_%s' % metric
                if name not in row.keys():
                    continue
                metrics_df.append(pd.DataFrame(
                    {**tr_dict, 'loss': metric, 'val': row[name]}, index=[0]))

    return pd.concat(metrics_df, sort=True)

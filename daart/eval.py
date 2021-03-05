"""Evaluation functions for the daart package."""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import recall_score, precision_score

from daart.io import make_dir_if_not_exists


def get_precision_recall(true_classes, pred_classes, background=0):
    """Compute precision and recall for classifier.

    Parameters
    ----------
    true_classes : array-like
        entries should be in [0, K-1] where K is the number of classes
    pred_classes : array-like
        entries should be in [0, K-1] where K is the number of classes
    background : int
        defines the background class that identifies points with no supervised label; these time
        points are omitted from the precision and recall calculations

    Returns
    -------
    dict:
        'precision' (array-like): precision for each class (including background class)
        'recall' (array-like): recall for each class (including background class)

    """

    assert true_classes.shape[0] == pred_classes.shape[0]

    # find all data points that are not background
    obs_idxs = np.where(true_classes != background)[0]
    n_classes = len(np.unique(true_classes[obs_idxs]))

    precision = precision_score(
        true_classes[obs_idxs], pred_classes[obs_idxs], average=None, zero_division=0)
    recall = recall_score(
        true_classes[obs_idxs], pred_classes[obs_idxs], average=None, zero_division=0)

    # replace 0s with NaNs for classes with no ground truth
    for n in range(precision.shape[0]):
        if precision[n] == 0 and recall[n] == 0:
            precision[n] = np.nan
            recall[n] = np.nan

    # chop off background class if it exists
    p = precision if len(precision) == n_classes else precision[1:]
    r = recall if len(recall) == n_classes else recall[1:]
    f1 = 2 * p * r / (p + r)
    return {'precision': p, 'recall': r, 'f1': f1}


def plot_training_curves(
        metrics_file, dtype='val', expt_ids=None, save_file=None, format='pdf'):
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

    metrics_list = ['loss', 'loss_weak', 'loss_strong', 'loss_pred', 'fc']

    metrics_dfs = []
    metrics_dfs.append(load_metrics_csv_as_df(metrics_file, metrics_list, expt_ids=expt_ids))
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


def load_metrics_csv_as_df(metric_file, metrics_list, expt_ids=None, test=False):
    """Load metrics csv file and return as a pandas dataframe for easy plotting.

    Parameters
    ----------
    metric_file : str
        csv file saved during training
    metrics_list : list
        names of metrics to pull from csv; do not prepend with 'tr', 'val', or 'test'
    expt_ids : list, optional
        dataset names for easier parsing
    test : bool
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
                metrics_df.append(pd.DataFrame(
                    {**test_dict, 'loss': metric, 'val': row['test_%s' % metric]}, index=[0]))
        else:
            # make dict for val data
            val_dict = {'dataset': dataset, 'epoch': row['epoch'], 'dtype': 'val'}
            for metric in metrics_list:
                metrics_df.append(pd.DataFrame(
                    {**val_dict, 'loss': metric, 'val': row['val_%s' % metric]}, index=[0]))
            # make dict for train data
            tr_dict = {'dataset': dataset, 'epoch': row['epoch'], 'dtype': 'train'}
            for metric in metrics_list:
                metrics_df.append(pd.DataFrame(
                    {**tr_dict, 'loss': metric, 'val': row['tr_%s' % metric]}, index=[0]))
    return pd.concat(metrics_df, sort=True)

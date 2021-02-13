"""Evaluation functions for the daart package."""

from sklearn.metrics import recall_score, precision_score
import numpy as np


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

    precision = precision_score(
        true_classes[obs_idxs], pred_classes[obs_idxs], average=None, zero_division=0)
    recall = recall_score(
        true_classes[obs_idxs], pred_classes[obs_idxs], average=None, zero_division=0)

    return {'precision': precision, 'recall': recall}

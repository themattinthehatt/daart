"""Utility functions for daart package."""

import logging
import os

from daart.data import compute_sequence_pad, DataGenerator
from daart.transforms import ZScore


# to ignore imports for sphix-autoapidoc
__all__ = ['build_data_generator']


def build_data_generator(hparams: dict) -> DataGenerator:
    """Helper function to build a data generator from hparam dict."""

    signals = []
    transforms = []
    paths = []

    for expt_id in hparams['expt_ids']:

        signals_curr = []
        transforms_curr = []
        paths_curr = []

        # DLC markers or features (i.e. from simba)
        input_type = hparams.get('input_type', 'markers')
        markers_file = os.path.join(hparams['data_dir'], input_type, expt_id + '_labeled.h5')
        if not os.path.exists(markers_file):
            markers_file = os.path.join(hparams['data_dir'], input_type, expt_id + '_labeled.csv')
        if not os.path.exists(markers_file):
            markers_file = os.path.join(hparams['data_dir'], input_type, expt_id + '_labeled.npy')
        if not os.path.exists(markers_file):
            msg = 'could not find marker file for %s at %s' % (expt_id, markers_file)
            logging.info(msg)
            raise FileNotFoundError(msg)
        signals_curr.append('markers')
        transforms_curr.append(ZScore())
        paths_curr.append(markers_file)

        # hand labels
        if hparams.get('lambda_strong', 0) > 0:
            if expt_id not in hparams.get('expt_ids_to_keep', hparams['expt_ids']):
                hand_labels_file = None
            else:
                hand_labels_file = os.path.join(
                    hparams['data_dir'], 'labels-hand', expt_id + '_labels.csv')
                if not os.path.exists(hand_labels_file):
                    logging.warning('did not find hand labels file for %s' % expt_id)
                    hand_labels_file = None
            signals_curr.append('labels_strong')
            transforms_curr.append(None)
            paths_curr.append(hand_labels_file)

        # heuristic labels
        if hparams.get('lambda_weak', 0) > 0:
            heur_labels_file = os.path.join(
                hparams['data_dir'], 'labels-heuristic', expt_id + '_labels.csv')
            signals_curr.append('labels_weak')
            transforms_curr.append(None)
            paths_curr.append(heur_labels_file)

        # tasks
        if hparams.get('lambda_task', 0) > 0:
            tasks_labels_file = os.path.join(hparams['data_dir'], 'tasks', expt_id + '.csv')
            signals_curr.append('tasks')
            transforms_curr.append(ZScore())
            paths_curr.append(tasks_labels_file)

        # define data generator signals
        signals.append(signals_curr)
        transforms.append(transforms_curr)
        paths.append(paths_curr)

    # compute padding needed to account for convolutions
    hparams['sequence_pad'] = compute_sequence_pad(hparams)

    # build data generator
    data_gen = DataGenerator(
        hparams['expt_ids'], signals, transforms, paths, device=hparams['device'],
        sequence_length=hparams['sequence_length'], sequence_pad=hparams['sequence_pad'],
        batch_size=hparams['batch_size'],
        trial_splits=hparams['trial_splits'], train_frac=hparams['train_frac'],
        input_type=hparams.get('input_type', 'markers'))

    # automatically compute input/output sizes from data
    input_size = 0
    for batch in data_gen.datasets[0].data['markers']:
        if batch.shape[1] == 0:
            continue
        else:
            input_size = batch.shape[1]
            break
    hparams['input_size'] = input_size

    if hparams.get('lambda_task', 0) > 0:
        task_size = 0
        for batch in data_gen.datasets[0].data['tasks']:
            if batch.shape[1] == 0:
                continue
            else:
                task_size = batch.shape[1]
                break
        hparams['task_size'] = task_size

    return data_gen

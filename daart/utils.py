"""Utility functions for daart package."""

import logging
import os

from daart.data import compute_sequence_pad, DataGenerator
from daart.transforms import ZScore


# to ignore imports for sphix-autoapidoc
__all__ = ['build_data_generator', 'collect_callbacks']


def build_data_generator(hparams: dict) -> DataGenerator:
    """Helper function to build a data generator from hparam dict."""

    signals = []
    transforms = []
    paths = []

    for expt_id in hparams['expt_ids']:

        signals_curr = []
        transforms_curr = []
        paths_curr = []

        # DLC markers or features (e.g. from simba)
        input_type = hparams.get('input_type', 'markers')
        base_dir = os.path.join(hparams['data_dir'], input_type)
        possible_markers_files = [
            os.path.join(base_dir, expt_id + '_labeled.h5'),
            os.path.join(base_dir, expt_id + '_labeled.csv'),
            os.path.join(base_dir, expt_id + '_labeled.npy'),
            os.path.join(base_dir, expt_id + '.h5'),
            os.path.join(base_dir, expt_id + '.csv'),
            os.path.join(base_dir, expt_id + '.npy'),
        ]
        markers_file = None
        for marker_file_ in possible_markers_files:
            if os.path.exists(marker_file_):
                markers_file = marker_file_
                break
        if markers_file is None:
            msg = f'did not find marker file for {expt_id} in {base_dir}'
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
                base_dir = os.path.join(hparams['data_dir'], 'labels-hand')
                possible_hand_labels_files = [
                    os.path.join(base_dir, expt_id + '_labels.csv'),
                    os.path.join(base_dir, expt_id + '.csv'),
                ]
                hand_labels_file = None
                for hand_labels_file_ in possible_hand_labels_files:
                    if os.path.exists(hand_labels_file_):
                        hand_labels_file = hand_labels_file_
                        break
                if hand_labels_file is None:
                    logging.warning(f'did not find hand labels file for {expt_id} in {base_dir}')
            signals_curr.append('labels_strong')
            transforms_curr.append(None)
            paths_curr.append(hand_labels_file)

        # heuristic labels
        if hparams.get('lambda_weak', 0) > 0:
            base_dir = os.path.join(hparams['data_dir'], 'labels-heuristic')
            possible_heur_labels_files = [
                os.path.join(base_dir, expt_id + '_labels.csv'),
                os.path.join(base_dir, expt_id + '.csv'),
            ]
            heur_labels_file = None
            for heur_labels_file_ in possible_heur_labels_files:
                if os.path.exists(heur_labels_file_):
                    heur_labels_file = heur_labels_file_
                    break
            if heur_labels_file is None:
                logging.warning(f'did not find heuristic labels file for {expt_id} in {base_dir}')
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
        hparams['expt_ids'], signals, transforms, paths,
        device=hparams['device'],
        sequence_length=hparams['sequence_length'],
        sequence_pad=hparams['sequence_pad'],
        batch_size=hparams['batch_size'],
        trial_splits=hparams['trial_splits'],
        train_frac=hparams['train_frac'],
        input_type=hparams.get('input_type', 'markers'),
    )

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


def collect_callbacks(hparams: dict) -> list:
    """Helper function to build a list of callbacks from hparam dict."""

    callbacks = []
    if hparams['enable_early_stop']:
        from daart.callbacks import EarlyStopping
        # Note that patience does not account for val check interval values greater than 1;
        # for example, if val_check_interval=5 and patience=20, then the model will train
        # for at least 5 * 20 = 100 epochs before training can terminate
        callbacks.append(EarlyStopping(patience=hparams['early_stop_history']))

    if hparams.get('semi_supervised_algo', 'none') == 'pseudo_labels':
        from daart.callbacks import AnnealHparam, PseudoLabels
        if hparams['lambda_weak'] == 0:
            print('warning! use lambda_weak in model.yaml to weight pseudo label loss')
        else:
            callbacks.append(AnnealHparam(
                hparams=hparams,
                key='lambda_weak',
                epoch_start=hparams['anneal_start'],
                epoch_end=hparams['anneal_end'],
            ))
            callbacks.append(PseudoLabels(
                prob_threshold=hparams['prob_threshold'],
                epoch_start=hparams['anneal_start'],
            ))
    elif hparams.get('semi_supervised_algo', 'none') == 'ups':
        from daart.callbacks import AnnealHparam, UPS
        if hparams['lambda_weak'] == 0:
            print('warning! use lambda_weak in model.yaml to weight pseudo label loss')
        else:
            callbacks.append(AnnealHparam(
                hparams=hparams,
                key='lambda_weak',
                epoch_start=hparams['anneal_start'],
                epoch_end=hparams['anneal_end'],
            ))
            callbacks.append(UPS(
                prob_threshold=hparams['prob_threshold'],
                variance_threshold=hparams['variance_threshold'],
                epoch_start=hparams['anneal_start'],
            ))

    if hparams.get('variational', False):
        from daart.callbacks import AnnealHparam
        callbacks.append(AnnealHparam(
            hparams=hparams,
            key='kl_weight',
            epoch_start=0,
            epoch_end=100,
        ))

    return callbacks

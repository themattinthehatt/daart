import argparse
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import shutil
import subprocess
import time
import yaml

from daart.io import find_experiment


# https://stackoverflow.com/a/39452138
CEND = '\33[0m'
BOLD = '\033[1m'
CBLACK = '\33[30m'
CRED = '\33[31m'
CGREEN = '\33[32m'
CYELLOW = '\33[33m'
CBLUE = '\33[34m'
CVIOLET = '\33[35m'

TEMP_DATA = {
    'n_time': 1000,
    'n_labels': 3,
    'n_markers': 4
}

SESSIONS = ['sess-0', 'sess-1']

MODELS_TO_FIT = [
    {'backbone': 'temporal-mlp', 'sessions': [SESSIONS[0]]},
    {'backbone': 'lstm', 'sessions': [SESSIONS[0]]},
    {'backbone': 'gru', 'sessions': [SESSIONS[0]]},
    {'backbone': 'dtcn', 'sessions': [SESSIONS[0]]},
    {'backbone': 'dtcn', 'sessions': SESSIONS},  # multiple sessions
    {'backbone': 'variational-dtcn', 'sessions': [SESSIONS[0]]},  # variational model
]

"""
TODO:
    - how to print traceback when testtube fails?
"""


def make_tmp_data(data_dir):
    """Make synthetic data: strong labels, weak labels, markers."""

    for session in SESSIONS:

        # DLC markers
        marker_file = os.path.join(data_dir, 'markers', session + '_labeled.csv')
        file_dir = os.path.dirname(marker_file)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        make_dlc_markers(marker_file, TEMP_DATA['n_time'], TEMP_DATA['n_markers'])

        # heuristic labels
        labels_file = os.path.join(data_dir, 'labels-heuristic', session + '_labels.csv')
        file_dir = os.path.dirname(labels_file)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        make_labels(labels_file, TEMP_DATA['n_time'], TEMP_DATA['n_labels'], nan_frac=0)

        # hand labels
        hand_labels_file = os.path.join(data_dir, 'labels-hand', session + '_labels.csv')
        file_dir = os.path.dirname(hand_labels_file)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        make_labels(hand_labels_file, TEMP_DATA['n_time'], TEMP_DATA['n_labels'], nan_frac=0.8)

        # tasks
        tasks_file = os.path.join(data_dir, 'tasks', session + '.csv')
        file_dir = os.path.dirname(tasks_file)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        make_tasks(tasks_file, TEMP_DATA['n_time'], TEMP_DATA['n_labels'] + 2)


def make_dlc_markers(save_file, n_time, n_markers):

    data = np.empty((n_time, 3 * n_markers), dtype='float')
    data[:, 0::3] = np.random.randn(n_time, n_markers)
    data[:, 1::3] = np.random.randn(n_time, n_markers)
    data[:, 2::3] = 0.5

    pdindex = pd.MultiIndex.from_product(
        [['scorer_'], ['marker_%i' % i for i in range(n_markers)], ['x', 'y', 'likelihood']],
        names=['scorer', 'bodyparts', 'coords'])

    imagenames = np.arange(n_time)
    df = pd.DataFrame(data, columns=pdindex, index=imagenames)
    df.to_csv(save_file)


def make_labels(save_file, n_time, n_labels, nan_frac=0.0):
    file_ext = save_file.split('.')[-1]
    if file_ext == 'pkl':
        # pickle file structure for heuristic labels
        states = np.random.randint(0, n_labels + 1, n_time)
        state_labels = {i + 1: 'state_%i' % i for i in range(n_labels)}
        state_labels[0] = 'background'
        with open(save_file, 'wb') as f:
            pickle.dump({'states': states, 'state_labels': state_labels}, f)
    elif file_ext == 'csv':
        # csv file structure output by deepethogram
        m = np.random.randint(0, 2, (n_time, n_labels))
        nans = np.zeros((n_time, 1))
        n_nans = int(n_time * nan_frac)
        nans[np.random.permutation(n_time)[:n_nans]] = 1
        labels = np.hstack([nans, m]).astype('int')
        df = pd.DataFrame(
            labels, columns=['background'] + ['label_%i' % i for i in range(n_labels)])
        df.to_csv(save_file)
    else:
        raise ValueError('invalid file extension "%s"; must use "pkl" or "csv"' % file_ext)


def make_tasks(save_file, n_time, n_tasks):
    data = np.random.randn(n_time, n_tasks)
    imagenames = np.arange(n_time)
    df = pd.DataFrame(data, columns=['task_%i' % i for i in range(n_tasks)], index=imagenames)
    df.to_csv(save_file)


def define_new_config_values(backbone, sessions=['sess-0'], base_dir=None):

    # data vals
    data_dict = {
        'output_size': TEMP_DATA['n_labels'] + 1,
        'expt_ids': sessions,
        'data_dir': base_dir,
        'results_dir': base_dir}

    # training vals
    train_frac = 0.5
    trial_splits = '9;1;0;0'

    train_dict = {
        'sequence_length': 50,
        'batch_size': 2,
        'min_epochs': 1,
        'max_epochs': 1,
        'enable_early_stop': False,
        'train_frac': train_frac,
        'trial_splits': trial_splits,
        'plot_train_curves': False,
        'gpus_vis': '0',
        'tt_n_cpu_workers': 2
    }

    # model vals
    expt_name = 'test'
    lambda_weak = 1
    lambda_strong = 1
    lambda_pred = 1
    lambda_task = 1

    if backbone == 'temporal-mlp':
        new_values = {
            'data': data_dict,
            'model': {
                'tt_experiment_name': expt_name,
                'backbone': backbone,
                'variational': False,
                'n_lags': 1,
                'lambda_weak': lambda_weak,
                'lambda_strong': lambda_strong,
                'lambda_pred': lambda_pred,
                'lambda_task': lambda_task,
            },
            'train': train_dict}
    elif backbone in ['lstm', 'gru']:
        new_values = {
            'data': data_dict,
            'model': {
                'tt_experiment_name': expt_name,
                'backbone': backbone,
                'variational': False,
                'n_lags': 1,
                'lambda_weak': lambda_weak,
                'lambda_strong': lambda_strong,
                'lambda_pred': lambda_pred,
                'lambda_task': lambda_task,
                'bidirectional': True,
            },
            'train': train_dict}
    elif backbone in ['tcn', 'dtcn']:
        new_values = {
            'data': data_dict,
            'model': {
                'tt_experiment_name': expt_name,
                'backbone': backbone,
                'variational': False,
                'n_hid_layers': 2,
                'n_lags': 1,
                'lambda_weak': lambda_weak,
                'lambda_strong': lambda_strong,
                'lambda_pred': lambda_pred,
                'lambda_task': lambda_task,
                'dropout': 0.1,
            },
            'train': train_dict}
    elif backbone == 'variational-dtcn':
        new_values = {
            'data': data_dict,
            'model': {
                'tt_experiment_name': expt_name,
                'backbone': 'dtcn',
                'variational': True,
                'n_hid_layers': 2,
                'n_lags': 1,
                'lambda_weak': lambda_weak,
                'lambda_strong': lambda_strong,
                'lambda_pred': lambda_pred,
                'lambda_task': lambda_task,
                'dropout': 0.1,
            },
            'train': train_dict}
    else:
        raise NotImplementedError

    return new_values


def update_config_files(config_files, new_values, save_dir=None):
    """

    Parameters
    ----------
    config_files : dict
        absolute paths to base config files
    new_values : dict of dict
        keys correspond to those in `config_files`; values are dicts with key-value pairs
        defining which keys in the config file are updated with which values
    save_dir : str or NoneType, optional
        if not None, directory in which to save updated config files; filename will be same as
        corresponding base config

    Returns
    -------
    tuple
        (updated config dicts, updated config files)

    """
    new_config_dicts = {}
    new_config_files = {}
    for config_name, config_file in config_files.items():
        # load base config file into dict
        config_dict = yaml.safe_load(open(config_file, 'r'))
        # change key/value pairs
        for key, val in new_values[config_name].items():
            config_dict[key] = val
        new_config_dicts[config_name] = config_dict
        # save as new config file in save_dir
        if save_dir is not None:
            filename = os.path.join(save_dir, os.path.basename(config_file))
            new_config_files[config_name] = filename
            yaml.dump(config_dict, open(filename, 'w'))
    return new_config_dicts, new_config_files


def get_call_str(fit_file, config_files):
    call_str = [
        'python',
        fit_file,
        '--data_config', config_files['data'],
        '--model_config', config_files['model'],
        '--train_config', config_files['train']]
    return call_str


def fit_model(fit_file, config_files):
    call_str = get_call_str(fit_file, config_files)
    try:
        subprocess.call(' '.join(call_str), shell=True)
        result_str = BOLD + CGREEN + 'passed' + CEND
    except BaseException as error:
        result_str = BOLD + CRED + 'failed: %s' % str(error) + CEND
    return result_str


def check_model(config_dicts):
    hparams = {**config_dicts['data'], **config_dicts['model'], **config_dicts['train']}
    # pick out single model if multiple were fit with test tube
    # for key, val in hparams.items():
    #     if isinstance(val, list):
    #         hparams[key] = val[-1]
    exists = find_experiment(hparams)
    if exists is not None:
        result_str = BOLD + CGREEN + 'passed' + CEND
    else:
        result_str = BOLD + CRED + 'failed' + CEND
    return result_str


def main(args):
    """Integration testing function.

    Must call from main daart directory as:
    $: python tests/integration.py

    """

    t_beg = time.time()

    # -------------------------------------------
    # setup
    # -------------------------------------------

    # create temp dir to store data
    base_dir = Path.home()
    if os.path.exists(base_dir):
        base_dir = os.path.join(base_dir, 'daart_tmp_data_AaA')
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    else:
        shutil.rmtree(base_dir)
        os.mkdir(base_dir)

    # make temp data
    print('creating temp data...', end='')
    make_tmp_data(base_dir)
    print('done')

    config_dir = os.path.join(os.getcwd(), 'data', 'configs')
    fit_file = os.path.join(os.getcwd(), 'examples', 'fit_models.py')

    # store results of tests
    print_strs = {}

    # -------------------------------------------
    # fit models
    # -------------------------------------------
    for model in MODELS_TO_FIT:

        # modify example config files
        base_config_files = {
            'data': os.path.join(config_dir, 'data.yaml'),
            'model': os.path.join(config_dir, 'model.yaml'),
            'train': os.path.join(config_dir, 'train.yaml')}
        new_values = define_new_config_values(
            model['backbone'], model['sessions'], base_dir)
        config_dicts, new_config_files = update_config_files(
            base_config_files, new_values, base_dir)

        # fit model
        print('\n\n---------------------------------------------------')
        print('model: %s' % model['backbone'])
        print('session: %s' % model['sessions'])
        print('---------------------------------------------------\n\n')
        fit_model(fit_file, new_config_files)

        # check model
        if len(model['sessions']) > 1:
            model_key = '%s-multisession' % model['backbone']
        else:
            model_key = model['backbone']
        print_strs[model_key] = check_model(config_dicts)

    # remove temp dirs
    shutil.rmtree(base_dir)

    # -------------------------------------------
    # print results
    # -------------------------------------------
    print('\n%s================== Integration Test Results ==================%s\n' % (BOLD, CEND))
    for key, val in print_strs.items():
        print('%s: %s' % (key, val))

    t_end = time.time()
    print('\ntotal time to perform integration test: %s%f sec%s\n' % (BOLD, t_end - t_beg, CEND))


if __name__ == '__main__':

    # parse command line args and send to main test function
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default=0, type=int)
    namespace, _ = parser.parse_known_args()
    main(namespace)

"""Example script for fitting daart models from the command line with test-tube package."""

import copy
import numpy as np
import os
import sys
from test_tube import HyperOptArgumentParser
import time
import torch
import yaml

from daart.data import DataGenerator
from daart.eval import plot_training_curves
from daart.io import export_expt_info_to_csv, export_hparams
from daart.io import find_experiment
from daart.io import get_expt_dir, get_model_dir, get_subdirs
from daart.models import Segmenter
from daart.transforms import ZScore
from daart.utils import compute_batch_pad


def run_main(hparams, *args):

    if not isinstance(hparams, dict):
        hparams = vars(hparams)

    # print hparams to console
    print_hparams(hparams)

    # start at random times (so test tube creates separate folders)
    t = time.time()
    np.random.seed(int(100000000000 * t) % (2 ** 32 - 1))
    time.sleep(np.random.uniform(2))

    # create test-tube experiment
    hparams['expt_ids'] = hparams['expt_ids'].split(';')
    hparams, exp = create_tt_experiment(hparams)
    if hparams is None:
        print('Experiment exists! Aborting fit')
        return

    # where model results will be saved
    model_save_path = hparams['tt_version_dir']

    # -------------------------------------
    # build data generator
    # -------------------------------------
    signals = []
    transforms = []
    paths = []

    for expt_id in hparams['expt_ids']:

        # DLC markers or features (i.e. from simba)
        input_type = hparams.get('input_type', 'markers')
        markers_file = os.path.join(hparams['data_dir'], input_type, expt_id + '_labeled.h5')
        if not os.path.exists(markers_file):
            markers_file = os.path.join(hparams['data_dir'], input_type, expt_id + '_labeled.csv')
        if not os.path.exists(markers_file):
            markers_file = os.path.join(hparams['data_dir'], input_type, expt_id + '_labeled.npy')
        if not os.path.exists(markers_file):
            raise FileNotFoundError('could not find marker file for %s' % expt_id)

        # heuristic labels
        labels_file = os.path.join(
            hparams['data_dir'], 'labels-heuristic', expt_id + '_labels.csv')

        # hand labels
        hand_labels_file = os.path.join(
            hparams['data_dir'], 'labels-hand', expt_id + '_labels.csv')

        # define data generator signals
        signals.append(['markers', 'labels_weak', 'labels_strong'])
        transforms.append([ZScore(), None, None])
        paths.append([markers_file, labels_file, hand_labels_file])

    # compute padding needed to account for convolutions
    hparams['batch_pad'] = compute_batch_pad(hparams)

    # build data generator
    data_gen = DataGenerator(
        hparams['expt_ids'], signals, transforms, paths, device=hparams['device'],
        batch_size=hparams['batch_size'], trial_splits=hparams['trial_splits'],
        train_frac=hparams['train_frac'], batch_pad=hparams['batch_pad'])
    print(data_gen)

    # automatically compute input/output sizes from data
    hparams['input_size'] = data_gen.datasets[0].data['markers'][0].shape[1]

    # -------------------------------------
    # build model
    # -------------------------------------
    torch.manual_seed(hparams.get('rng_seed_model', 0))
    model = Segmenter(hparams)
    model.to(hparams['device'])
    print(model)

    # -------------------------------------
    # train model
    # -------------------------------------
    model.fit(data_gen, save_path=model_save_path, **hparams)

    # update hparams upon successful training
    hparams['training_completed'] = True
    export_hparams(hparams)

    # save training curves
    if hparams.get('plot_train_curves', False):
        plot_training_curves(
            os.path.join(model_save_path, 'metrics.csv'), dtype='train',
            expt_ids=hparams['expt_ids'], save_file=os.path.join(model_save_path, 'train_curves'),
            format='png')
        plot_training_curves(
            os.path.join(model_save_path, 'metrics.csv'), dtype='val',
            expt_ids=hparams['expt_ids'], save_file=os.path.join(model_save_path, 'val_curves'),
            format='png')

    # get rid of unneeded logging info
    clean_tt_dir(hparams)


def get_all_params():
    # raise error if user has other command line arguments specified
    if len(sys.argv[1:]) != 6:
        raise ValueError('No command line arguments allowed other than config file names')

    def add_to_parser(parser, arg_name, value):
        if arg_name == 'expt_ids':
            # treat expt_ids differently, want to parse full lists as one
            if isinstance(value, list):
                value = ';'.join(value)
            parser.add_argument('--' + arg_name, default=value)
        elif isinstance(value, list):
            parser.opt_list('--' + arg_name, options=value, tunable=True)
        else:
            parser.add_argument('--' + arg_name, default=value)

    # create parser
    parser = HyperOptArgumentParser(strategy='grid_search')
    parser.add_argument('--data_config', type=str)
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--train_config', type=str)

    namespace, extra = parser.parse_known_args()

    # add arguments from all configs
    configs = [namespace.data_config, namespace.model_config, namespace.train_config]
    for config in configs:
        config_dict = yaml.safe_load(open(config))
        for (key, value) in config_dict.items():
            add_to_parser(parser, key, value)

    return parser.parse_args()


def print_hparams(hparams):
    """Pretty print hparams to console."""
    config_files = ['data', 'model', 'train']
    for config_file in config_files:
        print('\n%s CONFIG:' % config_file.upper())
        config_dict = yaml.safe_load(open(hparams['%s_config' % config_file]))
        for key in config_dict.keys():
            print('    {}: {}'.format(key, hparams[key]))
    print('')


def create_tt_experiment(hparams):
    """Create test-tube experiment for organizing model fits.

    Parameters
    ----------
    hparams : dict
        dictionary of hyperparameters defining experiment

    Returns
    -------
    tuple
        - if experiment defined by hparams already exists, returns `(None, None)`
        - if experiment does not exist, returns `(hparams, exp)`

    """
    from test_tube import Experiment

    # get model path
    hparams['expt_dir'] = get_expt_dir(hparams['results_dir'], hparams['expt_ids'])
    if not os.path.isdir(hparams['expt_dir']):
        os.makedirs(hparams['expt_dir'])
        export_expt_info_to_csv(hparams['expt_dir'], hparams['expt_ids'])
    hparams['model_dir'] = get_model_dir(hparams['expt_dir'], hparams)
    tt_expt_dir = os.path.join(hparams['model_dir'], hparams['tt_experiment_name'])
    if not os.path.isdir(tt_expt_dir):
        os.makedirs(tt_expt_dir)

    # check to see if experiment already exists
    if find_experiment(hparams) is not None:
        return None, None

    exp = Experiment(
        name=hparams['tt_experiment_name'],
        debug=False,
        save_dir=os.path.dirname(hparams['model_dir']))
    hparams['version'] = exp.version
    hparams['tt_version_dir'] = os.path.join(tt_expt_dir, 'version_%i' % exp.version)

    return hparams, exp


def clean_tt_dir(hparams):
    """Delete all (unnecessary) subdirectories in the model directory (created by test-tube)"""
    import shutil
    # get subdirs
    version_dir = hparams['tt_version_dir']
    subdirs = get_subdirs(version_dir)
    for subdir in subdirs:
        shutil.rmtree(os.path.join(version_dir, subdir))
    os.remove(os.path.join(version_dir, 'meta.experiment'))


if __name__ == '__main__':

    """To run:

    (daart) $: python fit_models.py --data_config /path/to/data.yaml 
       --model_config /path/to/model.yaml --train_config /path/to/train.yaml

    For example yaml files, see the `configs` subdirectory inside the daart home directory

    NOTE: this script assumes a specific naming convention for markers and labels (see L54-L65). 
    You'll need to update these lines to be consistent with your own naming conventions.
    
    """

    hyperparams = get_all_params()

    if hyperparams.device == 'cuda':
        if isinstance(hyperparams.gpus_vis, int):
            gpu_ids = [str(hyperparams.gpus_vis)]
        else:
            gpu_ids = hyperparams.gpus_vis.split(';')
        hyperparams.optimize_parallel_gpu(
            run_main,
            gpu_ids=gpu_ids)

    elif hyperparams.device == 'cpu':
        hyperparams.optimize_parallel_cpu(
            run_main,
            nb_trials=hyperparams.tt_n_cpu_trials,
            nb_workers=hyperparams.tt_n_cpu_workers)

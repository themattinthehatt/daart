"""Example script for fitting daart models from the command line with test-tube package."""

import copy
import logging
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
from daart.train import Trainer
from daart.utils import build_data_generator


def run_main(hparams, *args):

    if not isinstance(hparams, dict):
        hparams = vars(hparams)

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

    # set up error logging (different from train logging)
    logging.basicConfig(
        filename=os.path.join(hparams['tt_version_dir'], 'console.log'),
        filemode='w', level=logging.DEBUG,
        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # add logging to console

    # run train model script
    try:
        train_model(hparams)
    except:
        logging.exception('error traceback')


def train_model(hparams):

    # print hparams to console
    print_str = print_hparams(hparams)
    logging.info(print_str)

    # -------------------------------------
    # build data generator
    # -------------------------------------
    data_gen = build_data_generator(hparams)
    logging.info(data_gen)

    # -------------------------------------
    # build model
    # -------------------------------------
    torch.manual_seed(hparams.get('rng_seed_model', 0))
    model = Segmenter(hparams)
    model.to(hparams['device'])
    logging.info(model)

    # -------------------------------------
    # set up training callbacks
    # -------------------------------------
    callbacks = []
    if hparams['enable_early_stop']:
        from daart.callbacks import EarlyStopping
        # Note that patience does not account for val check interval values greater than 1;
        # for example, if val_check_interval=5 and patience=20, then the model will train
        # for at least 5 * 20 = 100 epochs before training can terminate
        callbacks.append(EarlyStopping(patience=hparams['early_stop_history']))
    if hparams.get('semi_supervised_algo', 'none') == 'pseudo_labels':
        from daart.callbacks import AnnealHparam, PseudoLabels
        if model.hparams['lambda_weak'] == 0:
            print('warning! use lambda_weak in model.yaml to weight pseudo label loss')
        else:
            callbacks.append(AnnealHparam(
                hparams=model.hparams, key='lambda_weak', epoch_start=hparams['anneal_start'],
                epoch_end=hparams['anneal_end']))
            callbacks.append(PseudoLabels(
                prob_threshold=hparams['prob_threshold'], epoch_start=hparams['anneal_start']))

    # -------------------------------------
    # train model + cleanup
    # -------------------------------------
    trainer = Trainer(**hparams, callbacks=callbacks)
    trainer.fit(model, data_gen, save_path=hparams['tt_version_dir'])

    # update hparams upon successful training
    hparams['training_completed'] = True
    export_hparams(hparams)

    # save training curves
    if hparams.get('plot_train_curves', False):
        plot_training_curves(
            os.path.join(model_save_path, 'metrics.csv'), dtype='train',
            expt_ids=hparams['expt_ids'],
            save_file=os.path.join(hparams['tt_version_dir'], 'train_curves'),
            format='png')
        plot_training_curves(
            os.path.join(model_save_path, 'metrics.csv'), dtype='val',
            expt_ids=hparams['expt_ids'],
            save_file=os.path.join(hparams['tt_version_dir'], 'val_curves'),
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
    """Nicely formatted hparams string."""
    config_files = ['data', 'model', 'train']
    print_str = ''
    for config_file in config_files:
        print_str += '\n%s CONFIG:\n' % config_file.upper()
        config_dict = yaml.safe_load(open(hparams['%s_config' % config_file]))
        for key in config_dict.keys():
            print_str += '    {}: {}\n'.format(key, hparams[key])
    print_str += '\n'
    return print_str


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

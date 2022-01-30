"""Example script for fitting daart models from the command line with test-tube package."""

import copy
import logging
import numpy as np
import os
import sys
import time
import torch

from daart.eval import plot_training_curves
from daart.io import export_hparams
from daart.testtube import get_all_params, print_hparams, create_tt_experiment, clean_tt_dir
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
    if hparams['model_class'].lower() == 'segmenter':
        from daart.models import Segmenter
        model = Segmenter(hparams)
    else:
        raise NotImplementedError
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
    if hparams.get('variational', False):
        from daart.callbacks import AnnealHparam
        callbacks.append(AnnealHparam(
            hparams=model.hparams, key='kl_weight', epoch_start=0, epoch_end=100))

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
            os.path.join(hparams['tt_version_dir'], 'metrics.csv'), dtype='train',
            expt_ids=hparams['expt_ids'],
            save_file=os.path.join(hparams['tt_version_dir'], 'train_curves'),
            format='png')
        plot_training_curves(
            os.path.join(hparams['tt_version_dir'], 'metrics.csv'), dtype='val',
            expt_ids=hparams['expt_ids'],
            save_file=os.path.join(hparams['tt_version_dir'], 'val_curves'),
            format='png')

    # get rid of unneeded logging info
    clean_tt_dir(hparams)


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

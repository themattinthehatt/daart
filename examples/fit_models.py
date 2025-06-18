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
from daart.utils import build_data_generator, collect_callbacks


def run_main(hparams, *args):

    # set return value
    ret_val = None

    try:

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
            filemode='w', level=logging.INFO,
            format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # add logging to console

        # run train model script
        train_model(hparams)

    except Exception as e:
        ret_val = e

    return ret_val


def train_model(hparams):

    # -------------------------------------
    # print hparams to console
    # -------------------------------------
    print_str = print_hparams(hparams)
    logging.info(print_str)

    # -------------------------------------
    # build data generator
    # -------------------------------------
    data_gen = build_data_generator(hparams)
    logging.info(data_gen)

    # pull class weights out of labeled training data
    if hparams.get('weight_classes', True):
        totals = data_gen.count_class_examples()
        idx_background = hparams.get('ignore_class', 0)
        if idx_background in np.arange(len(totals)):
            totals[idx_background] = 0  # get rid of background class
        # select class weights by choosing class with max labeled examples to have a value of 1;
        # the remaining weights will be inversely proportional to their prevalence. For example, a
        # class that has half as many examples as the most prevalent will be weighted twice as much
        class_weights = np.max(totals) / (totals + 1e-10)
        class_weights[totals == 0] = 0
        hparams['class_weights'] = class_weights.tolist()  # needs to be list to save out to yaml
        print('class weights: {}'.format(class_weights))
    else:
        hparams['class_weights'] = None

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
    # train model
    # -------------------------------------
    callbacks = collect_callbacks(hparams)
    trainer = Trainer(**hparams, callbacks=callbacks)
    trainer.fit(model, data_gen, save_path=hparams['tt_version_dir'])

    # update hparams upon successful training
    hparams['training_completed'] = True
    export_hparams(hparams)

    # -------------------------------------
    # export artifacts
    # -------------------------------------

    # save training curves
    if hparams.get('plot_train_curves', False):
        plot_training_curves(
            metrics_file=os.path.join(hparams['tt_version_dir'], 'metrics.csv'),
            dtype='train',
            expt_ids=hparams['expt_ids'],
            save_file=os.path.join(hparams['tt_version_dir'], 'train_curves'),
            format='png',
        )
        plot_training_curves(
            metrics_file=os.path.join(hparams['tt_version_dir'], 'metrics.csv'),
            dtype='val',
            expt_ids=hparams['expt_ids'],
            save_file=os.path.join(hparams['tt_version_dir'], 'val_curves'),
            format='png',
        )

    # run model inference on all training sessions
    if hparams['train_frac'] != 1.0:  # rebuild data generator to include all data if necessary
        hparams['train_frac'] = 1.0
        data_gen = build_data_generator(hparams)
    results_dict = model.predict_labels(data_gen)
    for sess, dataset in enumerate(data_gen.datasets):
        expt_id = dataset.id
        labels = np.vstack(results_dict['labels'][sess])
        np.save(os.path.join(hparams['tt_version_dir'], f'{expt_id}_states.npy'), labels)

    # get rid of unneeded logging info
    clean_tt_dir(hparams)


if __name__ == '__main__':

    """To run:

    (daart) $: python fit_models.py --data_config /path/to/data.yaml 
       --model_config /path/to/model.yaml --train_config /path/to/train.yaml

    For example yaml files, see the `configs` subdirectory inside the daart home directory

    NOTE: this script assumes a specific naming convention for markers and labels 
    (see daart.readthedocs.io). 
    
    """

    hyperparams = get_all_params()

    if hyperparams.device == 'cuda':
        if isinstance(hyperparams.gpus_vis, int):
            gpu_ids = [str(hyperparams.gpus_vis)]
        else:
            gpu_ids = hyperparams.gpus_vis.split(';')
        results = hyperparams.optimize_parallel_gpu(
            run_main,
            gpu_ids=gpu_ids)

    elif hyperparams.device == 'cpu':
        results = hyperparams.optimize_parallel_cpu(
            run_main,
            nb_trials=hyperparams.tt_n_cpu_trials,
            nb_workers=hyperparams.tt_n_cpu_workers)

    else:
        raise ValueError(f'Must choose "cuda" or "cpu" for device, not {hyperparams.device}')

    exitcode = 0
    for result in results:
        if result[1] is not None:
            exitcode = 1
    exit(exitcode)

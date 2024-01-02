"""Provide pytest fixtures for the entire test suite.

These fixtures create data and data modules that can be reused by other tests.

"""

import os
import pytest
import shutil
import subprocess
import torch
from typing import Callable, List, Optional
import yaml

from daart.data import DataGenerator
from daart.utils import build_data_generator


@pytest.fixture
def base_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.join(__file__)))


@pytest.fixture
def model_fitting_file(base_dir) -> str:
    return os.path.join(base_dir, 'examples', 'fit_models.py')


@pytest.fixture
def data_dir(base_dir) -> str:
    return os.path.join(base_dir, 'data')


@pytest.fixture
def config_dir(data_dir) -> str:
    return os.path.join(data_dir, 'configs')


@pytest.fixture
def config_files(config_dir) -> dict:
    base_config_files = {
        'data': os.path.join(config_dir, 'data.yaml'),
        'model': os.path.join(config_dir, 'model.yaml'),
        'train': os.path.join(config_dir, 'train.yaml'),
    }
    return base_config_files


@pytest.fixture
def hparams(config_files, data_dir) -> dict:
    """Load all example config files without test-tube."""

    hparams = {}

    for key, cfg_tmp in config_files.items():
        with open(cfg_tmp) as f:
            dict_tmp = yaml.load(f, Loader=yaml.FullLoader)
        hparams.update(dict_tmp)

    # update data dir with path to example data in github repo
    hparams['data_dir'] = data_dir

    # keep everything on cpu
    hparams['device'] = 'cpu'

    # no variational
    hparams['variational'] = False

    return hparams


@pytest.fixture
def data_generator(hparams) -> DataGenerator:

    # return to tests
    data_generator = build_data_generator(hparams)
    yield data_generator

    # clean up after all tests have run (no more calls to yield)
    del data_generator
    torch.cuda.empty_cache()


@pytest.fixture
def check_batch():

    def _check_batch(output_dict, variational):

        dtypes = output_dict.keys()
        assert 'labels' in dtypes
        assert 'labels_weak' in dtypes
        assert 'prediction' in dtypes
        assert 'task_prediction' in dtypes
        assert 'embedding' in dtypes
        assert 'latent_mean' in dtypes
        assert 'latent_logvar' in dtypes
        assert 'sample' in dtypes

        # (n_seqs, seq_len, embedding_dim)
        batch_size = output_dict['embedding'].shape[0]
        assert len(output_dict['embedding'].shape) == 3

        if output_dict['labels'] is not None:
            # (n_seqs, seq_len, n_classes)
            assert output_dict['labels'].shape[0] == batch_size
            assert len(output_dict['labels'].shape) == 3
        if output_dict['labels_weak'] is not None:
            # (n_seqs, seq_len, n_classes)
            assert output_dict['labels_weak'].shape[0] == batch_size
            assert len(output_dict['labels_weak'].shape) == 3
        if output_dict['prediction'] is not None:
            # (n_seqs, seq_len, n_markers)
            assert output_dict['prediction'].shape[0] == batch_size
            assert len(output_dict['prediction'].shape) == 3
        if output_dict['task_prediction'] is not None:
            # (n_seqs, seq_len, n_tasks)
            assert output_dict['task_prediction'].shape[0] == batch_size
            assert len(output_dict['task_prediction'].shape) == 3

        if variational:
            # we should get a logvar
            assert output_dict['latent_mean'] is not None
            # mean is different from sample
            assert ~torch.allclose(output_dict['latent_mean'], output_dict['sample'])
        else:
            # we should not get a logvar
            assert output_dict['latent_logvar'] is None
            # mean is the same as samples
            assert torch.allclose(output_dict['latent_mean'], output_dict['sample'])

    return _check_batch


@pytest.fixture
def default_config_vals(data_dir) -> dict:
    new_values = {
        'data': {
            'data_dir': data_dir,
        },
        'model': {
            'variational': False,
            'lambda_weak': 1,
            'lambda_strong': 1,
            'lambda_pred': 1,
            'lambda_recon': 1,
            'lambda_task': 0,
        },
        'train': {
            'min_epochs': 1,
            'max_epochs': 1,
            'sequence_length': 20,
            'train_frac': 0.01,
            'enable_early_stop': False,
            'plot_train_curves': False,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'tt_n_cpu_workers': 2,
        },
    }
    return new_values


@pytest.fixture
def update_config_files() -> Callable:

    def _update_config_files(config_files, new_values, save_dir=None):
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

    return _update_config_files


@pytest.fixture
def fit_model(model_fitting_file) -> Callable:

    def _fit_model_func(config_files):
        call_str = [
            'python',
            model_fitting_file,
            '--data_config', config_files['data'],
            '--model_config', config_files['model'],
            '--train_config', config_files['train'],
        ]
        code = subprocess.call(' '.join(call_str), shell=True)
        if code != 0:
            raise Exception('test-tube model fitting failed')

    return _fit_model_func

"""Provide pytest fixtures for the entire test suite.

These fixtures create data and data modules that can be reused by other tests.

"""

import os
import pytest
import shutil
import torch
from typing import Callable, List, Optional
import yaml

from daart.data import DataGenerator
from daart.utils import build_data_generator


@pytest.fixture
def hparams() -> dict:
    """Load all example config files without test-tube."""

    base_dir = os.path.dirname(os.path.dirname(os.path.join(__file__)))
    config_dir = os.path.join(base_dir, 'data', 'configs')

    keys = ['data', 'model', 'train']
    hparams = {}

    for key in keys:
        cfg_tmp = os.path.join(config_dir, '%s.yaml' % key)
        with open(cfg_tmp) as f:
            dict_tmp = yaml.load(f, Loader=yaml.FullLoader)
        hparams.update(dict_tmp)

    # update data dir with path to example data in github repo
    base_dir = os.path.dirname(os.path.dirname(os.path.join(__file__)))
    hparams['data_dir'] = os.path.join(base_dir, 'data')

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

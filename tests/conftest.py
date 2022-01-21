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
    config_dir = os.path.join(base_dir, 'configs')

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

    return hparams


@pytest.fixture
def data_generator(hparams) -> DataGenerator:

    # return to tests
    data_generator = build_data_generator(hparams)
    yield data_generator

    # clean up after all tests have run (no more calls to yield)
    del data_generator
    torch.cuda.empty_cache()

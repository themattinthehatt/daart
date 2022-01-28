import copy
import numpy as np
import pytest
import torch

from daart.models import Segmenter, Ensembler


def test_reparameterize_gaussian():

    from daart.models.base import reparameterize_gaussian

    n = 10
    m = 4

    mu = 40
    mus = mu * torch.ones((n, m))
    logvars = 0.1 * torch.ones((n, m))
    sample = reparameterize_gaussian(mus, logvars)
    assert sample.shape == (n, m)
    assert torch.isclose(torch.mean(sample), torch.tensor([mu], dtype=torch.float), 2)

    logvars2 = 10 * torch.ones((n, m))
    sample2 = reparameterize_gaussian(mus, logvars2)
    assert torch.std(sample2) > torch.std(sample)


def test_ensembler(hparams, data_generator):

    hp = copy.deepcopy(hparams)
    hp['backbone'] = 'dtcn'

    n_models = 3

    models = []
    for n in range(n_models):
        models.append(Segmenter(hp))
        models[-1].to(hp['device'])

    # initialze
    ens = Ensembler(models)

    # ensemble predictions with predefined weights
    ens.predict_labels(
        data_generator, combine_before_softmax=True, weights=np.random.rand(n_models))

    # ensemble predictions using output entropy
    ens.predict_labels(data_generator, combine_before_softmax=True, weights='entropy')

    # ensemble predictions using uniform weights
    ens.predict_labels(data_generator, combine_before_softmax=True, weights=None)

    # ensemble predictions with predefined weights
    ens.predict_labels(
        data_generator, combine_before_softmax=False, weights=np.random.rand(n_models))

    # ensemble predictions using output entropy
    ens.predict_labels(data_generator, combine_before_softmax=False, weights='entropy')

    # ensemble predictions using uniform weights
    ens.predict_labels(data_generator, combine_before_softmax=False, weights=None)

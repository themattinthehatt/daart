import copy
import numpy as np
import pytest

from daart.models import Segmenter, Ensembler


def test_ensembler(hparams, data_generator):

    hp = copy.deepcopy(hparams)
    hp['model_type'] = 'dtcn'

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

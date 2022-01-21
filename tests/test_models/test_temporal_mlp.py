import copy
import pytest

from daart.models import Segmenter
from daart.models.temporalmlp import TemporalMLP


def test_tcn(hparams, data_generator):

    hp = copy.deepcopy(hparams)
    hp['model_type'] = 'temporal-mlp'

    model = Segmenter(hp)
    model.to(hp['device'])

    # load the correct backbone
    assert isinstance(model.model, TemporalMLP)

    # print doesn't fail
    assert model.__str__()

    # process a batch of data
    batch = data_generator.datasets[0][0]
    output_dict = model(batch['markers'])
    dtypes = output_dict.keys()
    assert 'labels' in dtypes
    assert 'labels_weak' in dtypes
    assert 'prediction' in dtypes
    assert 'task_prediction' in dtypes
    assert 'embedding' in dtypes

    batch_size = output_dict['embedding'].shape[0]
    if output_dict['labels'] is not None:
        assert output_dict['labels'].shape[0] == batch_size
    if output_dict['labels_weak'] is not None:
        assert output_dict['labels_weak'].shape[0] == batch_size
    if output_dict['prediction'] is not None:
        assert output_dict['prediction'].shape[0] == batch_size
    if output_dict['task_prediction'] is not None:
        assert output_dict['task_prediction'].shape[0] == batch_size

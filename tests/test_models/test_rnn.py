import copy
import pytest
import torch

from daart.models import Segmenter
from daart.models.rnn import RNN


def test_lstm(hparams, data_generator, check_batch):

    hp = copy.deepcopy(hparams)
    hp['backbone'] = 'lstm'
    hp['variational'] = False

    model = Segmenter(hp)
    model.to(hp['device'])

    # load the correct backbone
    assert isinstance(model.model['encoder'], RNN)
    for name, layer in model.model['encoder'].model.named_children():
        assert isinstance(layer, torch.nn.LSTM)
        break

    # print doesn't fail
    assert model.__str__()

    # process a batch of data
    batch = data_generator.datasets[0][0]
    output_dict = model(batch['markers'].unsqueeze(0))  # add batch dim
    check_batch(output_dict, variational=hp['variational'])

    # compute losses
    batch_ext = {}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            batch_ext[key] = val.unsqueeze(0)  # add batch dim
        else:
            batch_ext[key] = val

    loss_dict = model.training_step(batch_ext, accumulate_grad=False)
    assert 'loss' in loss_dict


def test_gru(hparams, data_generator, check_batch):

    hp = copy.deepcopy(hparams)
    hp['backbone'] = 'gru'
    hp['variational'] = False

    model = Segmenter(hp)
    model.to(hp['device'])

    # load the correct backbone
    assert isinstance(model.model['encoder'], RNN)
    for name, layer in model.model['encoder'].model.named_children():
        assert isinstance(layer, torch.nn.GRU)
        break

    # print doesn't fail
    assert model.__str__()

    # process a batch of data
    batch = data_generator.datasets[0][0]
    output_dict = model(batch['markers'].unsqueeze(0))  # add batch dim
    check_batch(output_dict, variational=hp['variational'])

    # compute losses
    batch_ext = {}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            batch_ext[key] = val.unsqueeze(0)  # add batch dim
        else:
            batch_ext[key] = val

    loss_dict = model.training_step(batch_ext, accumulate_grad=False)
    assert 'loss' in loss_dict


def test_lstm_variational(hparams, data_generator, check_batch):

    hp = copy.deepcopy(hparams)
    hp['backbone'] = 'lstm'
    hp['variational'] = True

    model = Segmenter(hp)
    model.to(hp['device'])

    # load the correct backbone
    assert isinstance(model.model['encoder'], RNN)
    for name, layer in model.model['encoder'].model.named_children():
        assert isinstance(layer, torch.nn.LSTM)
        break

    # print doesn't fail
    assert model.__str__()

    # process a batch of data
    batch = data_generator.datasets[0][0]
    output_dict = model(batch['markers'].unsqueeze(0))  # add batch dim
    check_batch(output_dict, variational=hp['variational'])

    # compute losses
    batch_ext = {}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            batch_ext[key] = val.unsqueeze(0)  # add batch dim
        else:
            batch_ext[key] = val

    loss_dict = model.training_step(batch_ext, accumulate_grad=False)
    assert 'loss' in loss_dict


def test_gru_variational(hparams, data_generator, check_batch):

    hp = copy.deepcopy(hparams)
    hp['backbone'] = 'gru'
    hp['variational'] = True

    model = Segmenter(hp)
    model.to(hp['device'])

    # load the correct backbone
    assert isinstance(model.model['encoder'], RNN)
    for name, layer in model.model['encoder'].model.named_children():
        assert isinstance(layer, torch.nn.GRU)
        break

    # print doesn't fail
    assert model.__str__()

    # process a batch of data
    batch = data_generator.datasets[0][0]
    output_dict = model(batch['markers'].unsqueeze(0))  # add batch dim
    check_batch(output_dict, variational=hp['variational'])

    # compute losses
    batch_ext = {}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            batch_ext[key] = val.unsqueeze(0)  # add batch dim
        else:
            batch_ext[key] = val

    loss_dict = model.training_step(batch_ext, accumulate_grad=False)
    assert 'loss' in loss_dict

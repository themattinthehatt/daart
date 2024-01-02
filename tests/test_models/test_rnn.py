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


def test_lstm_testtube(tmp_path, default_config_vals, config_files, update_config_files, fit_model):

    tmp_path = str(tmp_path)
    new_vals = default_config_vals.copy()

    # define new config values
    new_vals['data']['results_dir'] = tmp_path
    new_vals['model']['backbone'] = 'lstm'

    # update config files, save in tmp_path
    config_dicts, new_config_files = update_config_files(config_files, new_vals, tmp_path)

    # fit model
    fit_model(new_config_files)


def test_gru_testtube(tmp_path, default_config_vals, config_files, update_config_files, fit_model):

    tmp_path = str(tmp_path)
    new_vals = default_config_vals.copy()

    # define new config values
    new_vals['data']['results_dir'] = tmp_path
    new_vals['model']['backbone'] = 'gru'

    # update config files, save in tmp_path
    config_dicts, new_config_files = update_config_files(config_files, new_vals, tmp_path)

    # fit model
    fit_model(new_config_files)

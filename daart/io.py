"""File IO for daart package."""

import csv
import numpy as np
import os
import pickle
from typing import List, Optional, Union
from typeguard import typechecked
import yaml


__all__ = [
    'get_subdirs',
    'get_expt_dir',
    'get_model_dir',
    'get_model_params',
    'find_experiment',
    'read_expt_info_from_csv',
    'export_expt_info_to_csv',
    'export_hparams',
    'make_dir_if_not_exists',
]


@typechecked
def get_subdirs(path: str) -> List[str]:
    """Get all first-level subdirectories in a given path (no recursion).

    Parameters
    ----------
    path : str
        absolute path

    Returns
    -------
    list
        first-level subdirectories in :obj:`path`

    """
    if not os.path.exists(path):
        raise NotADirectoryError('%s is not a path' % path)
    try:
        s = next(os.walk(path))[1]
    except StopIteration:
        raise StopIteration('%s does not contain any subdirectories' % path)
    if len(s) == 0:
        raise StopIteration('%s does not contain any subdirectories' % path)
    return s


@typechecked
def get_expt_dir(base_dir: str, expt_ids: Union[str, List[str]]) -> str:
    """Construct experiment directory given base directory and list of experiment ids.

    Parameters
    ----------
    base_dir : str
        base results directory
    expt_ids : str or list
        single experiment id (str) of a list of experiment ids that will define a "multisession"
        directory

    Returns
    -------
    str
        absolute path of experiment directory

    """
    if isinstance(expt_ids, list) and len(expt_ids) > 1:
        # multisession; see if multisession already exists; if not, create a new one
        try:
            subdirs = get_subdirs(base_dir)
        except StopIteration:
            # fresh results directory
            subdirs = []
        expt_dir = None
        max_val = -1
        for subdir in subdirs:
            if subdir[:5] == 'multi':
                # load csv containing expt_ids
                multi_sess = read_expt_info_from_csv(
                    os.path.join(base_dir, subdir, 'expt_info.csv'))
                # compare to current ids
                multi_sess = [row['expt'] for row in multi_sess]
                if sorted(multi_sess) == sorted(expt_ids):
                    expt_dir = subdir
                    break
                else:
                    max_val = np.max([max_val, int(subdir.split('-')[-1])])
        if expt_dir is None:
            expt_dir = 'multi-' + str(max_val + 1)
            # save csv with expt ids
            export_expt_info_to_csv(os.path.join(base_dir, expt_dir), expt_ids)
    else:
        if isinstance(expt_ids, list):
            expt_dir = expt_ids[0]
        else:
            expt_dir = expt_ids
    return os.path.join(base_dir, expt_dir)


@typechecked
def get_model_dir(base_dir: str, model_params: dict) -> str:
    """Helper function to construct model directory from model param dict.

    Parameters
    ----------
    base_dir : str
        base results directory
    model_params : dict
        should contain the keys `backbone` and optionally `experiment_name`

    Returns
    -------
    str
        absolute path of model directory

    """
    if model_params['model_class'] == 'segmenter':
        model_dir = model_params['backbone']
    else:
        model_dir = model_params['model_class']
    return os.path.join(base_dir, model_dir, model_params.get('experiment_name', ''))


@typechecked
def get_model_params(hparams: dict) -> dict:
    """Returns dict containing all params considered essential for defining a model of that type.

    Parameters
    ----------
    hparams : dict
        all relevant hparams for the given model type will be pulled from this dict

    Returns
    -------
    dict
        hparams dict

    """

    model_class = hparams['model_class']
    backbone = hparams['backbone']

    # start with general params
    hparams_less = {
        'model_class': model_class,
        'rng_seed_train': hparams['rng_seed_train'],
        'rng_seed_model': hparams['rng_seed_model'],
        'trial_splits': hparams['trial_splits'],
        'train_frac': hparams['train_frac'],
        'backbone': hparams['backbone'],
        'sequence_length': hparams['sequence_length'],
        'batch_size': hparams['batch_size'],
        'input_type': hparams['input_type'],
    }

    if model_class == 'segmenter':
        hparams_less['lambda_weak'] = hparams['lambda_weak']
        hparams_less['lambda_strong'] = hparams['lambda_strong']
        hparams_less['lambda_pred'] = hparams['lambda_pred']
        hparams_less['lambda_task'] = hparams.get('lambda_task', 0)
        hparams_less['variational'] = hparams.get('variational', False)
        hparams_less['semi_supervised_algo'] = hparams.get('semi_supervised_algo', None)
        if hparams_less['semi_supervised_algo'] == 'pseudo_labels':
            hparams_less['prob_threshold'] = hparams['prob_threshold']
            hparams_less['anneal_start'] = hparams['anneal_start']
            hparams_less['anneal_end'] = hparams['anneal_end']
        elif hparams_less['semi_supervised_algo'] == 'ups':
            hparams_less['prob_threshold'] = hparams['prob_threshold']
            hparams_less['variance_threshold'] = hparams['variance_threshold']
            hparams_less['anneal_start'] = hparams['anneal_start']
            hparams_less['anneal_end'] = hparams['anneal_end']

    elif model_class == 'random-forest' or model_class == 'xgboost':
        hparams_less.pop('rng_seed_train')
        hparams_less.pop('backbone')
        hparams_less.pop('sequence_length')
        hparams_less.pop('batch_size')

    else:
        raise NotImplementedError('"%s" is not a valid model class' % model_class)

    # get backbone-specific params
    if model_class == 'segmenter':
        if backbone == 'temporal-mlp':
            hparams_less['learning_rate'] = hparams['learning_rate']
            hparams_less['n_hid_layers'] = hparams['n_hid_layers']
            if hparams['n_hid_layers'] != 0:
                hparams_less['n_hid_units'] = hparams['n_hid_units']
            hparams_less['n_lags'] = hparams['n_lags']
            hparams_less['activation'] = hparams['activation']
            hparams_less['l2_reg'] = hparams['l2_reg']
        elif backbone in ['lstm', 'gru']:
            hparams_less['learning_rate'] = hparams['learning_rate']
            hparams_less['n_hid_layers'] = hparams['n_hid_layers']
            if hparams['n_hid_layers'] != 0:
                hparams_less['n_hid_units'] = hparams['n_hid_units']
            hparams_less['activation'] = hparams['activation']
            hparams_less['l2_reg'] = hparams['l2_reg']
            hparams_less['bidirectional'] = hparams['bidirectional']
        elif backbone in ['tcn', 'dtcn']:
            hparams_less['learning_rate'] = hparams['learning_rate']
            hparams_less['n_hid_layers'] = hparams['n_hid_layers']
            if hparams['n_hid_layers'] != 0:
                hparams_less['n_hid_units'] = hparams['n_hid_units']
            hparams_less['n_lags'] = hparams['n_lags']
            hparams_less['activation'] = hparams['activation']
            hparams_less['l2_reg'] = hparams['l2_reg']
            if backbone == 'dtcn':
                hparams_less['dropout'] = hparams['dropout']
        else:
            raise NotImplementedError('"%s" is not a valid backbone network' % backbone)

    return hparams_less


@typechecked
def find_experiment(
        hparams: dict, verbose: bool = False, keys_to_sweep: List[str] = []) -> List[str]:
    """Search testtube versions to find if experiment with the same hyperparameters has been fit.

    Parameters
    ----------
    hparams : dict
        needs to contain enough information to specify a test tube experiment (model + training
        parameters)
    verbose : bool
        True to print desired hparams
    keys_to_sweep : list of strs
        these can be any value

    Returns
    -------
    list

    """

    # fill out path info if not present
    if 'tt_expt_dir' in hparams:
        tt_expt_dir = hparams['tt_expt_dir']
    else:
        if 'model_dir' not in hparams:
            if 'expt_dir' not in hparams:
                hparams['expt_dir'] = get_expt_dir(hparams['results_dir'], hparams['expt_ids'])
            hparams['model_dir'] = get_model_dir(hparams['expt_dir'], hparams)
        tt_expt_dir = os.path.join(hparams['model_dir'], hparams['tt_experiment_name'])

    try:
        tt_versions = get_subdirs(tt_expt_dir)
    except StopIteration:
        # no versions yet
        return []

    # get model-specific params
    hparams_req = get_model_params(hparams)

    # remove params if we don't want a specific value
    for key in keys_to_sweep:
        del hparams_req[key]

    version_list = []
    for version in tt_versions:
        # try to load hparams
        try:

            try:
                version_file = os.path.join(tt_expt_dir, version, 'hparams.pkl')
                with open(version_file, 'rb') as f:
                    hparams_ = pickle.load(f)
            except FileNotFoundError:
                version_file = os.path.join(tt_expt_dir, version, 'hparams.yaml')
                with open(version_file, 'r') as f:
                    hparams_ = yaml.safe_load(f)

            if all([hparams_[key] == hparams_req[key] for key in hparams_req.keys()]):
                # found match - did it finish training?
                if hparams_['training_completed']:
                    version_list.append(os.path.join(tt_expt_dir, version))
                    if len(keys_to_sweep) == 0:
                        # we found the only model we're looking for
                        break
            else:
                if verbose:
                    print('unmatched keys, %s:' % version)
                    for key in hparams_req.keys():
                        if hparams_[key] != hparams_req[key]:
                            print('{}: {} vs {}'.format(key, hparams_[key], hparams_req[key]))
                    print()

        except IOError:
            # various reasons why this may fail; all mean that this version is not what we seek
            continue
        except KeyError:
            # usually occurs when checking older models against newer models with more hparams
            continue

    if len(version_list) == 0 and verbose:
        print('could not find match for requested hyperparameters: {}'.format(hparams_req))

    return version_list


@typechecked
def read_expt_info_from_csv(expt_file: str) -> List[dict]:
    """Read csv file that contains expt id info.

    Parameters
    ----------
    expt_file : str
        /full/path/to/expt_info.csv

    Returns
    -------
    list
        list of dicts with expt info

    """
    expts_multi = []
    # load and parse csv file that contains single session info
    with open(expt_file) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            expts_multi.append(dict(row))
    return expts_multi


@typechecked
def export_expt_info_to_csv(expt_dir: str, ids_list: List[str]) -> None:
    """Export list of expt ids to csv file.

    Parameters
    ----------
    expt_dir : str
        absolute path for where to save `expt_info.csv` file
    ids_list : list
        list which contains each expt name

    """
    expt_file = os.path.join(expt_dir, 'expt_info.csv')
    if not os.path.isdir(expt_dir):
        os.makedirs(expt_dir)
    with open(expt_file, mode='w') as f:
        expt_writer = csv.DictWriter(f, fieldnames=['expt'])
        expt_writer.writeheader()
        for id in ids_list:
            expt_writer.writerow({'expt': id})


@typechecked
def export_hparams(hparams: dict, filename: Optional[str] = None) -> None:
    """Export hyperparameter dictionary as a yaml file.

    Parameters
    ----------
    hparams : dict
        hyperparameter dict to export
    filename : str, optional
        filename to save hparams as; if None, filename is constructed from hparams

    """
    if filename is None:
        filename = os.path.join(hparams['tt_version_dir'], 'hparams.yaml')

    with open(filename, 'w') as f:
        yaml.dump(hparams, f)


@typechecked
def make_dir_if_not_exists(save_file: str) -> None:
    """Utility function for creating necessary dictories for a specified filename.

    Parameters
    ----------
    save_file : str
        absolute path of save file

    """
    save_dir = os.path.dirname(save_file)
    os.makedirs(save_dir, exist_ok=True)

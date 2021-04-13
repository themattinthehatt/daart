"""File IO for daart package."""

import csv
import numpy as np
import os
import pickle


def get_subdirs(path):
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


def get_expt_dir(base_dir, expt_ids):
    if isinstance(expt_ids, list) and len(expt_ids) > 1:
        # multisession; see if multisession already exists; if not, create a new one
        subdirs = get_subdirs(base_dir)
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


def get_model_dir(base_dir, model_params):
    model_dir = model_params['model_type']
    return os.path.join(base_dir, model_dir, model_params.get('experiment_name', ''))


def get_model_params(hparams):
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

    model_type = hparams['model_type']

    # start with general params
    hparams_less = {
        'rng_seed_train': hparams['rng_seed_train'],
        'rng_seed_model': hparams['rng_seed_model'],
        'trial_splits': hparams['trial_splits'],
        'train_frac': hparams['train_frac'],
        'model_type': hparams['model_type'],
        'batch_size': hparams['batch_size'],
        'lambda_weak': hparams['lambda_weak'],
        'lambda_strong': hparams['lambda_strong'],
        'lambda_pred': hparams['lambda_pred'],
    }

    if model_type == 'temporal-mlp':
        hparams_less['learning_rate'] = hparams['learning_rate']
        hparams_less['n_hid_layers'] = hparams['n_hid_layers']
        if hparams['n_hid_layers'] != 0:
            hparams_less['n_hid_units'] = hparams['n_hid_units']
        hparams_less['n_lags'] = hparams['n_lags']
        hparams_less['activation'] = hparams['activation']
        hparams_less['l2_reg'] = hparams['l2_reg']
    elif model_type in ['lstm', 'gru']:
        hparams_less['learning_rate'] = hparams['learning_rate']
        hparams_less['n_hid_layers'] = hparams['n_hid_layers']
        if hparams['n_hid_layers'] != 0:
            hparams_less['n_hid_units'] = hparams['n_hid_units']
        hparams_less['activation'] = hparams['activation']
        hparams_less['l2_reg'] = hparams['l2_reg']
        hparams_less['bidirectional'] = hparams['bidirectional']
    elif model_type == 'tcn':
        hparams_less['learning_rate'] = hparams['learning_rate']
        hparams_less['n_hid_layers'] = hparams['n_hid_layers']
        if hparams['n_hid_layers'] != 0:
            hparams_less['n_hid_units'] = hparams['n_hid_units']
        hparams_less['n_lags'] = hparams['n_lags']
        hparams_less['activation'] = hparams['activation']
        hparams_less['l2_reg'] = hparams['l2_reg']
    else:
        raise NotImplementedError('"%s" is not a valid model type' % model_type)

    return hparams_less


def find_experiment(hparams):
    """Search testtube versions to find if experiment with the same hyperparameters has been fit.

    Parameters
    ----------
    hparams : dict
        needs to contain enough information to specify a test tube experiment (model + training
        parameters)

    Returns
    -------
    variable
        - int if experiment is found
        - None if experiment is not found

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
        return None

    # get model-specific params
    hparams_less = get_model_params(hparams)
    found_match = False
    version = None
    for version in tt_versions:
        # load hparams
        version_file = os.path.join(tt_expt_dir, version, 'hparams.pkl')
        try:
            with open(version_file, 'rb') as f:
                hparams_ = pickle.load(f)
            if all([hparams_[key] == hparams_less[key] for key in hparams_less.keys()]):
                # found match - did it finish training?
                if hparams_['training_completed']:
                    found_match = True
                    break
            # else:
            #     for key in hparams_less.keys():
            #         if hparams_[key] != hparams_less[key]:
            #             print('{} : {}'.format(hparams_[key], hparams_less[key]))
        except IOError:
            continue

    if found_match:
        return int(version.split('_')[-1])
    else:
        return None


def read_expt_info_from_csv(expt_file):
    """Read csv file that contains expt id info.

    Parameters
    ----------
    expt_file : str
        /full/path/to/expt_info.csv

    Returns
    -------
    list
        list of expts

    """
    expts_multi = []
    # load and parse csv file that contains single session info
    with open(expt_file) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            expts_multi.append(dict(row))
    return expts_multi


def export_expt_info_to_csv(expt_dir, ids_list):
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


def export_hparams(hparams, filename=None):
    """Export hyperparameter dictionary as a pickle file.

    Parameters
    ----------
    hparams : dict
        hyperparameter dict to export
    filename : str, optional
        filename to save hparams as; if None, filename is constructed from hparams

    """
    if filename is None:
        filename = os.path.join(hparams['tt_version_dir'], 'hparams.pkl')

    with open(filename, 'wb') as f:
        pickle.dump(hparams, f)


def make_dir_if_not_exists(save_file):
    """Utility function for creating necessary dictories for a specified filename.

    Parameters
    ----------
    save_file : str
        absolute path of save file

    """
    save_dir = os.path.dirname(save_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

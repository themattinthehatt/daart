"""Loop over multiple model types/sets of data."""

import argparse
import itertools
import numpy as np
import os
import shutil
import subprocess
import yaml

from daart_utils.paths import data_path, config_path, results_path

# assumes `fit_models_loop.py` and `fit_models.py` are in the same directory
grid_search_file = '/home/bsb2144/daart/examples/fit_models.py'


def run_main(args):
    print('here')
    if args.dataset == 'fish':
        from daart_utils.session_ids.fish import SESS_IDS_TRAIN_10 as sess_ids_list
    elif args.dataset == 'fly' or args.dataset == 'fly-4' or args.dataset == 'fly-5':
        from daart_utils.session_ids.fly import SESS_IDS_TRAIN_5 as sess_ids_list
    elif args.dataset == 'ibl':
        from daart_utils.session_ids.ibl import SESS_IDS_TRAIN_5 as sess_ids_list
    elif args.dataset == 'mouse-oft-aligned' \
            or args.dataset == 'mouse-oft-aligned-new' \
            or args.dataset == 'mouse-oft':
        from daart_utils.session_ids.oft import SESS_IDS_TRAIN_10 as sess_ids_list
    elif args.dataset == 'resident-intruder':
        from daart_utils.session_ids.resident_intruder import SESS_IDS_TRAIN_6 as sess_ids_list
    elif args.dataset == 'calms21':
        from daart_utils.session_ids.calms21 import SESS_IDS_TRAIN_10 as sess_ids_list
    else:
        raise NotImplementedError('"%s" is an invalid dataset' % args.dataset)

    # set config file paths
    config_files = {
        'data': args.data,
        'model': args.model,
        'train': args.train
    }
    configs_to_update = ['data', 'model', 'train']

    # get list of models
    model_types = []
    if args.fit_mlp:
        model_types.append('temporal-mlp')
    if args.fit_dtcn:
        model_types.append('dtcn')
    if args.fit_lstm:
        model_types.append('lstm')
    if args.fit_gru:
        model_types.append('gru')
    if args.fit_transformer:
        model_types.append('transformer')
    if args.fit_rf:
        model_types.append('random-forest')
    if args.fit_xgb:
        model_types.append('xgboost')
    
    if args.fit_tcn:
        model_types.append('dtcn')
    if args.fit_rsldsm:
        model_types.append('rsldsm')
    if args.fit_gmdgm:
        model_types.append('gmdgm')

    sess_ids = sess_ids_list[0]
    pre = args.pre
    fracs = args.fracs.split(';')

    for model_type in model_types:

        # create temporary config files (will be updated each iteration, then deleted)
        for config in configs_to_update:
            dirname = os.path.dirname(config_files[config])
            filename = os.path.basename(config_files[config]).split('.')[0]
            tmp_file = os.path.join(
                dirname, '%s_tmp_%s_%s_1.yaml' % (filename, args.dataset, model_type))
            shutil.copy(config_files[config], tmp_file)
            config_files[config] = tmp_file

        # get input type for tt expt directory names
        config_data = yaml.safe_load(open(config_files['data']))
        input_type = config_data['input_type']

        for frac in fracs:

            # find sessions in which to keep hand labels
            # if args.dataset == 'mouse-oft-aligned' or args.dataset == 'mouse-oft':
            #     # for mouse-oft dataset, where all behaviors are labeled in all videos,
            #     # automatically remove labels from final 5 sessions, then subsample labels from
            #     # first sessions (generally 5 if using SESS_IDS_TRAIN_10)
            #     sess_ids_tmp = sess_ids[:5]
            #     # get total number of sessions from "frac"
            #     n_sessions = int(np.ceil(float(frac) * len(sess_ids_tmp)))
            #     # list out all possible combinations of n_sessions from sess_ids
            #     sess_to_keep_list = list(itertools.combinations(sess_ids_tmp, n_sessions))
            if args.dataset == 'resident-intruder-old':
                # for old resident-intruder dataset, where not all videos have hand labels,
                # subsample labels from initial sessions (generally 3 if using SESS_IDS_TRAIN_9)
                # and keep all videos (but not labels) from the remaining sessions
                sess_ids_tmp = sess_ids[:3]
                # get total number of sessions from "frac"
                n_sessions = int(np.ceil(float(frac) * len(sess_ids_tmp)))
                # list out all possible combinations of n_sessions from sess_ids
                sess_to_keep_list = list(itertools.combinations(sess_ids_tmp, n_sessions))
            else:
                # get total number of sessions from "frac"
                n_sessions = int(float(frac) * len(sess_ids))
                # list out all possible combinations of n_sessions from sess_ids
                sess_to_keep_list = list(itertools.combinations(sess_ids, n_sessions))
                # print(sess_to_keep_list)

            if len(sess_to_keep_list) == 0:
                print('warning! a provided fraction resulted in zero sessions to keep')
                continue

            # shuffle
            np.random.seed(5)
            #np.random.shuffle(sess_to_keep_list)

            # loop over a subset of all possible combinations
            for s in range(args.n_samples):

                if s >= len(sess_to_keep_list):
                    # for 5 choose 5, for example
                    break

                sess_to_keep = sess_to_keep_list[s]

                # print(sess_ids)
                # print(list(sess_to_keep))
                # modify configs
                update_config(
                    config_files['model'], 'backbone', 'dtcn')
                update_config(
                    config_files['data'], 'expt_ids', sess_ids)
                update_config(
                    config_files['data'], 'data_dir', os.path.join(data_path, args.dataset))
                update_config(
                    config_files['data'], 'results_dir', os.path.join(results_path, args.dataset))
                update_config(
                    config_files['data'], 'expt_ids_to_keep', list(sess_to_keep))
                update_config(
                    config_files['model'], 'tt_experiment_name',
                    '%s-%i-good_sample-%i_%s' % (pre, n_sessions, s, input_type))
                
                # check if tcn
                if model_type == 'dtcn':
                    update_config(
                    config_files['model'], 'model_class', 'segmenter')
                    
                elif model_type == "rsldsm":
                    update_config(
                    config_files['model'], 'model_class', 'rslds_marginal')
                    # check for linear vs nonlinear
                    if args.linear:
                        update_config(config_files['model'], 'backbone_dynamic', "linear")
                        update_config(config_files['model'], 'backbone_transition', "linear")
                    else:
                        update_config(config_files['model'], 'backbone_dynamic', "mlp")
                        update_config(config_files['model'], 'backbone_transition', "mlp")
                    
                elif model_type == "gmdgm":
                    update_config(
                    config_files['model'], 'model_class', 'gmdgm')
                    # check for temporal/non temporal
                    if args.temporal:
                        update_config(config_files['model'], 'temporal_inference', True)
                    else:
                        update_config(config_files['model'], 'temporal_inference', False)

                elif model_type == 'random-forest' or model_type == 'xgboost':
                    # default to simba features for tree-based models
                    # input_type = 'features-simba'
                    # input_type = 'features-sturman'
                    # input_type = 'features-treba-simba'
                    # input_type = 'markers'
                    #input_type = 'features-aligned'
                    #update_config(config_files['data'], 'input_type', input_type)
                    #update_config(config_files['train'], 'device', 'cpu')
                    update_config(config_files['model'], 'lambda_weak', 0)
                    update_config(config_files['model'], 'lambda_pred', 0)
                    update_config(config_files['model'], 'lambda_task', 0)
                    update_config(
                        config_files['model'], 'model_class', model_type)
                    args.grid_search_file = '/home/bsb2144/daart_utils/scripts/fit_models.py'

                call_str = [
                    'python',
                    args.grid_search_file,
                    '--data_config', config_files['data'],
                    '--model_config', config_files['model'],
                    '--train_config', config_files['train']
                ]
                subprocess.call(' '.join(call_str), shell=True)

    for config in configs_to_update:
        os.remove(config_files[config])


def update_config(file, key, value):

    # load yaml file as dict
    config = yaml.safe_load(open(file))

    # update value
    config[key] = value

    # resave file
    with open(file, 'w') as f:
        yaml.dump(config, f)


if __name__ == '__main__':
                          
    """To fit, for example, dtcn models on the fly data:

    (daart) $: python fit_models_subsample_loop.py --dataset fly --fit_dtcn --fracs '0.2;0.4;0.6;0.8;1.0' --n_samples 5

    The details of the hyperparameter search will be defined in the user config files.

    defaults for other datasets:
    
    --dataset fly-4 --fit_dtcn --fracs '0.2;0.4;0.6;0.8;1.0' --n_samples 5
    --dataset fish --fracs '0.2;0.4;0.6;0.8;1.0' --n_samples 5 
    --dataset ibl --fracs '0.2;0.4;0.6;0.8;1.0' --n_samples 5
    --dataset mouse-oft-aligned --fracs '0.1;0.2;0.3;0.4;0.5' --n_samples 5
    --dataset resident-intruder-old --fracs '0.3;0.6;0.9' --n_samples 3
    --dataset resident-intruder --fracs '0.2;0.4;0.6;0.8;0.9' --n_samples 3
    --dataset calms21 --fracs '0.2;0.4;0.6;0.8;1.0' --n_samples 5
    
    """

    parser = argparse.ArgumentParser()

    # define dataset to fit:
    # 'fish' | 'fly' | 'fly-4' | 'ibl' | 'mouse-oft' | 'mouse-oft-aligned' | 'resident-intruder'
    parser.add_argument('--dataset')

    # define fractions of training data, separated by a semi-colon, i.e. '0.2;0.4;1.0'
    parser.add_argument('--fracs')

    # number of samples for each fraction
    parser.add_argument('--n_samples', default=1, type=int)
    
    # results dir prefix
    parser.add_argument('--pre', default="", type=str)
    
    parser.add_argument('--linear', action='store_true', default=False)
    parser.add_argument('--temporal', action='store_true', default=False)
    

    # define models to run
    parser.add_argument('--fit_mlp', action='store_true', default=False)
    parser.add_argument('--fit_lstm', action='store_true', default=False)
    parser.add_argument('--fit_gru', action='store_true', default=False)
    parser.add_argument('--fit_dtcn', action='store_true', default=False)
    parser.add_argument('--fit_transformer', action='store_true', default=False)
    parser.add_argument('--fit_rf', action='store_true', default=False)
    parser.add_argument('--fit_xgb', action='store_true', default=False)
    
    parser.add_argument('--fit_tcn', action='store_true', default=False)
    parser.add_argument('--fit_rsldsm', action='store_true', default=False)
    parser.add_argument('--fit_gmdgm', action='store_true', default=False)
    
    parser.add_argument('--data', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--train', type=str)
    
    parser.add_argument('--grid_search_file', type=str, default=grid_search_file)

    namespace, _ = parser.parse_known_args()
    run_main(namespace)
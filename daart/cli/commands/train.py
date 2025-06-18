"""Command to train a model."""

import datetime
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

from daart.cli.types import config_file, output_dir
from daart.eval import plot_training_curves
from daart.io import export_hparams, load_config
from daart.train import Trainer
from daart.utils import build_data_generator, collect_callbacks

_logger = logging.getLogger('DAART.CLI.TRAIN')


def register_parser(subparsers):
    """Register the train command parser."""

    parser = subparsers.add_parser(
        'train',
        description='Train a neural network model on feature data.',
        usage='daart train --config <config_path> [options]',
    )

    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--config', '-c',
        type=config_file,
        required=True,
        help='Path to model configuration file (YAML)',
    )

    # Optional arguments
    optional = parser.add_argument_group('options')
    optional.add_argument(
        '--output', '-o',
        type=output_dir,
        help='Directory to save model outputs (default: ./runs/YYYY-MM-DD/HH-MM-SS)',
    )
    optional.add_argument(
        '--data', '-d',
        type=Path,
        help='Override data directory specified in config',
    )
    optional.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='Device to use for training (default: cuda)',
    )
    optional.add_argument(
        '--cpu-workers',
        type=int,
        default=4,
        help='Number of CPU workers for parallel training (default: 4)',
    )
    optional.add_argument(
        '--cpu-trials',
        type=int,
        default=1,
        help='Number of CPU trials for parallel training (default: 1)',
    )
    optional.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility',
    )
    optional.add_argument(
        '--overrides',
        nargs='*',
        metavar='KEY=VALUE',
        help='Override specific config values (format: key=value)',
    )


def handle(args):
    """Handle the train command execution."""

    # Load config
    hparams = load_config(args.config)

    # Determine output directory
    if not args.output:
        now = datetime.datetime.now()
        args.output = Path(hparams['results_dir']) \
            / now.strftime('%Y-%m-%d') / now.strftime('%H-%M-%S')

    args.output.mkdir(parents=True, exist_ok=True)

    # Convert to dict if needed
    if not isinstance(hparams, dict):
        hparams = vars(hparams)

    # Apply command line overrides
    if args.data:
        hparams['data_dir'] = str(args.data)
    if args.device:
        hparams['device'] = args.device
    if args.seed:
        hparams['rng_seed_model'] = args.seed
    if args.cpu_workers:
        hparams['tt_n_cpu_workers'] = args.cpu_workers
    if args.cpu_trials:
        hparams['tt_n_cpu_trials'] = args.cpu_trials

    # Apply custom overrides
    if args.overrides:
        hparams = apply_config_overrides(hparams, args.overrides)

    # Set output directory
    hparams['tt_version_dir'] = str(args.output)

    # Set up experiment IDs if not present
    if 'expt_ids' not in hparams:
        hparams['expt_ids'] = ['default']
    elif isinstance(hparams['expt_ids'], str):
        hparams['expt_ids'] = hparams['expt_ids'].split(';')

    _logger.info(f'Output directory: {args.output}')
    _logger.info(f'Device: {hparams["device"]}')

    # Handle parallel training based on device
    if hparams['device'] == 'cuda':
        results = run_parallel_gpu_training(hparams)

    elif hparams['device'] == 'cpu':
        results = run_parallel_cpu_training(
            hparams,
            nb_trials=hparams.get('tt_n_cpu_trials', 1),
            nb_workers=hparams.get('tt_n_cpu_workers', 4)
        )
    else:
        raise ValueError(f'Must choose "cuda" or "cpu" for device, not {hparams["device"]}')

    # Check for training errors
    exitcode = 0
    for result in results:
        if result[1] is not None:
            _logger.error(f'Training failed with error: {result[1]}')
            exitcode = 1

    if exitcode == 0:
        _logger.info('Training completed successfully')
    else:
        _logger.error('Training completed with errors')
        sys.exit(exitcode)


def apply_config_overrides(config, overrides):
    """Apply command line overrides to config."""
    for override in overrides:
        if '=' not in override:
            raise ValueError(f"Override must be in format 'key=value', got: {override}")

        key, value = override.split('=', 1)

        # Try to convert value to appropriate type
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        elif value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string

        config[key] = value

    return config


def run_main(hparams, *args):
    """Main training function (adapted from your original code)."""
    ret_val = None

    try:
        if not isinstance(hparams, dict):
            hparams = vars(hparams)

        # Set up error logging
        log_file = os.path.join(hparams['tt_version_dir'], 'console.log')
        logger = logging.getLogger('DAART.CLI.TRAIN')
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            logger.addHandler(logging.FileHandler(log_file))

        # Run train model script
        train_model(hparams)

    except Exception as e:
        _logger.error(f'Training failed: {e}', exc_info=True)
        ret_val = e

    return ret_val


def train_model(hparams):
    """Core training logic (adapted from your original code)."""

    # -------------------------------------
    # print hparams to console
    # -------------------------------------
    _logger.info("Training configuration:")
    for key, value in hparams.items():
        _logger.info(f"  {key}: {value}")

    # -------------------------------------
    # build data generator
    # -------------------------------------
    data_gen = build_data_generator(hparams)
    _logger.info(f"Data generator: {data_gen}")

    # Handle class weights
    if hparams.get('weight_classes', True):
        totals = data_gen.count_class_examples()
        idx_background = hparams.get('ignore_class', 0)
        if idx_background in np.arange(len(totals)):
            totals[idx_background] = 0
        # select class weights by choosing class with max labeled examples to have a value of 1;
        # the remaining weights will be inversely proportional to their prevalence. For example, a
        # class that has half as many examples as the most prevalent will be weighted twice as much
        class_weights = np.max(totals) / (totals + 1e-10)
        class_weights[totals == 0] = 0
        hparams['class_weights'] = class_weights.tolist()  # needs to be list to save out to yaml
        _logger.info(f'Class weights: {class_weights}')
    else:
        hparams['class_weights'] = None

    # -------------------------------------
    # build model
    # -------------------------------------
    torch.manual_seed(hparams.get('rng_seed_model', 0))
    if hparams['model_class'].lower() == 'segmenter':
        from daart.models import Segmenter
        model = Segmenter(hparams)
    else:
        raise NotImplementedError(f"Model class {hparams['model_class']} not implemented")

    model.to(hparams['device'])
    _logger.info(f"Model: {model}")

    # -------------------------------------
    # train model
    # -------------------------------------
    callbacks = collect_callbacks(hparams)
    trainer = Trainer(**hparams, callbacks=callbacks)
    trainer.fit(model, data_gen, save_path=hparams['tt_version_dir'])

    # Update hparams upon successful training
    hparams['training_completed'] = True
    export_hparams(hparams)

    # Export artifacts
    export_training_artifacts(hparams, model, data_gen)

    # Run model inference on additional sessions if desired
    if hparams.get('eval_dir'):
        export_model_predictions(model_dir=hparams['tt_version_dir'], eval_dir=hparams['eval_dir'])


def export_training_artifacts(hparams, model, data_gen):
    """Export training artifacts (plots, predictions, etc.)."""

    # Save training curves
    if hparams.get('plot_train_curves', False):
        metrics_file = os.path.join(hparams['tt_version_dir'], 'metrics.csv')

        plot_training_curves(
            metrics_file=metrics_file,
            dtype='train',
            expt_ids=hparams['expt_ids'],
            save_file=os.path.join(hparams['tt_version_dir'], 'train_curves'),
            format='png',
        )
        plot_training_curves(
            metrics_file=metrics_file,
            dtype='val',
            expt_ids=hparams['expt_ids'],
            save_file=os.path.join(hparams['tt_version_dir'], 'val_curves'),
            format='png',
        )

    # Run model inference on all training sessions
    if hparams['train_frac'] != 1.0:
        # Rebuild data generator to include all data
        hparams_full = hparams.copy()
        hparams_full['train_frac'] = 1.0
        data_gen = build_data_generator(hparams_full)

    results_dict = model.predict_labels(data_gen)
    for sess, dataset in enumerate(data_gen.datasets):
        expt_id = dataset.id
        labels = np.vstack(results_dict['labels'][sess])
        output_file = os.path.join(hparams['tt_version_dir'], f'{expt_id}_states.npy')
        np.save(output_file, labels)
        _logger.info(f'Saved predictions to {output_file}')


def export_model_predictions(model_dir, eval_dir):
    if not Path(eval_dir).is_dir():
        _logger.error(f'{eval_dir} is not a directory; aborting')
        return
    from daart.api.model import Model
    model = Model.from_dir(model_dir)
    eval_dir = Path(eval_dir)
    expt_files = list(eval_dir.glob('*.csv'))
    _logger.info(f'Evaluating model on {len(expt_files)} sessions in {eval_dir}')
    for expt_file in expt_files:
        if not expt_file.is_file():
            _logger.error(f'{expt_file} does not exist; skipping')
            continue
        expt_id = expt_file.stem
        output_file = os.path.join(model_dir, f'{expt_id}_states.npy')
        if os.path.exists(output_file):
            _logger.info(f'{output_file} already exists; skipping')
            continue
        model.predict(expt_file, output_file, expt_id)
        _logger.info(f'Saved predictions to {output_file}')


def run_parallel_gpu_training(hparams: dict):
    """Run parallel training on GPUs (placeholder - implement based on your test-tube setup)."""
    # This would need to be implemented based on how test-tube handles parallel GPU training
    # For now, just run single training
    _logger.info(f'Running training on GPU(s): {hparams["gpus_vis"]}')
    result = run_main(hparams)
    return [(0, result)]


def run_parallel_cpu_training(hparams, nb_trials=1, nb_workers=4):
    """Run parallel training on CPUs (placeholder - implement based on your test-tube setup)."""
    # This would need to be implemented based on how test-tube handles parallel CPU training
    # For now, just run single training
    _logger.info(f'Running training on CPU with {nb_workers} workers, {nb_trials} trials')
    result = run_main(hparams)
    return [(0, result)]

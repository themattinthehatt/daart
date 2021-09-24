"""Utility functions for daart package."""

# to ignore imports for sphix-autoapidoc
__all__ = ['compute_batch_pad']


def compute_batch_pad(hparams):
    """Compute padding needed to account for convolutions.

    Parameters
    ----------
    hparams : dict
        contains model architecture type and hyperparameter info (lags, n_hidden_layers, etc)

    Returns
    -------
    int
        amount of padding that needs to be added to beginning/end of each batch

    """

    if hparams['model_type'].lower() == 'temporal-mlp':
        pad = hparams['n_lags']
    elif hparams['model_type'].lower() == 'tcn':
        pad = (2 ** hparams['n_hid_layers']) * hparams['n_lags']
    elif hparams['model_type'].lower() == 'dtcn':
        pad = (2 ** hparams['n_hid_layers']) * hparams['n_lags']
    elif hparams['model_type'].lower() in ['lstm', 'gru']:
        pad = 0
    elif hparams['model_type'].lower() == 'tgm':
        raise NotImplementedError
    else:
        raise ValueError('"%s" is not a valid model type' % hparams['model_type'])

    return pad

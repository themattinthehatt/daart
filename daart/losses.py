"""Custom losses for PyTorch models."""

import numpy as np
import torch
from typeguard import typechecked

# to ignore imports for sphix-autoapidoc
__all__ = ['kl_div_to_std_normal']


@typechecked
def kl_div_to_std_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute element-wise KL(q(z) || N(0, 1)) where q(z) is a normal parameterized by mu, logvar.

    Parameters
    ----------
    mu : torch.Tensor
        mean parameter of shape (n_sequences, sequence_length, n_dims)
    logvar : torch.Tensor
        log variance parameter of shape (n_sequences, sequence_length, n_dims)

    Returns
    -------
    torch.Tensor
        KL divergence summed across dims, averaged across batch

    """
    kl = 0.5 * torch.sum(logvar.exp() - logvar + mu.pow(2) - 1, dim=-1)
    return torch.mean(kl)

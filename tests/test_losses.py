import torch

from daart import losses


def test_kl_div_to_std_normal():

    mu = torch.zeros(1, 1)
    logvar = torch.zeros(1, 1)
    kl = losses.kl_div_to_std_normal(mu, logvar)
    assert kl == 0

    mu = torch.ones(1, 1)
    logvar = torch.zeros(1, 1)
    kl = losses.kl_div_to_std_normal(mu, logvar)
    assert kl > 0

    mu = torch.zeros(1, 1)
    logvar = torch.ones(1, 1)
    kl = losses.kl_div_to_std_normal(mu, logvar)
    assert kl > 0

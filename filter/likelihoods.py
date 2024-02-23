import math
import torch
import torch.nn as nn
import filter.utils as utils
import torch.nn.functional as Fn
import torch.distributions as dist


class GaussianLikelihood(nn.Module):
    def __init__(self, readout_fn, n_neurons, R=None, device='cpu'):
        super(GaussianLikelihood, self).__init__()

        self.n_neurons = n_neurons
        self.readout_fn = readout_fn

        if R is None:
            self.log_R = torch.nn.Parameter(utils.softplus_inv(torch.ones(n_neurons).to(device)))
        else:
            self.log_R = torch.nn.Parameter(utils.softplus_inv(R))

    def get_ell(self, y, z):
        mean = self.readout_fn(z)
        cov = Fn.softplus(self.log_R)

        pdf = torch.distributions.Normal(mean, torch.sqrt(cov))
        log_p_y = pdf.log_prob(y).sum(dim=-1)

        return log_p_y


class PoissonLikelihood(nn.Module):
    def __init__(self, readout_fn, n_neurons, delta, device='cpu', encode_p_mask=0.00):
        super(PoissonLikelihood, self).__init__()
        self.delta = delta
        self.device = device
        self.n_neurons = n_neurons
        self.readout_fn = readout_fn
        self.encode_p_mask = encode_p_mask

    def get_ell(self, y, z):
        log_exp = math.log(self.delta) + self.readout_fn(z) # C @ z
        log_p_y = -torch.nn.functional.poisson_nll_loss(log_exp, y, full=True, reduction='none')
        return log_p_y.sum(dim=-1)



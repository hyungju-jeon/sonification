import math
import torch
import torch.nn as nn
import filter.utils as utils
import torch.nn.functional as Fn
import torch.distributions as dist


class GaussianLikelihood(nn.Module):
    def __init__(self, readout_fn, n_neurons, R=None, device="cpu"):
        super(GaussianLikelihood, self).__init__()

        self.n_neurons = n_neurons
        self.readout_fn = readout_fn

        if R is None:
            self.log_R = torch.nn.Parameter(
                utils.softplus_inv(torch.ones(n_neurons).to(device))
            )
        else:
            self.log_R = torch.nn.Parameter(utils.softplus_inv(R))

    def get_ell(self, y, z):
        mean = self.readout_fn(z)
        cov = Fn.softplus(self.log_R)

        pdf = torch.distributions.Normal(mean, torch.sqrt(cov))
        log_p_y = pdf.log_prob(y).sum(dim=-1)

        return log_p_y


class PoissonLikelihood(nn.Module):
    def __init__(self, readout_fn, n_neurons, delta, device="cpu", encode_p_mask=0.00):
        super(PoissonLikelihood, self).__init__()
        self.delta = delta
        self.device = device
        self.n_neurons = n_neurons
        self.readout_fn = readout_fn
        self.encode_p_mask = encode_p_mask

    def get_ell(self, y, z):
        log_exp = math.log(self.delta) + self.readout_fn(z)  # C @ z
        log_p_y = -torch.nn.functional.poisson_nll_loss(
            log_exp, y, full=True, reduction="none"
        )
        return log_p_y.sum(dim=-1)


class LinearPolarToCartesian(nn.Module):
    def __init__(self, dim_in, dim_out, n_pairs, loading, bias, device="cpu"):
        super(LinearPolarToCartesian, self).__init__()

        self.device = device
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_pairs = n_pairs

        self.linear = nn.Linear(dim_in, dim_out, bias=True, device=device)
        self.linear.weight.data = torch.tensor(loading.T).type(torch.float32)
        self.linear.bias.data = torch.tensor(bias).type(torch.float32)

    def forward(self, z_polar):
        z_cart = torch.zeros_like(z_polar)

        for i in range(self.n_pairs):
            z_cart[:, :, 2 * i] = z_polar[:, :, 2 * i] * torch.cos(
                z_polar[:, :, 2 * i + 1]
            )
            z_cart[:, :, 2 * i + 1] = z_polar[:, :, 2 * i] * torch.sin(
                z_polar[:, :, 2 * i + 1]
            )
        out = self.linear(z_cart)
        return out

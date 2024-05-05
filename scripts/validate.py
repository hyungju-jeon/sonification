import numpy as np
import os
import random
# %%

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from numpy import random
from tqdm import tqdm
from scipy.linalg import block_diag

from filter.approximations import DenseGaussianApproximations
from filter.dynamics import DenseGaussianInitialCondition, DenseGaussianNonlinearDynamics
from filter.encoders import LocalEncoderLRMvn
from filter.likelihoods import PoissonLikelihood
from filter.nonlinear_smoother import FullRankNonlinearStateSpaceModelFilter, NonlinearFilter
from utils.ndlib.dslib import *
from utils.ndlib.dynlib import *

from utils import plotting

from utils.reconstruction_utils import *


dt = 1e-3
num_neurons = 50

CYCLE_FAST = {
    "x0": np.array([0.5, 0]),
    "d": 1,
    "w": 2 * np.pi * 1.5,
    "Q": None,
    "dt": dt,
}
CYCLE_SLOW = {
    "x0": np.array([0.5, 0]),
    "d": 1,
    "w": 2 * np.pi * 0.2,
    "Q": None,
    "dt": dt,
}

loading_matrix_slow_name = "./data/loading_matrix_slow.npz"

param = np.load(loading_matrix_slow_name, allow_pickle=True)
C_slow, b_slow = block_diag(*param["C"]), param["b"].flatten()

loading = C_slow
b = b_slow


def validate():

    bin_sz = 20e-3
    #device = "cuda:0"
    device = "cpu"
    data_device = "cpu"
    bin_sz_ms = int(bin_sz * 1e3)

    default_dtype = torch.float32
    torch.set_default_dtype(default_dtype)

    """hyperparameters"""
    n_inputs = 2
    n_latents = 4
    n_hidden_current_obs = 128
    n_samples = 50
    rank_y = n_latents

    batch_sz = 64
    n_epochs = 500
    blues = cm.get_cmap("Blues", n_samples)

    """data params"""
    n_trials = 250
    n_neurons = 100
    n_time_bins = 1000

    B = torch.nn.Linear(n_inputs, n_latents, bias=False, device=device).requires_grad_(False)

    # Defines how the inputs (u) affects the perturb_cycle
    B.weight.data = torch.tensor([[0, 0], [0, 0], [1, 0], [0, 1]], device=device).type(default_dtype) * 1e-3

    Q_0_diag = torch.ones(n_latents, device=device).requires_grad_(False) * 1e-2
    Q_diag = torch.ones(n_latents, device=device).requires_grad_(False) * 1e-2
    R_diag = torch.ones(n_neurons, device=device).requires_grad_(False)
    m_0 = torch.tensor([0.5, 0, 0.5, 0], device=device).requires_grad_(False).type(default_dtype)

    """generate input and latent/observations"""
    u = torch.zeros((n_trials, n_time_bins, n_inputs), device=device)
    y_gt = torch.zeros((n_trials, n_time_bins, n_neurons), device=device)
    z_gt = torch.zeros((n_trials, n_time_bins, n_latents), device=device)

    for i in range(n_trials):
        y_gt[i], z_gt[i], u[i] = generate_sample(n_time_bins)

    y_test_dataset = torch.utils.data.TensorDataset(
        y_gt,
        u,
        z_gt,
    )
    test_dataloader = torch.utils.data.DataLoader(y_test_dataset, batch_size=batch_sz, shuffle=True)

    """approximation pdf"""
    approximation_pdf = DenseGaussianApproximations(n_latents, device)

    """likelihood pdf"""
    C = torch.nn.Linear(n_latents, n_neurons, bias=True, device=device).requires_grad_(False)
    C.weight.data = torch.tensor(loading.T, device=device).type(torch.float32)
    C.bias.data = torch.tensor(b, device=device).type(torch.float32)

    likelihood_pdf = PoissonLikelihood(C, n_neurons, delta=bin_sz_ms, device=device)

    """dynamics module"""

    def A(x):
        Ax = torch.zeros_like(x, device=device)
        for i in range(2):
            r = torch.sqrt(x[:, :, 2 * i] ** 2 + x[:, :, 2 * i + 1] ** 2)
            theta = torch.atan2(x[:, :, 2 * i + 1], x[:, :, 2 * i])

            r_new = r + r * (1 - r) * bin_sz
            theta_new = theta + 2 * np.pi * 0.2 * bin_sz

            Ax[:, :, 2 * i] = r_new * torch.cos(theta_new)
            Ax[:, :, 2 * i + 1] = r_new * torch.sin(theta_new)
        return Ax

    dynamics_fn = A
    dynamics_mod = DenseGaussianNonlinearDynamics(dynamics_fn, n_latents, approximation_pdf, Q_diag, device=device)

    """initial condition"""
    initial_condition_pdf = DenseGaussianInitialCondition(n_latents, m_0, Q_0_diag, device=device)

    """local/backward encoder"""
    observation_to_nat = LocalEncoderLRMvn(
        n_neurons,
        n_hidden_current_obs,
        n_latents,
        likelihood_pdf=likelihood_pdf,
        rank=rank_y,
        device=device,
    )
    nl_filter = NonlinearFilter(dynamics_mod, initial_condition_pdf, device)

    """sequence vae"""
    ssm = FullRankNonlinearStateSpaceModelFilter(
        dynamics_mod,
        approximation_pdf,
        likelihood_pdf,
        B,
        initial_condition_pdf,
        observation_to_nat,
        nl_filter,
        device=device,
    )

    ssm.load_state_dict(torch.load("results/ssm_cart_state_dict_cart_epoch_500.pt", map_location=torch.device('cpu')))
    # ssm.load_state_dict(torch.load("results/ssm_cart_state_dict_cart_epoch_500.pt"))
    ssm.eval()

    """real-time test"""
    z_f = []

    for t in range(n_time_bins):
        if t == 0:
            stats_t, z_f_t = ssm.step_0(y_gt[:, t], u[:, t], n_samples)
        else:
            stats_t, z_f_t = ssm.step_t(y_gt[:, t], u[:, t], n_samples, z_f[t - 1])

        z_f.append(z_f_t)

    z_f = torch.stack(z_f, dim=2)
    # z_c = polar_to_cartesian(z_f)

    with torch.no_grad():
        # Why does it still plot sinusoidals tho we are using the polar z_t?
        # (see results/epoch_500.png) Note in the plot both z_gt and z-f are in cart, right?
        plotting.plot_latents(n_latents, n_samples, blues, z_f.cpu(), z_gt.cpu(), epoch=n_epochs)


if __name__ == '__main__':
    validate()
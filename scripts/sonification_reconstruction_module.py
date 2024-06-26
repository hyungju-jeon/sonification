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


# TODO: create config.yaml

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


def polar_to_cartesian(z_p):
    """
    Converts complex numbers from polar form to cartesian form.

    Args:
        z_p (torch.Tensor): Complex numbers in polar form with shape (batch_size, channels, height, width, 2).

    Returns:
        torch.Tensor: Complex numbers in cartesian form with shape (batch_size, channels, height, width, 2).
    """
    if isinstance(z_p, np.ndarray):
        z_p = torch.tensor(z_p)
    z_c = torch.zeros_like(z_p)

    for i in range(z_p.shape[-1] // 2):
        z_c[..., 2 * i] = z_p[..., 2 * i] * torch.cos(z_p[..., 2 * i + 1])
        z_c[..., 2 * i + 1] = z_p[..., 2 * i] * torch.sin(z_p[..., 2 * i + 1])

    return z_c


def cartesian_to_polar(z_c):
    """
    Converts complex numbers from cartesian form to polar form.

    Args:
        z_c (torch.Tensor): Complex numbers in cartesian form with shape (batch_size, channels, height, width, 2).

    Returns:
        torch.Tensor: Complex numbers in polar form with shape (batch_size, channels, height, width, 2).
    """
    if isinstance(z_c, np.ndarray):
        z_c = torch.tensor(z_c)
    z_p = torch.zeros_like(z_c)

    for i in range(z_c.shape[-1] // 2):
        z_p[..., 2 * i] = torch.sqrt(z_c[..., 2 * i] ** 2 + z_c[..., 2 * i + 1] ** 2)
        z_p[..., 2 * i + 1] = torch.atan2(z_c[..., 2 * i + 1], z_c[..., 2 * i])

    return z_p


def compute_cartesian_input(u, z_p, dt=1e-3):
    """
    Computes the cartesian input.

    Args:
        u (torch.Tensor): The input tensor.
        z_c (torch.Tensor): The latent tensor in cartesian form.

    Returns:
        torch.Tensor: The cartesian input tensor.
    """
    u_c = np.zeros_like(u)
    z_w_input = np.zeros_like(z_p)
    z_wo_input = np.zeros_like(z_p)
    for i in range(u.shape[0]):
        z_w_input[i, 0] = z_p[i, 0] + (z_p[i, 0] * (1 - z_p[i, 0]) - u[i, 1]) * dt
        z_w_input[i, 1] = z_p[i, 1] + (2 * np.pi * 0.2 + u[i, 0]) * dt

        z_wo_input[i, 0] = z_p[i, 0] + (z_p[i, 0] * (1 - z_p[i, 0])) * dt
        z_wo_input[i, 1] = z_p[i, 1] + (2 * np.pi * 0.2) * dt

    u_c = polar_to_cartesian(z_w_input) - polar_to_cartesian(z_wo_input)
    return u_c


def generate_sample(n_time_bins):
    """
    Generate a sample for sonification reconstruction.

    Args:
        n_time_bins (int): The number of time bins.

    Returns:
        tuple: A tuple containing three torch tensors:
            - sum_y_slow (torch.Tensor): A tensor of shape (n_time_bins, 100) representing the sum of y_slow values.
            - z_slow (torch.Tensor): A tensor representing the z_slow values.
            - u (torch.Tensor): A tensor representing the u values, multiplied by 20 and of type torch.float32.
    """

    # Shut off the input to the perturb_cycle
    # u = 1.5 * np.zeros((n_time_bins, 2))
    u = 1.5 * (np.random.rand(n_time_bins, 2) - 0.5) * 10
    u_repeat = np.repeat(u, 20, axis=0)

    reference_cycle = limit_circle(**CYCLE_SLOW)
    perturb_cycle = limit_circle(**CYCLE_SLOW)
    coupled_cycle = two_limit_circle(reference_cycle, perturb_cycle)

    z_slow = coupled_cycle.generate_trajectory(20 * n_time_bins, u_repeat)
    y_slow = np.random.poisson(np.exp(z_slow @ C_slow + b_slow))
    u_c = compute_cartesian_input(u_repeat, z_slow[:, :2])

    sum_y_slow = np.zeros((n_time_bins, 100))

    for i in range(n_time_bins):
        sum_y_slow[i, :] = np.sum(y_slow[i * 20 : (i + 1) * 20, :], axis=0)

    return (
        torch.tensor(sum_y_slow),
        torch.tensor(z_slow[::20]),
        torch.tensor(u_c[::20] * 20).type(torch.float32),
    )


def train_network():

    bin_sz = 20e-3
    device = "cuda:0"
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

    y_train_dataset = torch.utils.data.TensorDataset(
        y_gt,
        u,
        z_gt,
    )
    train_dataloader = torch.utils.data.DataLoader(y_train_dataset, batch_size=batch_sz, shuffle=True)

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

    # ssm.likelihood_pdf.readout_fn[-1].bias.data = prob_utils.estimate_poisson_rate_bias(y_train, bin_sz)
    """train model"""
    opt = torch.optim.Adam(ssm.parameters(), lr=1e-3, weight_decay=1e-6)

    for t in (p_bar := tqdm(range(n_epochs), position=0, leave=True)):
        avg_loss = 0.0

        print(f"epoch: {t}")
        for dx, (y_tr, u_tr, z_tr) in enumerate(train_dataloader):
            ssm.train()
            opt.zero_grad()
            loss, z_p, stats = ssm(y_tr, n_samples, u_tr)
            avg_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(ssm.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            p_bar.set_description(f"loss: {loss.item()}")

        avg_loss /= len(train_dataloader)

        with torch.no_grad():
            if t % 10 == 0:
                # z_c = polar_to_cartesian(z_s)

                torch.save(ssm.state_dict(), f"results/ssm_state_dict_cart_epoch_{t}.pt")

                plotting.plot_latents(n_latents, n_samples, blues, z_p.cpu(), z_gt.cpu(), epoch=t)

    torch.save(ssm.state_dict(), f"results/ssm_cart_state_dict_cart_epoch_{n_epochs}.pt")

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
        plotting.plot_latents(n_latents, n_samples, blues, z_f.cpu(), z_gt.cpu(), epoch=n_epochs)


if __name__ == "__main__":

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    # initialize_loading_matrix()
    # generate_sample(10)
    train_network()

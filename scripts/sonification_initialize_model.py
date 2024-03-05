# %%
import os
import random
from re import A

import h5py
from matplotlib.pylab import f
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.loggers import CSVLogger
from matplotlib import cm
from numpy import random
from sklearn.linear_model import Ridge
from tqdm import tqdm
from scipy.linalg import block_diag

import filter.utils as utils
from filter.approximations import DenseGaussianApproximations
from filter.dynamics import (
    DenseGaussianInitialCondition,
    DenseGaussianNonlinearDynamics,
)
from filter.encoders import BackwardEncoderLRMvn, LocalEncoderLRMvn
from filter.likelihoods import (
    GaussianLikelihood,
    PoissonLikelihood,
    LinearPolarToCartesian,
)
from filter.nonlinear_smoother import (
    FullRankNonlinearStateSpaceModelFilter,
    NonlinearFilter,
)
from utils.ndlib.dslib import *
from utils.ndlib.dynlib import *

dt = 1e-3
num_neurons = 25

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
    "w": 2 * np.pi * 0.5,
    "Q": None,
    "dt": dt,
}


loading_matrix_fast_name = (
    "/Users/hyungju/Desktop/hyungju/Project/sonification/data/loading_matrix_fast.npz"
)
loading_matrix_slow_name = (
    "/Users/hyungju/Desktop/hyungju/Project/sonification/data/loading_matrix_slow.npz"
)

param = np.load(loading_matrix_fast_name, allow_pickle=True)
C_fast, b_fast = block_diag(*param["C"]), param["b"].flatten()
param = np.load(loading_matrix_slow_name, allow_pickle=True)
C_slow, b_slow = block_diag(*param["C"]), param["b"].flatten()

loading = block_diag(C_fast, C_slow)
b = np.vstack([b_fast, b_slow]).flatten()


def generate_loading_matrix(num_neurons, cycle_info, scale=1, noise=5):
    dt = cycle_info["dt"]
    reference_cycle = limit_circle(**cycle_info)
    perturb_cycle = limit_circle(**cycle_info)
    two_cycle = two_limit_circle(reference_cycle, perturb_cycle)
    latent_trajectory = two_cycle.generate_trajectory(2000) * 1e-3
    target_rate = 10

    theta = np.random.uniform(0, 2 * np.pi, num_neurons)
    r = scale + np.random.randn(num_neurons) * noise
    C_r = np.array([np.cos(theta), np.sin(theta)]) * r
    target_firing_rate = target_rate + np.random.randn(num_neurons) * 5
    target_firing_rate *= 1e-3
    b = 1.0 * np.random.rand(1, num_neurons)
    firing_rates = computeFiringRate(latent_trajectory[:, :2], C_r, b)
    b_r = updateBiasToMatchTargetFiringRate(
        np.mean(firing_rates, axis=0), b, targetRatePerBin=target_firing_rate
    )

    theta = np.random.uniform(0, 2 * np.pi, num_neurons)
    r = scale + np.random.randn(num_neurons) * noise
    C_p = np.array([np.cos(theta), np.sin(theta)]) * r
    target_firing_rate = target_rate + np.random.randn(num_neurons) * 5
    target_firing_rate *= 1e-3
    b = 1.0 * np.random.rand(1, num_neurons)
    firing_rates = computeFiringRate(latent_trajectory[:, :2], C_p, b)
    b_p = updateBiasToMatchTargetFiringRate(
        np.mean(firing_rates, axis=0), b, targetRatePerBin=target_firing_rate
    )
    return [C_r, C_p], [b_r, b_p]


def initialize_loading_matrix():
    num_neurons = 25
    loading_matrix_fast_name = "./data/loading_matrix_fast.npz"
    loading_matrix_slow_name = "./data/loading_matrix_slow.npz"

    if not os.path.exists(loading_matrix_fast_name) or not os.path.exists(
        loading_matrix_slow_name
    ):
        C_fast, b_fast = generate_loading_matrix(
            num_neurons, CYCLE_FAST, scale=3, noise=0.3
        )

        np.savez(loading_matrix_fast_name, C=C_fast, b=b_fast)

        C_slow, b_slow = generate_loading_matrix(
            num_neurons, CYCLE_SLOW, scale=3, noise=0.3
        )
        np.savez(loading_matrix_slow_name, C=C_slow, b=b_slow)


def generate_sample(n_time_bins):
    reference_cycle = limit_circle(**CYCLE_FAST)
    perturb_cycle = limit_circle(**CYCLE_FAST)
    coupled_cycle = two_limit_circle(reference_cycle, perturb_cycle)
    z_fast = coupled_cycle.generate_trajectory(20 * n_time_bins)
    y_fast = np.exp(z_fast @ C_fast + b_fast)
    # Y_fast is size of (20*n_time_bins, 50) moving average of size 20 with step size 20 along axis 0
    y_fast = np.mean(y_fast.reshape(20, 50, -1), axis=0).reshape(n_time_bins, 50)

    reference_cycle = limit_circle(**CYCLE_SLOW)
    perturb_cycle = limit_circle(**CYCLE_SLOW)
    coupled_cycle = two_limit_circle(reference_cycle, perturb_cycle)
    z_slow = coupled_cycle.generate_trajectory(20000)
    y_slow = np.exp(z_fast @ C_slow + b_slow)
    y_slow = np.mean(y_slow.reshape(20, 50, -1), axis=0).reshape(n_time_bins, 50)

    return torch.tensor(np.hstack([y_fast, y_slow])), torch.tensor(
        np.hstack([z_fast[::20], z_slow[::20]])
    )


def train_network():
    bin_sz = 20e-3
    device = "cpu"
    data_device = "cpu"
    bin_sz_ms = int(bin_sz * 1e3)

    default_dtype = torch.float32
    torch.set_default_dtype(default_dtype)

    """hyperparameters"""
    n_inputs = 2
    n_latents = 8
    n_hidden_current_obs = 128
    n_samples = 25
    rank_y = 2

    batch_sz = 256
    n_epochs = 250
    blues = cm.get_cmap("Blues", n_samples)

    """data params"""
    n_trials = 5
    n_neurons = 100
    n_time_bins = 1000

    B = torch.nn.Linear(n_inputs, n_latents, bias=False, device=device).requires_grad_(
        False
    )
    B.weight.data = torch.tensor(
        [[0, -1], [1, 0], [0, -1], [1, 0], [0, -1], [1, 0], [0, -1], [1, 0]]
    ).type(default_dtype)
    Q_0_diag = torch.ones(n_latents, device=device).requires_grad_(False)
    Q_diag = torch.ones(n_latents, device=device).requires_grad_(False)
    R_diag = torch.ones(n_neurons, device=device).requires_grad_(False)
    m_0 = torch.zeros(n_latents, device=device).requires_grad_(False)

    """generate input and latent/observations"""
    u = torch.rand((n_trials, n_time_bins, n_inputs), device=device) * bin_sz
    y_gt = torch.zeros((n_trials, n_time_bins, n_neurons), device=device)
    z_gt = torch.zeros((n_trials, n_time_bins, n_latents), device=device)
    for i in range(n_trials):
        print(f"trial: {i}")
        y_gt[i], z_gt[i] = generate_sample(n_time_bins)

    y_train_dataset = torch.utils.data.TensorDataset(
        y_gt,
        u,
        z_gt,
    )
    train_dataloader = torch.utils.data.DataLoader(
        y_train_dataset, batch_size=batch_sz, shuffle=True
    )

    """approximation pdf"""
    approximation_pdf = DenseGaussianApproximations(n_latents, device)

    """likelihood pdf"""
    C = LinearPolarToCartesian(
        n_latents, n_neurons, 4, loading=loading, bias=b, device=device
    )
    likelihood_pdf = PoissonLikelihood(C, n_neurons, delta=bin_sz, device=device)

    """dynamics module"""

    # dynamics_fn = utils.build_gru_dynamics_function(n_latents, n_hidden_dynamics, device=device)
    def A(x):
        Ax = torch.zeros_like(x)
        Ax[:, :, 0] = x[:, :, 0] + x[:, :, 0] * (1 - x[:, :, 0] ** 2) * bin_sz
        Ax[:, :, 1] = x[:, :, 1] + 2 * np.pi * 1.5 * bin_sz
        Ax[:, :, 2] = x[:, :, 2] + x[:, :, 2] * (1 - x[:, :, 2] ** 2) * bin_sz
        Ax[:, :, 3] = x[:, :, 3] + 2 * np.pi * 0.5 * bin_sz
        return Ax

    dynamics_fn = A
    dynamics_mod = DenseGaussianNonlinearDynamics(
        dynamics_fn, n_latents, approximation_pdf, Q_diag, device=device
    )

    """initial condition"""
    initial_condition_pdf = DenseGaussianInitialCondition(
        n_latents, m_0, Q_0_diag, device=device
    )

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
            loss, z_s, stats = ssm(y_tr, n_samples, u_tr)
            avg_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(ssm.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            p_bar.set_description(f"loss: {loss.item()}")

        avg_loss /= len(train_dataloader)

        with torch.no_grad():
            if t % 10 == 0:
                torch.save(ssm.state_dict(), f"results/ssm_state_dict_epoch_{t}.pt")
                fig, axs = plt.subplots(1, n_latents)
                [
                    axs[i].plot(z_s[j, 0, :, i], color=blues(j), alpha=0.5)
                    for i in range(n_latents)
                    for j in range(n_samples)
                ]
                [
                    axs[i].plot(z_tr[0, :, i], color="black", alpha=0.7, label="true")
                    for i in range(n_latents)
                ]
                [axs[i].set_box_aspect(1.0) for i in range(n_latents)]
                [axs[i].set_title(f"dim {i}") for i in range(n_latents)]
                plt.show()

    torch.save(ssm.state_dict(), f"results/ssm_state_dict_epoch_{n_epochs}.pt")


if __name__ == "__main__":
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    # initialize_loading_matrix()
    # train_network()
    bin_sz = 20e-3
    device = "cpu"
    data_device = "cpu"
    bin_sz_ms = int(bin_sz * 1e3)

    default_dtype = torch.float32
    torch.set_default_dtype(default_dtype)

    """hyperparameters"""
    n_inputs = 2
    n_latents = 8
    n_hidden_current_obs = 128
    n_samples = 25
    rank_y = 2

    batch_sz = 256
    n_epochs = 250
    blues = cm.get_cmap("Blues", n_samples)

    """data params"""
    n_trials = 5
    n_neurons = 100
    n_time_bins = 1000

    B = torch.nn.Linear(n_inputs, n_latents, bias=False, device=device).requires_grad_(
        False
    )
    B.weight.data = torch.tensor(
        [[0, -1], [1, 0], [0, -1], [1, 0], [0, -1], [1, 0], [0, -1], [1, 0]]
    ).type(default_dtype)
    Q_0_diag = torch.ones(n_latents, device=device).requires_grad_(False)
    Q_diag = torch.ones(n_latents, device=device).requires_grad_(False)
    R_diag = torch.ones(n_neurons, device=device).requires_grad_(False)
    m_0 = torch.zeros(n_latents, device=device).requires_grad_(False)

    """generate input and latent/observations"""
    u = torch.rand((n_trials, n_time_bins, n_inputs), device=device) * bin_sz
    y_gt = torch.zeros((n_trials, n_time_bins, n_neurons), device=device)
    z_gt = torch.zeros((n_trials, n_time_bins, n_latents), device=device)
    for i in range(n_trials):
        print(f"trial: {i}")
        y_gt[i], z_gt[i] = generate_sample(n_time_bins)

    y_train_dataset = torch.utils.data.TensorDataset(
        y_gt,
        u,
        z_gt,
    )
    train_dataloader = torch.utils.data.DataLoader(
        y_train_dataset, batch_size=batch_sz, shuffle=True
    )

    """approximation pdf"""
    approximation_pdf = DenseGaussianApproximations(n_latents, device)

    """likelihood pdf"""
    C = LinearPolarToCartesian(
        n_latents, n_neurons, 4, loading=loading, bias=b, device=device
    )
    likelihood_pdf = PoissonLikelihood(C, n_neurons, delta=bin_sz, device=device)

    """dynamics module"""

    # dynamics_fn = utils.build_gru_dynamics_function(n_latents, n_hidden_dynamics, device=device)
    def A(x):
        Ax = torch.zeros_like(x)
        Ax[:, :, 0] = x[:, :, 0] + x[:, :, 0] * (1 - x[:, :, 0] ** 2) * bin_sz
        Ax[:, :, 1] = x[:, :, 1] + 2 * np.pi * 1.5 * bin_sz
        Ax[:, :, 2] = x[:, :, 2] + x[:, :, 2] * (1 - x[:, :, 2] ** 2) * bin_sz
        Ax[:, :, 3] = x[:, :, 3] + 2 * np.pi * 0.5 * bin_sz
        return Ax

    dynamics_fn = A
    dynamics_mod = DenseGaussianNonlinearDynamics(
        dynamics_fn, n_latents, approximation_pdf, Q_diag, device=device
    )

    """initial condition"""
    initial_condition_pdf = DenseGaussianInitialCondition(
        n_latents, m_0, Q_0_diag, device=device
    )

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
            loss, z_s, stats = ssm(y_tr, n_samples, u_tr)
            avg_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(ssm.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            p_bar.set_description(f"loss: {loss.item()}")

        avg_loss /= len(train_dataloader)

        with torch.no_grad():
            if t % 10 == 0:
                # torch.save(ssm.state_dict(), f"results/ssm_state_dict_epoch_{t}.pt")
                fig, axs = plt.subplots(1, n_latents)
                [
                    axs[i].plot(z_s[j, 0, :, i], color=blues(j), alpha=0.5)
                    for i in range(n_latents)
                    for j in range(n_samples)
                ]
                [
                    axs[i].plot(z_tr[0, :, i], color="black", alpha=0.7, label="true")
                    for i in range(n_latents)
                ]
                [axs[i].set_box_aspect(1.0) for i in range(n_latents)]
                [axs[i].set_title(f"dim {i}") for i in range(n_latents)]
                plt.show()

    torch.save(ssm.state_dict(), f"results/ssm_state_dict_epoch_{n_epochs}.pt")

    """real-time test"""
    z_f = []

    for t in range(n_time_bins):
        if t == 0:
            stats_t, z_f_t = ssm.step_0(y_gt[:, t], u[:, t], n_samples)
        else:
            stats_t, z_f_t = ssm.step_t(y_gt[:, t], u[:, t], n_samples, z_f[t - 1])

        z_f.append(z_f_t)

    z_f = torch.stack(z_f, dim=2)

    with torch.no_grad():
        fig, axs = plt.subplots(1, n_latents)
        [
            axs[i].plot(z_f[j, 0, :, i], color=blues(j), alpha=0.5)
            for i in range(n_latents)
            for j in range(n_samples)
        ]
        [
            axs[i].plot(z_gt[0, :, i], color="black", alpha=0.7, label="true")
            for i in range(n_latents)
        ]
        [axs[i].set_box_aspect(1.0) for i in range(n_latents)]
        [axs[i].set_title(f"dim {i}") for i in range(n_latents)]
        plt.show()

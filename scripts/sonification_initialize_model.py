# %%
import os
import random
import time
from scipy.signal import convolve2d

import h5py
from matplotlib.pylab import f
import matplotlib.pyplot as plt
import numpy as np
from sympy import N
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
    u_polar = 1.5 * (np.random.rand(n_time_bins, 2) - 0.5)
    u_polar[:, 0] = np.clip(u_polar[:, 0] * 2 * np.pi, -np.pi, np.pi) * 0.001
    u_polar[:, 1] = np.clip(u_polar[:, 1], -1, 1) * 0.01
    u_repeat = np.repeat(u_polar, 20, axis=0)

    reference_cycle = limit_circle(**CYCLE_FAST)
    perturb_cycle = limit_circle(**CYCLE_FAST)
    coupled_cycle = two_limit_circle(reference_cycle, perturb_cycle)
    z_fast = coupled_cycle.generate_trajectory(20 * n_time_bins, u_repeat)
    y_fast = np.random.poisson(np.exp(z_fast @ C_fast + b_fast))

    reference_cycle = limit_circle(**CYCLE_SLOW)
    perturb_cycle = limit_circle(**CYCLE_SLOW)
    coupled_cycle = two_limit_circle(reference_cycle, perturb_cycle)
    z_slow = coupled_cycle.generate_trajectory(20 * n_time_bins, u_repeat)
    y_slow = np.random.poisson(np.exp(z_slow @ C_slow + b_slow))

    sum_y_fast = np.zeros((n_time_bins, 50))
    sum_y_slow = np.zeros((n_time_bins, 50))
    for i in range(n_time_bins):
        sum_y_fast[i, :] = np.sum(y_fast[i * 20 : (i + 1) * 20, :], axis=0)
        sum_y_slow[i, :] = np.sum(y_slow[i * 20 : (i + 1) * 20, :], axis=0)

    z_fast = z_fast[::20]
    z_slow = z_slow[::20]
    u_cart = np.zeros((n_time_bins, 4))

    for j in range(n_time_bins):
        r = np.sqrt(z_fast[j, 2] ** 2 + z_fast[j, 3] ** 2)
        theta = np.arctan2(z_fast[j, 3], z_fast[j, 2])
        x, y = r * np.cos(theta), r * np.sin(theta)
        x_n, y_n = (r - u_polar[j, 1]) * np.cos(theta + u_polar[j, 0]), (
            r - u_polar[j, 1]
        ) * np.sin(theta + u_polar[j, 0])
        u_cart[j, 0] = x_n - x
        u_cart[j, 1] = y_n - y

        r = np.sqrt(z_slow[j, 2] ** 2 + z_slow[j, 3] ** 2)
        theta = np.arctan2(z_slow[j, 3], z_slow[j, 2])
        x, y = r * np.cos(theta), r * np.sin(theta)
        x_n, y_n = (r - u_polar[j, 1]) * np.cos(theta + u_polar[j, 0]), (
            r - u_polar[j, 1]
        ) * np.sin(theta + u_polar[j, 0])
        u_cart[j, 2] = x_n - x
        u_cart[j, 3] = y_n - y

    return (
        torch.tensor(np.hstack([sum_y_fast, sum_y_slow])),
        torch.tensor(np.hstack([z_fast, z_slow])),
        torch.tensor(u_cart * 20).type(torch.float32),
    )


def train_network():
    bin_sz = 20e-3
    device = "cpu"
    data_device = "cpu"
    bin_sz_ms = int(bin_sz * 1e3)

    default_dtype = torch.float32
    torch.set_default_dtype(default_dtype)

    """hyperparameters"""
    n_inputs = 4
    n_latents = 8
    n_hidden_current_obs = 128
    n_samples = 25
    rank_y = n_latents

    batch_sz = 256
    n_epochs = 250
    blues = cm.get_cmap("Blues", n_samples)

    """data params"""
    n_trials = 1000
    n_neurons = 100
    n_time_bins = 100

    # def B(u):
    #     Bu = torch.ones(u.shape[0], u.shape[1], n_latents, device=device).type(
    #         default_dtype
    #     )
    #     for i in range(4):
    #         if i % 2 == 1:
    #             Bu[:, :, 2 * i] = u[:, :, 0]
    #             Bu[:, :, 2 * i + 1] = u[:, :, 1]
    #     return Bu
    B = torch.nn.Linear(n_inputs, n_latents, bias=False, device=device).requires_grad_(
        False
    )
    B.weight.data = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    ).type(default_dtype)
    Q_0_diag = torch.ones(n_latents, device=device).requires_grad_(False) * 1e-2
    Q_diag = torch.ones(n_latents, device=device).requires_grad_(False) * 1e-2
    R_diag = torch.ones(n_neurons, device=device).requires_grad_(False)
    m_0 = (
        torch.tensor([0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0], device=device)
        .requires_grad_(False)
        .type(default_dtype)
    )

    """generate input and latent/observations"""
    u = torch.zeros((n_trials, n_time_bins, n_inputs), device=device)
    y_gt = torch.zeros((n_trials, n_time_bins, n_neurons), device=device)
    z_gt = torch.zeros((n_trials, n_time_bins, n_latents), device=device)
    for i in range(n_trials):
        print(f"trial: {i}")
        y_gt[i], z_gt[i], u[i] = generate_sample(n_time_bins)

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
    # C = LinearPolarToCartesian(
    #     n_latents, n_neurons, 4, loading=loading, bias=b, device=device
    # )
    C = torch.nn.Linear(n_latents, n_neurons, bias=True, device=device).requires_grad_(
        False
    )
    C.weight.data = torch.tensor(loading.T).type(torch.float32)
    C.bias.data = torch.tensor(b).type(torch.float32)

    likelihood_pdf = PoissonLikelihood(C, n_neurons, delta=bin_sz, device=device)

    """dynamics module"""

    # dynamics_fn = utils.build_gru_dynamics_function(n_latents, n_hidden_dynamics, device=device)
    # def A(x):
    #     Ax = torch.zeros_like(x)
    #     Ax[:, :, 0] = x[:, :, 0] + x[:, :, 0] * (1 - x[:, :, 0] ** 2) * bin_sz
    #     Ax[:, :, 1] = x[:, :, 1] + 2 * np.pi * 1.5 * bin_sz
    #     Ax[:, :, 2] = x[:, :, 2] + x[:, :, 2] * (1 - x[:, :, 2] ** 2) * bin_sz
    #     Ax[:, :, 3] = x[:, :, 3] + 2 * np.pi * 1.5 * bin_sz
    #     Ax[:, :, 4] = x[:, :, 4] + x[:, :, 4] * (1 - x[:, :, 4] ** 2) * bin_sz
    #     Ax[:, :, 5] = x[:, :, 5] + 2 * np.pi * 0.5 * bin_sz
    #     Ax[:, :, 6] = x[:, :, 6] + x[:, :, 6] * (1 - x[:, :, 6] ** 2) * bin_sz
    #     Ax[:, :, 7] = x[:, :, 7] + 2 * np.pi * 0.5 * bin_sz
    #     return Ax

    def A(x):
        Ax = torch.zeros_like(x)
        for i in range(4):
            r = torch.sqrt(x[:, :, 2 * i] ** 2 + x[:, :, 2 * i + 1] ** 2)
            theta = torch.atan2(x[:, :, 2 * i + 1], x[:, :, 2 * i])

            r_new = r + r * (1 - r**2) * bin_sz
            theta_new = (
                theta + 2 * np.pi * 1.5 * bin_sz
                if i < 2
                else theta + 2 * np.pi * 0.5 * bin_sz
            )

            Ax[:, :, 2 * i] = r_new * torch.cos(theta_new)
            Ax[:, :, 2 * i + 1] = r_new * torch.sin(theta_new)
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
                z_c = torch.zeros_like(z_s)

                for i in range(4):
                    z_c[:, :, :, 2 * i] = z_s[:, :, :, 2 * i] * torch.cos(
                        z_s[:, :, :, 2 * i + 1]
                    )
                    z_c[:, :, :, 2 * i + 1] = z_s[:, :, :, 2 * i] * torch.sin(
                        z_s[:, :, :, 2 * i + 1]
                    )

                torch.save(
                    ssm.state_dict(), f"results/ssm_state_dict_cart_epoch_{t}.pt"
                )
                fig, axs = plt.subplots(1, n_latents, figsize=(20, 5))
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
                plt.savefig(f"results/epoch_{t}.png")
                plt.close()
                # plt.show()

    torch.save(
        ssm.state_dict(), f"results/ssm_cart_state_dict_cart_epoch_{n_epochs}.pt"
    )

    """real-time test"""
    z_f = []

    for t in range(n_time_bins):
        if t == 0:
            stats_t, z_f_t = ssm.step_0(y_gt[:, t], u[:, t], n_samples)
        else:
            stats_t, z_f_t = ssm.step_t(y_gt[:, t], u[:, t], n_samples, z_f[t - 1])

        z_f.append(z_f_t)

    z_f = torch.stack(z_f, dim=2)
    z_c = torch.zeros_like(z_f)

    for i in range(4):
        z_c[:, :, :, 2 * i] = z_f[:, :, :, 2 * i] * torch.cos(z_f[:, :, :, 2 * i + 1])
        z_c[:, :, :, 2 * i + 1] = z_f[:, :, :, 2 * i] * torch.sin(
            z_f[:, :, :, 2 * i + 1]
        )

    with torch.no_grad():
        fig, axs = plt.subplots(1, n_latents, figsize=(20, 5))
        [
            axs[i].plot(z_c[j, 0, :, i], color=blues(j), alpha=0.5)
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


if __name__ == "__main__":
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    # initialize_loading_matrix()
    train_network()

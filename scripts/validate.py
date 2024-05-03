import numpy as np
import os
import random
import math

from numpy import random
from tqdm import tqdm

from scipy.signal import convolve2d
from scipy.linalg import block_diag

import matplotlib.pyplot as plt
from matplotlib.pylab import f
from matplotlib import cm

import torch
import torch.nn as nn

import pytorch_lightning as lightning

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from filter.approximations import DenseGaussianApproximations
from filter.dynamics import (
    DenseGaussianInitialCondition,
    DenseGaussianNonlinearDynamics,
)
from filter.encoders import LocalEncoderLRMvn
from filter.likelihoods import (
    PoissonLikelihood,
)
from filter.nonlinear_smoother import (
    FullRankNonlinearStateSpaceModelFilter,
    NonlinearFilter,
)
from utils.ndlib.dslib import *
from utils.ndlib.dynlib import *

from utils import plotting


def main():

    bin_sz = 20e-3
    device = "cpu"
    data_device = "cpu"
    bin_sz_ms = int(bin_sz * 1e3)

    default_dtype = torch.float32
    torch.set_default_dtype(default_dtype)

    """hyperparameters"""
    n_inputs = 2
    n_latents = 4
    n_hidden_current_obs = 128
    n_samples = 25
    rank_y = n_latents

    batch_sz = 256
    n_epochs = 700
    blues = plt.cm.get_cmap("Blues", n_samples)

    """data params"""
    n_trials = 3500
    n_neurons = 100
    n_time_bins = 300

    B = torch.nn.Linear(n_inputs, n_latents, bias=False, device=device).requires_grad_(
        False
    )

    # Defines how the inputs (u) affects the perturb_cycle
    B.weight.data = torch.tensor(
        [[0, 0], [0, 0], [0, -1], [1, 0]]
    ).type(default_dtype)
    
    Q_0_diag = torch.ones(n_latents, device=device).requires_grad_(False) * 1e-2
    Q_diag = torch.ones(n_latents, device=device).requires_grad_(False) * 1e-2
    R_diag = torch.ones(n_neurons, device=device).requires_grad_(False)
    
    m_0 = (
        torch.tensor([0.5, 0, 0.5, 0], device=device)
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

    likelihood_pdf = PoissonLikelihood(C, n_neurons, delta=20, device=device)

    """dynamics module"""
    # dynamics_fn = utils.build_gru_dynamics_function(n_latents, n_hidden_dynamics, device=device)
    def A(x):
        Ax = torch.zeros_like(x)
        for i in range(2):
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



if __name__ == '__main__':
    main()
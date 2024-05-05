# %%

import numpy as np
import matplotlib.pyplot as plt

import torch

from scipy.linalg import block_diag

from utils.ndlib.dslib import *
from utils.ndlib.dynlib import *


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
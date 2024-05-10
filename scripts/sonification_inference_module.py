# %%
import asyncio
import os
import random
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.loggers import CSVLogger
from matplotlib.pylab import f
from numpy import random
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from scipy.linalg import block_diag
from scipy.signal import convolve2d

from sonification_communication_module import *
from tqdm import tqdm

import filter.utils as utils
from filter.approximations import DenseGaussianApproximations
from filter.dynamics import (
    DenseGaussianInitialCondition,
    DenseGaussianNonlinearDynamics,
)
from filter.encoders import BackwardEncoderLRMvn, LocalEncoderLRMvn
from filter.likelihoods import (
    GaussianLikelihood,
    LinearPolarToCartesian,
    PoissonLikelihood,
)
from filter.nonlinear_smoother import (
    FullRankNonlinearStateSpaceModelFilter,
    NonlinearFilter,
)
from scripts.check_max_communication_limit import INPUT
from utils.ndlib.dslib import *
from utils.ndlib.dynlib import *

# ---------------------------------------------------------------- #
# Common Parameters
dt = 20e-3  # 1ms for dynamic system update
SPIKES = [np.zeros((100, 1))]
INPUT_X = [0]
INPUT_Y = [0]
loading_matrix_slow_name = "./data/loading_matrix_slow.npz"

param = np.load(loading_matrix_slow_name, allow_pickle=True)
C_slow, b_slow = block_diag(*param["C"]), param["b"].flatten()

loading = C_slow
b = b_slow

DISPATCHER = Dispatcher()


# ---------------------------------------------------------------- #
# Timing related functions
def ms_to_ns(ms):
    """
    Convert milliseconds to nanoseconds.

    Args:
        ms (float): The time in milliseconds.

    Returns:
        float: The time in nanoseconds.
    """
    return ms * 1e6


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


class LatentInference:

    def __init__(self, verbose=False):
        # pass the handlers to the dispatcher
        # Load trained network
        self.verbose = verbose
        self.spikes = np.zeros((20, 100))
        DISPATCHER.map("/SPIKES", self.latent_to_inference_osc_handler)
        DISPATCHER.map("/MOTION_ENERGY", self.camera_to_latent_osc_handler)
        self.t = 0
        self.inferred = None
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
        rank_y = n_latents

        """data params"""
        n_neurons = 100

        B = torch.nn.Linear(
            n_inputs, n_latents, bias=False, device=device
        ).requires_grad_(False)

        # Defines how the inputs (u) affects the perturb_cycle
        B.weight.data = (
            torch.tensor([[0, 0], [0, 0], [1, 0], [0, 1]]).type(default_dtype) * 1e-3
        )

        Q_0_diag = torch.ones(n_latents, device=device).requires_grad_(False) * 1e-2
        Q_diag = torch.ones(n_latents, device=device).requires_grad_(False) * 1e-2
        R_diag = torch.ones(n_neurons, device=device).requires_grad_(False)
        m_0 = (
            torch.tensor([0.5, 0, 0.5, 0], device=device)
            .requires_grad_(False)
            .type(default_dtype)
        )

        """approximation pdf"""
        approximation_pdf = DenseGaussianApproximations(n_latents, device)

        """likelihood pdf"""
        C = torch.nn.Linear(
            n_latents, n_neurons, bias=True, device=device
        ).requires_grad_(False)
        C.weight.data = torch.tensor(loading.T).type(torch.float32)
        C.bias.data = torch.tensor(b).type(torch.float32)

        likelihood_pdf = PoissonLikelihood(C, n_neurons, delta=bin_sz_ms, device=device)

        """dynamics module"""

        def A(x):
            Ax = torch.zeros_like(x)
            for i in range(2):
                r = torch.sqrt(x[:, :, 2 * i] ** 2 + x[:, :, 2 * i + 1] ** 2)
                theta = torch.atan2(x[:, :, 2 * i + 1], x[:, :, 2 * i])

                r_new = r + r * (1 - r) * bin_sz
                theta_new = theta + 2 * np.pi * 0.2 * bin_sz

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
        self.ssm = FullRankNonlinearStateSpaceModelFilter(
            dynamics_mod,
            approximation_pdf,
            likelihood_pdf,
            B,
            initial_condition_pdf,
            observation_to_nat,
            nl_filter,
            device=device,
        )
        self.ssm.load_state_dict(
            torch.load(
                f"results/ssm_cart_state_dict_cart_epoch_500.pt",
                map_location=torch.device("cpu"),
            )
        )
        self.sum_spikes = torch.zeros((1, 100))
        self.input = torch.zeros((1, 2))

        self.MAX_OSCsender = SimpleUDPClient(MAX_SERVER, MAX_OUTPUT_PORT)
        self.LOCAL_OSCsender = SimpleUDPClient(LOCAL_SERVER, INFERRED_LATENT_PORT)
        self.prev = np.zeros(4)

    def update_spikes(self, spikes):
        self.spikes[-1] = spikes * 1.0
        self.spikes = np.roll(self.spikes, -1, axis=0)

    def get_inference(self):
        result_cartesian = np.mean(self.inferred.detach().numpy(), axis=0)
        return result_cartesian

    async def start(self):
        await self.setup_server()
        while True:
            start_t = time.perf_counter_ns()
            self.sum_spikes[0, :] = torch.from_numpy(np.sum(self.spikes, axis=0)).type(
                torch.float32
            )
            self.input[0, :] = polar_to_cartesian(
                torch.tensor([INPUT_X[0], INPUT_Y[0]]).type(torch.float32)
            )

            if self.t == 0:
                stats_t, z_f_t = self.ssm.step_0(self.sum_spikes, self.input, 500)
            else:
                stats_t, z_f_t = self.ssm.step_t(
                    self.sum_spikes, self.input, 500, self.inferred
                )
            self.inferred = z_f_t
            self.t += 1

            if self.inferred is not None:
                self.MAX_OSCsender.send_message(
                    "/INFERRED_TRAJECTORY",
                    self.get_inference()[0].tolist(),
                )
                self.LOCAL_OSCsender.send_message(
                    "/INFERRED_TRAJECTORY",
                    self.get_inference().tolist(),
                )
            elapsed_time = time.perf_counter_ns() - start_t
            sleep_duration = np.fmax(dt * 1e9 - (time.perf_counter_ns() - start_t), 0)

            if sleep_duration == 0:
                print(
                    f"Inference system Iteration took {elapsed_time/1e6}ms which is longer than {dt*1e3} ms"
                )
            await busy_timer(sleep_duration)

    async def setup_server(self):
        # python-osc method for establishing the UDP communication with max
        # Connect server if address not in use (check with try catch)
        try:
            server_motion = AsyncIOOSCUDPServer(
                (LOCAL_SERVER, SPIKE_INFERENCE_PORT),
                DISPATCHER,
                asyncio.get_event_loop(),
            )
            transport_motion, _ = await server_motion.create_serve_endpoint()
        except:
            print("Address already in use")

    def latent_to_inference_osc_handler(self, address, *args):
        self.update_spikes(np.array(args))
        SPIKES[0] = args[0]
        if self.verbose:
            print(f"Received update for SPIKES : {args}")

    def camera_to_latent_osc_handler(self, address, *args):
        INPUT_X[0] = args[0]
        INPUT_Y[0] = args[1]
        if self.verbose:
            print(f"Received update for INPUT_X, INPUT_Y : {args}")


async def init_main():
    inference = LatentInference()
    await asyncio.gather(
        inference.start(),
        inferred_latent_sending_loop(ms_to_ns(200), inference, verbose=True),
    )


if __name__ == "__main__":
    inference = LatentInference()
    asyncio.run(inference.start())

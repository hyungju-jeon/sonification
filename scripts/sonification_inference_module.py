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
dt = 10e-3  # 1ms for dynamic system update
SPIKES = [np.zeros((100, 1))]
INPUT_X = [0]
INPUT_Y = [0]
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
        n_latents = 8
        n_hidden_current_obs = 128
        n_samples = 25
        rank_y = n_latents

        batch_sz = 256

        """data params"""
        n_trials = 1000
        n_neurons = 100
        n_time_bins = 100

        B = torch.nn.Linear(
            n_inputs, n_latents, bias=False, device=device
        ).requires_grad_(False)
        B.weight.data = torch.tensor(
            [[0, 0], [0, 0], [0, -1], [1, 0], [0, 0], [0, 0], [0, -1], [1, 0]]
        ).type(default_dtype)
        Q_0_diag = torch.ones(n_latents, device=device).requires_grad_(False) * 1e-2
        Q_diag = torch.ones(n_latents, device=device).requires_grad_(False) * 1e-2
        R_diag = torch.ones(n_neurons, device=device).requires_grad_(False)
        m_0 = (
            torch.tensor([0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0], device=device)
            .requires_grad_(False)
            .type(default_dtype)
        )

        """approximation pdf"""
        approximation_pdf = DenseGaussianApproximations(n_latents, device)

        """likelihood pdf"""
        C = LinearPolarToCartesian(
            n_latents, n_neurons, 4, loading=loading, bias=b, device=device
        )
        likelihood_pdf = PoissonLikelihood(C, n_neurons, delta=bin_sz, device=device)

        """dynamics module"""

        def A(x):
            Ax = torch.zeros_like(x)
            Ax[:, :, 0] = x[:, :, 0] + x[:, :, 0] * (1 - x[:, :, 0] ** 2) * bin_sz
            Ax[:, :, 1] = x[:, :, 1] + 2 * np.pi * 1.5 * bin_sz
            Ax[:, :, 2] = x[:, :, 2] + x[:, :, 2] * (1 - x[:, :, 2] ** 2) * bin_sz
            Ax[:, :, 3] = x[:, :, 3] + 2 * np.pi * 1.5 * bin_sz
            Ax[:, :, 4] = x[:, :, 4] + x[:, :, 4] * (1 - x[:, :, 4] ** 2) * bin_sz
            Ax[:, :, 5] = x[:, :, 5] + 2 * np.pi * 0.5 * bin_sz
            Ax[:, :, 6] = x[:, :, 6] + x[:, :, 6] * (1 - x[:, :, 6] ** 2) * bin_sz
            Ax[:, :, 7] = x[:, :, 7] + 2 * np.pi * 0.5 * bin_sz
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
        self.ssm.load_state_dict(torch.load(f"data/ssm_state_dict_big.pt"))
        self.sum_spikes = torch.zeros((1, 100))
        self.input = torch.zeros((1, 2))

        self.MAX_OSCsender = SimpleUDPClient(MAX_SERVER, MAX_OUTPUT_PORT)
        self.LOCAL_OSCsender = SimpleUDPClient(LOCAL_SERVER, INFERRED_LATENT_PORT)

    def update_spikes(self, spikes):
        self.spikes[-1] = spikes
        self.spikes = np.roll(self.spikes, -1, axis=0)

    def get_inference(self):
        result_polar = np.mean(self.inferred.detach().numpy(), axis=0).flatten()
        result_cart = np.zeros_like(result_polar)
        for i in range(4):
            result_cart[2 * i] = result_polar[2 * i] * np.cos(result_polar[2 * i + 1])
            result_cart[2 * i + 1] = result_polar[2 * i] * np.sin(
                result_polar[2 * i + 1]
            )
        return result_cart

    async def start(self):
        await self.setup_server()
        while True:
            start_t = time.perf_counter_ns()
            self.sum_spikes[0, :] = torch.from_numpy(np.sum(self.spikes, axis=0)).type(
                torch.float32
            )
            self.input[0, :] = torch.tensor([INPUT_X[0], INPUT_Y[0]]).type(
                torch.float32
            )

            if self.t == 0:
                stats_t, z_f_t = self.ssm.step_0(self.sum_spikes, self.input, 100)
            else:
                stats_t, z_f_t = self.ssm.step_t(
                    self.sum_spikes, self.input, 100, self.inferred
                )
            self.inferred = z_f_t
            self.t += 1

            if self.inferred is not None:
                self.MAX_OSCsender.send_message(
                    "/INFERRED_TRAJECTORY",
                    [self.get_inference().tolist()],
                )
                self.LOCAL_OSCsender.send_message(
                    "/INFERRED_TRAJECTORY",
                    [self.get_inference().tolist()],
                )
            elapsed_time = time.perf_counter_ns() - start_t
            sleep_duration = np.fmax(dt * 1e9 - (time.perf_counter_ns() - start_t), 0)

            if sleep_duration == 0 and self.verbose:
                print(
                    f"Dynamical system Iteration took {elapsed_time/1e6}ms which is longer than {dt*1e3} ms"
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
        self.update_spikes(args[0])
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

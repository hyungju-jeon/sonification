# %%
import asyncio
from re import L
import sys
import sonification_latent_module as LatentModule
from multiprocessing import Process, process
from sonification_latent_module import SPIKES_FAST, SPIKES_SLOW
from sonification_communication_module import *

from utils.ndlib.dslib import *
from utils.ndlib.dynlib import *

import numpy as np
import random


# ---------------------------------------------------------------- #
# Common Parameters
dt = 1e-3  # 1ms for dynamic system update
num_neurons = 50
TARGET_FIRING_RATE = 20
TARGET_SNR = 0
CYCLE_FAST = {
    "x0": np.array([0.5, 0]),
    "d": 1,
    "w": 2 * np.pi * 2,
    "Q": np.array([[1e-3, 0.0], [0.0, 1e-3]]),
    "dt": dt,
}
CYCLE_SLOW = {
    "x0": np.array([1, 1]),
    "d": 1,
    "w": 2 * np.pi * 1,
    "Q": np.array([[1e-3, 0.0], [0.0, 1e-3]]),
    "dt": dt,
}


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


# ---------------------------------------------------------------- #
# Construct Loading matrix


async def init_main():
    loading_matrix_fast_name = "./data/loading_matrix_fast.npz"
    loading_matrix_slow_name = "./data/loading_matrix_slow.npz"
    param = np.load(loading_matrix_fast_name, allow_pickle=True)
    C_fast, b_fast = param["C"], param["b"]
    param = np.load(loading_matrix_slow_name, allow_pickle=True)
    C_slow, b_slow = param["C"], param["b"]

    C = np.vstack([C_fast, C_slow])
    b = np.vstack([b_fast, b_slow])

    fast_latent_block = LatentModule.LatentDynamics(CYCLE_FAST, verbose=False)
    slow_latent_block = LatentModule.LatentDynamics(CYCLE_SLOW, verbose=False)
    fast_spike_block = LatentModule.SpikeGenerator(
        C_fast, b_fast, dt, latent_block=fast_latent_block
    )

    slow_spike_block = LatentModule.SpikeGenerator(
        C_slow, b_slow, dt, latent_block=slow_latent_block
    )

    await asyncio.gather(
        fast_latent_block.start(),
        slow_latent_block.start(),
        fast_spike_block.start(SPIKES_FAST),
        slow_spike_block.start(SPIKES_SLOW),
        trajectory_sending_loop(
            ms_to_ns(1), fast_latent_block, slow_latent_block, verbose=True
        ),
        spike_sending_loop(
            ms_to_ns(1), fast_spike_block, slow_spike_block, verbose=True
        ),
    )


# %%

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    asyncio.run(init_main())

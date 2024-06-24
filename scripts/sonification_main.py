# %%
import asyncio
import random

import numpy as np
import sonification_latent_module as LatentModule
from sonification_communication_module import *
from sonification_latent_module import SPIKES_FAST, SPIKES_SLOW

from utils.ndlib.dslib import *
from utils.ndlib.dynlib import *

# ---------------------------------------------------------------- #
# Common Parameters
dt = 1e-3  # 1ms for dynamic system update
num_neurons = 50
TARGET_FIRING_RATE = 20
TARGET_SNR = 0
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
    loading_matrix_slow_name = "./data/loading_matrix_slow.npz"
    param = np.load(loading_matrix_slow_name, allow_pickle=False)
    C_slow, b_slow = param["C"], param["b"]

    slow_latent_block = LatentModule.LatentDynamics(CYCLE_SLOW, verbose=False)
    slow_spike_block = LatentModule.SpikeGenerator(
        C_slow, b_slow, dt, latent_block=slow_latent_block
    )

    await asyncio.gather(
        slow_latent_block.start(),
        slow_spike_block.start(SPIKES_SLOW),
        spike_sending_loop(ms_to_ns(1), slow_spike_block, verbose=False),
        true_latent_sending_loop(ms_to_ns(1), slow_latent_block, verbose=False),
        phase_diff_sending_loop(ms_to_ns(1), slow_latent_block, verbose=False),
        # fake_latent_sending_loop(ms_to_ns(1), slow_latent_block, verbose=False),
    )


# %%

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    asyncio.run(init_main())

# %%
import asyncio

import sonification_latent_module as LatentModule
from sonification_latent_module import SPIKES_FAST, SPIKES_SLOW
from sonification_communication_module import *

from utils.ndlib.dslib import *
from utils.ndlib.dynlib import *

import numpy as np
from numpy import random


# ---------------------------------------------------------------- #
# Common Parameters
dt = 1e-3  # 1ms for dynamic system update
num_neurons = 50
TARGET_FIRING_RATE = 5
TARGET_SNR = -5
CYCLE_FAST = {
    "x0": np.array([0.5, 0]),
    "d": 1,
    "w": 2 * np.pi * 10,
    "Q": np.array([[1e-10, 0.0], [0.0, 1e-10]]),
    "dt": dt,
}
CYCLE_SLOW = {
    "x0": np.array([1, 1]),
    "d": 1,
    "w": 2 * np.pi * 1,
    "Q": np.array([[1e-10, 0.0], [0.0, 1e-10]]),
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


async def init_main():
    global C, b
    fast_latent_block = LatentModule.LatentDynamics(CYCLE_FAST, verbose=False)
    slow_latent_block = LatentModule.LatentDynamics(CYCLE_SLOW, verbose=False)
    C, b = simulate_neuron_parameters(
        CYCLE_FAST, num_neurons, TARGET_SNR, TARGET_FIRING_RATE * dt
    )
    fast_spike_block = LatentModule.SpikeGenerator(C, b, latent_block=fast_latent_block)
    print(C)
    C, b = simulate_neuron_parameters(
        CYCLE_SLOW, num_neurons, TARGET_SNR, TARGET_FIRING_RATE * dt
    )
    slow_spike_block = LatentModule.SpikeGenerator(C, b, latent_block=slow_latent_block)

    await asyncio.gather(
        fast_latent_block.start(),
        slow_latent_block.start(),
        fast_spike_block.start(SPIKES_FAST),
        slow_spike_block.start(SPIKES_SLOW),
        trajectory_sending_loop(ms_to_ns(1), fast_latent_block, verbose=True),
        spike_sending_loop(
            ms_to_ns(0.5), fast_spike_block, slow_spike_block, verbose=True
        ),
    )


# ----------------- Loop Components  ------------------- #
def simulate_neuron_parameters(
    cycle_info, num_neurons, target_SNR, target_rate_per_bin
):
    dt = cycle_info["dt"]
    reference_cycle = limit_circle(**cycle_info)
    perturb_cycle = limit_circle(**cycle_info)
    two_cycle = two_limit_circle(reference_cycle, perturb_cycle)

    latent_trajectory = two_cycle.generate_trajectory(2000)
    latent_dim = latent_trajectory.shape[1]

    C = generate_random_loading_matrix(latent_dim, num_neurons, 1, 0.2)
    b = 1.0 * np.random.rand(1, num_neurons) - np.log(target_rate_per_bin)

    C, b, SNR = scaleCforTargetSNR(
        latent_trajectory,
        C,
        b,
        target_rate_per_bin,
        targetSNR=target_SNR,
        SNR_method=computeSNR,
    )

    return C, b


if __name__ == "__main__":
    random.seed(0)
    asyncio.run(init_main())

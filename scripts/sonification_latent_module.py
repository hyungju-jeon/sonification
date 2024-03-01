# %%
import time
import asyncio

from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import AsyncIOOSCUDPServer
from scripts.check_max_communication_limit import INPUT

from utils.ndlib.dynlib import *
from utils.ndlib.dslib import *

import sonification_input_module as InputModule
from sonification_communication_module import *

# ---------------------------------------------------------------- #
# Global Arrays
global LATENT_FAST, SPIKES_FAST, PHASE_DIFF_FAST
SPIKES_FAST = [np.zeros((50, 1))]
LATENT_FAST = [np.zeros((200, 2))]
PHASE_DIFF_FAST = [0]
LATENT_SLOW = [np.zeros((200, 2))]
SPIKES_SLOW = [np.zeros((50, 1))]
PHASE_DIFF_SLOW = [0]

# Common Parameters
dt = 1e-3  # 1ms for dynamic system update
num_neurons = 50
TARGET_FIRING_RATE = 5
TARGET_SNR = 0


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


class LatentDynamics:
    INPUT_MAX = 0
    INPUT_X = 0
    INPUT_Y = 0

    def __init__(self, cycle_info, verbose=False):
        self.dt = cycle_info["dt"]
        self.reference_cycle = limit_circle(**cycle_info)
        self.perturb_cycle = limit_circle(**cycle_info)
        self.coupled_cycle = two_limit_circle(self.reference_cycle, self.perturb_cycle)

        self.latent = self.coupled_cycle.get_state()
        self.phase_diff = self.coupled_cycle.get_phase_diff().item()
        self.verbose = verbose

        # pass the handlers to the dispatcher
        DISPATCHER.map("/INPUT_MAX", self.max_to_latent_osc_handler)
        DISPATCHER.map("/MOTION_ENERGY", self.camera_to_latent_osc_handler)

    async def start(self):
        await self.setup_server()
        while True:
            start_t = time.perf_counter_ns()
            u = np.stack([self.INPUT_X, self.INPUT_Y], axis=0) / 10
            # u = self.INPUT_X

            self.coupled_cycle.update_state(u)
            self.latent = self.get_state()
            # print()
            self.phase_diff = self.coupled_cycle.get_phase_diff().item()

            elapsed_time = time.perf_counter_ns() - start_t
            sleep_duration = np.fmax(dt * 1e9 - (time.perf_counter_ns() - start_t), 0)

            if sleep_duration == 0 and self.verbose:
                print(
                    f"Dynamical system Iteration took {elapsed_time/1e6}ms which is longer than {dt*1e3} ms"
                )
            await busy_timer(sleep_duration)

    def get_state(self):
        return self.coupled_cycle.get_state()

    async def setup_server(self):
        # python-osc method for establishing the UDP communication with max
        # Connect server if address not in use (check with try catch)
        try:
            server_max = AsyncIOOSCUDPServer(
                (SERVER_IP, MAX_INPUT_PORT), DISPATCHER, asyncio.get_event_loop()
            )
            transport_max, _ = await server_max.create_serve_endpoint()
        except:
            print("Address already in use")

        try:
            server_motion = AsyncIOOSCUDPServer(
                (SERVER_IP, MOTION_ENERGY_PORT), DISPATCHER, asyncio.get_event_loop()
            )
            transport_motion, _ = await server_motion.create_serve_endpoint()
        except:
            print("Address already in use")

        # return transport_max, transport_motion

    def max_to_latent_osc_handler(self, address, *args):
        INPUT_MAX = args[0]
        if self.verbose:
            print(f"Received update for INPUT_MAX : {args}")

    def camera_to_latent_osc_handler(self, address, *args):
        INPUT_X = args[0]
        INPUT_Y = args[1]
        if self.verbose:
            print(f"Received update for INPUT_X, INPUT_Y : {args}")


class SpikeGenerator:
    def __init__(self, C, b, dt, latent_block: LatentDynamics):
        self.num_neurons = C.shape[1]
        self.dt = dt
        # Imposing explicit structure for testing...
        # for i in range(10):
        #     C[:, 5 * int(i) : 5 * int(i) + 5] = C[:, i][:, None]

        self.C = C
        self.b = b
        self.latent_block = latent_block
        self.firing_rates = np.exp(self.latent_block.get_state() @ self.C + self.b)
        self.y = np.random.poisson(self.firing_rates)

    async def start(self, spike):
        while True:
            self.firing_rates = np.exp(self.latent_block.get_state() @ self.C + self.b)
            self.y = np.random.poisson(self.firing_rates)
            spike[0] = self.y
            await busy_timer(self.dt * 1e9)


class LatentInference:
    def __init__(self):
        pass


# ----------------- Loop Components  ------------------- #
def simulate_neuron_parameters(
    cycle_info, num_neurons, target_SNR, target_rate_per_bin
):
    dt = cycle_info["dt"]
    reference_cycle = limit_circle(**cycle_info)
    perturb_cycle = limit_circle(**cycle_info)
    two_cycle = two_limit_circle(reference_cycle, perturb_cycle)

    latent_trajectory = two_cycle.generate_trajectory(1000)
    latent_dim = latent_trajectory.shape[1]

    C = np.random.randn(latent_dim, num_neurons)  # loading matrix
    b = 1.0 * np.random.rand(1, num_neurons) + np.log(target_rate_per_bin)  # bias

    C = generate_random_loading_matrix(latent_dim, num_neurons, 1, 0, C=C)

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
    asyncio.run(init_main())

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
# Common Parameters
dt = 1e-3  # 1ms for dynamic system update
SPIKES = [np.zeros((100, 1))]


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

    def __init__(self):
        # pass the handlers to the dispatcher
        DISPATCHER.map("/SPIKES", self.latent_to_inference_osc_handler)

    async def start(self, verbose=False):
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

    async def setup_server(self):
        # python-osc method for establishing the UDP communication with max
        # Connect server if address not in use (check with try catch)
        try:
            server_motion = AsyncIOOSCUDPServer(
                (SERVER_IP, SPIKE_PORT_2), DISPATCHER, asyncio.get_event_loop()
            )
            transport_motion, _ = await server_motion.create_serve_endpoint()
        except:
            print("Address already in use")

        # return transport_max, transport_motion

    def latent_to_inference_osc_handler(self, address, *args):
        SPIKES[0] = args[0]
        if self.verbose:
            print(f"Received update for INPUT_MAX : {args}")


if __name__ == "__main__":
    inference = LatentInference()
    asyncio.run(inference.start())

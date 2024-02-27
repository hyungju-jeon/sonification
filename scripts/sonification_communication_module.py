import asyncio
import time
import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import AsyncIOOSCUDPServer


# ----------------------- OSC Related Stuff ---------------------- #
# OSC ips / ports
global SERVER_IP, MOTION_ENERGY_PORT, MAX_INPUT_PORT, SPIKE_PORT, LATENT_PORT, DISPATCHER
SERVER_IP = "127.0.0.1"
SIGNAL_PORT = 1110
MOTION_ENERGY_PORT = 1111
SPIKE_PORT = 1112
LATENT_PORT = 1113
MAX_INPUT_PORT = 1211

DISPATCHER = Dispatcher()
DISPATCHER.set_default_handler(
    lambda address, *args: print(f"No action taken for message {address}: {args}")
)


async def busy_timer(duration):
    """
    A busy timer that blocks the event loop for a given duration.

    Args:
        duration (float): The duration of the busy timer in seconds.

    Returns:
        None
    """
    start = time.perf_counter_ns()
    while True:
        await asyncio.sleep(0)
        if time.perf_counter_ns() - start >= duration:
            break


async def trajectory_sending_loop(interval_ns, latent_block, verbose=False):
    """
    Sends packets of data to Max/MSP at regular intervals.

    Args:
        interval (float): The interval between each packet sending.

    Returns:
        None
    """
    # global TRAJECTORY, PHASE, SPIKE, PHASE_DIFF
    OSCsender = SimpleUDPClient(SERVER_IP, LATENT_PORT)
    while True:
        start_t = time.perf_counter_ns()
        OSCsender.send_message(
            "/trajectory",
            [latent_block.INPUT_X, latent_block.INPUT_Y],
        )
        elapsed_time = time.perf_counter_ns() - start_t
        sleep_duration = np.fmax(interval_ns - (time.perf_counter_ns() - start_t), 0)

        if sleep_duration == 0 and verbose:
            print(
                f"Trajectory Communication took {elapsed_time/1e6} ms longer than {interval_ns/1e6} ms"
            )
        await busy_timer(interval_ns)


async def spike_sending_loop(interval_ns, fast_block, slow_block, verbose=False):
    """
    Sends spikes to Max/MSP at regular intervals.

    Args:
        interval (float): The interval between each spike sending.

    Returns:
        None
    """
    OSCsender = SimpleUDPClient(SERVER_IP, SPIKE_PORT)
    while True:
        start_t = time.perf_counter_ns()
        OSCsender.send_message(
            "/SPIKES",
            np.stack(
                [
                    fast_block.y[0] > 0,
                    slow_block.y[0] > 0,
                ],
                axis=0,
            ).tolist(),
        )
        elapsed_time = time.perf_counter_ns() - start_t
        sleep_duration = np.fmax(interval_ns - (time.perf_counter_ns() - start_t), 0)

        if sleep_duration == 0 and verbose:
            print(
                f"Spike Communication took {elapsed_time/1e6} ms longer than {interval_ns/1e6} ms"
            )
        await busy_timer(interval_ns)

import asyncio
import time
import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import AsyncIOOSCUDPServer


# ----------------------- OSC Related Stuff ---------------------- #
# OSC ips / ports
global SERVER_IP, MOTION_ENERGY_PORT, MAX_INPUT_PORT, SPIKE_PORT, SPIKE_PORT_2, TRUE_LATENT_PORT, INFERRED_LATENT_PORT, DISPATCHER, TIMER_DELAY
LOCAL_SERVER = "127.0.0.1"
MAX_SERVER = "192.168.0.3"
SIGNAL_PORT = 1110
MOTION_ENERGY_PORT = 1111
SPIKE_PORT = 1112
SPIKE_PORT_2 = 1113
TRUE_LATENT_PORT = 1114
INFERRED_LATENT_PORT = 1114
MAX_INPUT_PORT = 1211
MAX_CONTROL_PORT = 1212
TIMER_DELAY = 1

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
        if time.perf_counter_ns() - start >= duration * TIMER_DELAY:
            break


async def trajectory_sending_loop(interval_ns, fask_block, slow_block, verbose=False):
    """
    Sends packets of data to Max/MSP at regular intervals.

    Args:
        interval (float): The interval between each packet sending.

    Returns:
        None
    """
    # global TRAJECTORY, PHASE, SPIKE, PHASE_DIFF
    OSCsender = SimpleUDPClient(LOCAL_SERVER, TRUE_LATENT_PORT)
    while True:
        start_t = time.perf_counter_ns()
        OSCsender.send_message(
            "/trajectory",
            np.concatenate(
                [fask_block.get_state(), slow_block.get_state()], axis=0
            ).tolist(),
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
    OSCsender = SimpleUDPClient(LOCAL_SERVER, SPIKE_PORT)
    OSCsender_2 = SimpleUDPClient(LOCAL_SERVER, SPIKE_PORT_2)
    while True:
        start_t = time.perf_counter_ns()
        msg = np.stack(
            [
                fast_block.y[0] > 0,
                slow_block.y[0] > 0,
            ],
            axis=0,
        ).tolist()
        OSCsender.send_message(
            "/SPIKES",
            msg,
        )
        OSCsender_2.send_message(
            "/SPIKES",
            msg,
        )
        elapsed_time = time.perf_counter_ns() - start_t
        sleep_duration = np.fmax(interval_ns - (time.perf_counter_ns() - start_t), 0)

        if sleep_duration == 0 and verbose:
            print(
                f"Spike Communication took {elapsed_time/1e6} ms longer than {interval_ns/1e6} ms"
            )
        await busy_timer(interval_ns)

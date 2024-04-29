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
MAX_SERVER = "127.0.0.1"
SIGNAL_PORT = 1110
MOTION_ENERGY_PORT = 1111
SPIKE_VISUALIZE_PORT = 1112
SPIKE_INFERENCE_PORT = 1113
TRUE_LATENT_PORT = 1114
INFERRED_LATENT_PORT = 1114
INFERENCE_LATENT_PORT = 1115
MAX_INPUT_PORT = 1211
MAX_CONTROL_PORT = 1211
MAX_OUTPUT_PORT = 1212
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


async def true_latent_sending_loop(interval_ns, slow_block, verbose=False):
    """
    Sends packets of data to Max/MSP at regular intervals.

    Args:
        interval (float): The interval between each packet sending.

    Returns:
        None
    """
    # global TRAJECTORY, PHASE, SPIKE, PHASE_DIFF
    MAX_OSCsender = SimpleUDPClient(MAX_SERVER, MAX_OUTPUT_PORT)
    LOCAL_OSCsender = SimpleUDPClient(LOCAL_SERVER, TRUE_LATENT_PORT)
    INFERENCE_OSCsender = SimpleUDPClient(LOCAL_SERVER, INFERENCE_LATENT_PORT)
    while True:
        start_t = time.perf_counter_ns()
        MAX_OSCsender.send_message(
            "/TRAJECTORY",
            np.concatenate([slow_block.get_state()], axis=0).tolist(),
        )

        LOCAL_OSCsender.send_message(
            "/TRAJECTORY",
            np.concatenate([slow_block.get_state()], axis=0).tolist(),
        )
        INFERENCE_OSCsender.send_message(
            "/TRAJECTORY",
            np.concatenate([slow_block.get_state()], axis=0).tolist(),
        )
        elapsed_time = time.perf_counter_ns() - start_t
        sleep_duration = np.fmax(interval_ns - (time.perf_counter_ns() - start_t), 0)

        if sleep_duration == 0 and verbose:
            print(
                f"True Trajectory Communication took {elapsed_time/1e6} ms longer than {interval_ns/1e6} ms"
            )
        await busy_timer(interval_ns)


async def fake_latent_sending_loop(interval_ns, slow_block, verbose=False):
    """
    Sends packets of data to Max/MSP at regular intervals.

    Args:
        interval (float): The interval between each packet sending.

    Returns:
        None
    """
    # global TRAJECTORY, PHASE, SPIKE, PHASE_DIFF
    MAX_OSCsender = SimpleUDPClient(MAX_SERVER, MAX_OUTPUT_PORT)
    LOCAL_OSCsender = SimpleUDPClient(LOCAL_SERVER, INFERRED_LATENT_PORT)
    INFERENCE_OSCsender = SimpleUDPClient(LOCAL_SERVER, INFERENCE_LATENT_PORT)
    while True:
        start_t = time.perf_counter_ns()
        MAX_OSCsender.send_message(
            "/INFERRED_TRAJECTORY",
            np.concatenate(
                [slow_block.get_state() + np.random.randn(4) * 0.02], axis=0
            ).tolist(),
        )

        LOCAL_OSCsender.send_message(
            "/INFERRED_TRAJECTORY",
            np.concatenate(
                [slow_block.get_state() + np.random.randn(4) * 0.02], axis=0
            ).tolist(),
        )
        elapsed_time = time.perf_counter_ns() - start_t
        sleep_duration = np.fmax(interval_ns - (time.perf_counter_ns() - start_t), 0)

        if sleep_duration == 0 and verbose:
            print(
                f"Fake Trajectory Communication took {elapsed_time/1e6} ms longer than {interval_ns/1e6} ms"
            )
        await busy_timer(interval_ns)


async def phase_diff_sending_loop(interval_ns, slow_block, verbose=False):
    """
    Sends packets of data to Max/MSP at regular intervals.

    Args:
        interval (float): The interval between each packet sending.

    Returns:
        None
    """
    # global TRAJECTORY, PHASE, SPIKE, PHASE_DIFF
    MAX_OSCsender = SimpleUDPClient(MAX_SERVER, MAX_OUTPUT_PORT)
    while True:
        start_t = time.perf_counter_ns()
        MAX_OSCsender.send_message(
            "/PHASE_DIFF",
            [slow_block.get_phase_diff()],
        )
        elapsed_time = time.perf_counter_ns() - start_t
        sleep_duration = np.fmax(interval_ns - (time.perf_counter_ns() - start_t), 0)

        if sleep_duration == 0 and verbose:
            print(
                f"Phase Diff Communication took {elapsed_time/1e6} ms longer than {interval_ns/1e6} ms"
            )
        await busy_timer(interval_ns)


async def spike_sending_loop(interval_ns, slow_block, verbose=False):
    """
        Sends spikes to Max/MSP at regular intervals.

        Args:
            interval (float): The interval between each spike sending.

    Returns:
            None
    """
    MAX_OSCsender = SimpleUDPClient(MAX_SERVER, MAX_OUTPUT_PORT)
    local_inference_OSCsender = SimpleUDPClient(LOCAL_SERVER, SPIKE_INFERENCE_PORT)
    local_visualize_OSCsender = SimpleUDPClient(LOCAL_SERVER, SPIKE_VISUALIZE_PORT)
    while True:
        start_t = time.perf_counter_ns()
        msg = np.concatenate(
            [
                slow_block.y[0] > 0,
            ],
            axis=0,
        ).tolist()
        MAX_OSCsender.send_message(
            "/SPIKES",
            msg,
        )
        local_inference_OSCsender.send_message(
            "/SPIKES",
            msg,
        )
        local_visualize_OSCsender.send_message(
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

# %%
from doctest import SKIP
import sys
import time
import torch
import asyncio

from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import AsyncIOOSCUDPServer

from utils.ndlib.vislib import *
from utils.ndlib.dynlib import *
from utils.ndlib.dslib import *


# ------------------ OSC ips / ports ------------------ #
# connection parameters
SERVER_IP = "127.0.0.1"
RCV_PORT = 1415
SEND_PORT = 1123

# ------------------- Global Arrays ----------------------
x_input = [0]
y_input = [0]
user_input = [0]
TRAJECTORY_1 = [np.zeros((200, 4))]
TRAJECTORY_2 = [np.zeros((200, 4))]
SPIKES_1 = [np.zeros((50, 1))]
SPIKES_2 = [np.zeros((50, 1))]
PHASE_DIFF_1 = [0]
PHASE_DIFF_2 = [0]
target_SNR = 0


# ---------------------- Timing  ----------------------- #
def ms_to_ns(ms):
    return ms * 1e6


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


# ------------------ OSC Related Stuff ------------------- #
py_to_max_OscSender = SimpleUDPClient(SERVER_IP, SEND_PORT)


def max_to_python_osc_handler(address, *args):
    """
    Handle OSC messages received from Max and update global variables accordingly.

    Parameters:
        address (str): The OSC address of the received message.
                       !! IMPORTANT !!
                       OSC address will be used as variable name. Make sure to match them!
        *args: Variable number of arguments representing the values of the OSC message.
    """
    exec("global " + address[1:])
    exec(address[1:] + "[0] = args[0]")
    print(f"Received message {address}: {args}")


def max_to_python_update_mean_firing(address, *args):
    global C, b
    cycle_info = {
        "x0": np.array([0.5, 0]),
        "d": 1,
        "w": 2 * np.pi * 10,  # 5 Hz
        "Q": np.array([[1e-10, 0.0], [0.0, 1e-10]]),
        "dt": dt,
    }
    new_C, new_b = simulate_neuron_parameters(cycle_info, 50, target_SNR, args[0] * dt)
    C = new_C
    b = new_b


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


async def spike_sending_loop(interval_ns, trajectory, spikes, verbose=False):
    """
    Sends spikes to Max/MSP at regular intervals.

    Args:
        interval (float): The interval between each spike sending.

    Returns:
        None
    """
    global C, b
    C[:, :10] = np.repeat(C[:, 0], 10, axis=0)
    C[:, 10:20] = np.repeat(C[:, 1], 10, axis=0)
    C[:, 20:30] = np.repeat(C[:, 2], 10, axis=0)
    C[:, 30:40] = np.repeat(C[:, 3], 10, axis=0)
    C[:, 40:] = np.repeat(C[:, 4], 10, axis=0)
    while True:
        start_t = time.perf_counter_ns()
        py_to_max_OscSender.send_message(
            "/spike",
            spikes[0][:, 0],
        )
        firing_rates = np.exp(trajectory[0][-1, :] @ C + b)

        # Poisson Model
        y = np.random.poisson(firing_rates)
        spikes[0][:, 0] = y

        elapsed_time = time.perf_counter_ns() - start_t
        sleep_duration = np.fmax(interval_ns - (time.perf_counter_ns() - start_t), 0)
        if sleep_duration == 0 and verbose:
            print(
                f"Spike Iteration took {elapsed_time/1e6} ms longer than {interval_ns/1e6} ms"
            )
        await busy_timer(interval_ns)


async def trajectory_sending_loop(interval_ns, verbose=False):
    """
    Sends packets of data to Max/MSP at regular intervals.

    Args:
        interval (float): The interval between each packet sending.

    Returns:
        None
    """
    # global TRAJECTORY, PHASE, SPIKE, PHASE_DIFF
    while True:
        start_t = time.perf_counter_ns()
        # Trajectory 1
        py_to_max_OscSender.send_message(
            "/trajectory_1/latent1",
            TRAJECTORY_1[0][-1, 1],
        )
        py_to_max_OscSender.send_message(
            "/trajectory_1/latent2",
            TRAJECTORY_1[0][-1, 3],
        )
        py_to_max_OscSender.send_message(
            "/trajectory_1/phase_diff",
            PHASE_DIFF_1,
        )
        # Trajectory 2
        py_to_max_OscSender.send_message(
            "/trajectory_2/latent1",
            TRAJECTORY_2[0][-1, 1],
        )
        py_to_max_OscSender.send_message(
            "/trajectory_2/latent2",
            TRAJECTORY_2[0][-1, 3],
        )
        py_to_max_OscSender.send_message(
            "/trajectory_2/phase_diff",
            PHASE_DIFF_2,
        )
        elapsed_time = time.perf_counter_ns() - start_t
        sleep_duration = np.fmax(interval_ns - (time.perf_counter_ns() - start_t), 0)
        if sleep_duration == 0 and verbose:
            print(
                f"Trajectory Iteration took {elapsed_time/1e6}ms longer than {interval_ns/1e6} ms"
            )
        await busy_timer(interval_ns)


async def dynamical_system_loop(
    cycle_info, trajectory, phase_diff, visualize=True, verbose=False
):
    dt = cycle_info["dt"]
    reference_cycle = limit_circle(**cycle_info)
    perturb_cycle = limit_circle(**cycle_info)
    two_cycle = two_limit_circle(reference_cycle, perturb_cycle)
    if visualize:
        fig = plt.figure(figsize=(9, 3))
        ax_ref = fig.add_subplot(1, 3, 1)
        ax_perturb = fig.add_subplot(1, 3, 2)
        ax_phase = fig.add_subplot(1, 3, 3)
        plot_info = {"xlim": (-1.1, 1.1), "ylim": (-1.1, 1.1)}
        refTraj = BlitPlot(
            np.zeros((1, 2)),
            "trajectory",
            fig=fig,
            ax=ax_ref,
            **plot_info,
            title="Reference trajectory",
        )
        pertTraj = BlitPlot(
            np.zeros((1, 2)),
            "trajectory",
            fig=fig,
            ax=ax_perturb,
            **plot_info,
            title="Perturbed trajectory",
        )
        phaseTraj = BlitPlot(
            np.zeros((1, 2)),
            "trajectory",
            fig=fig,
            ax=ax_phase,
            **plot_info,
            title="Direction (Phase difference)",
        )

    phase = np.zeros((trajectory[0].shape[0], 2))
    while True:
        start_t = time.perf_counter_ns()

        def transform_input(x):
            return np.max([(x - 3), 0])

        u = user_input[0] * 10 + transform_input(x_input[0]) * 10
        two_cycle.update_state(u)
        trajectory[0][0, :] = two_cycle.get_state()
        trajectory[0] = np.roll(trajectory[0], -1, axis=0)
        # print()
        phase_diff[0] = two_cycle.get_phase_diff().item()
        phase[0, :] = [np.cos(phase_diff[0]), np.sin(phase_diff[0])]
        phase = np.roll(phase, -1, axis=0)
        if visualize:
            refTraj.refresh(trajectory[0][:, :2])
            pertTraj.refresh(trajectory[0][:, 2:])
            phaseTraj.refresh(np.vstack([phase, [0, 0]]))
            fig.canvas.flush_events()

        elapsed_time = time.perf_counter_ns() - start_t
        sleep_duration = np.fmax(dt * 1e9 - (time.perf_counter_ns() - start_t), 0)

        if sleep_duration == 0 and verbose:
            print(
                f"Dynamical system Iteration took {elapsed_time/1e6}ms which is longer than {dt*1e3} ms"
            )
        await busy_timer(sleep_duration)


async def init_main():
    # ----------------------------------------------------------- #
    # ------------------ OSC Receiver from Max ------------------ #
    dispatcher = Dispatcher()

    # pass the handlers to the dispatcher
    dispatcher.map("/x_input", max_to_python_osc_handler)
    dispatcher.map("/y_input", max_to_python_osc_handler)
    dispatcher.map("/user_input", max_to_python_osc_handler)
    dispatcher.map("/mean_firing_rate", max_to_python_update_mean_firing)

    # you can have a default_handler for messages that don't have dedicated handlers
    def default_handler(address, *args):
        print(f"No action taken for message {address}: {args}")

    dispatcher.set_default_handler(default_handler)

    # python-osc method for establishing the UDP communication with max
    server = AsyncIOOSCUDPServer(
        (SERVER_IP, RCV_PORT), dispatcher, asyncio.get_event_loop()
    )
    transport, protocol = await server.create_serve_endpoint()
    # ---------------------------------------------------------- #
    global dt
    dt = 1e-3  # 1ms for dynamic system update
    cycle_info_1 = {
        "x0": np.array([0.5, 0]),
        "d": 1,
        "w": 2 * np.pi * 10,  # 5 Hz
        "Q": np.array([[1e-10, 0.0], [0.0, 1e-10]]),
        "dt": dt,
    }
    cycle_info_2 = {
        "x0": np.array([1, 1]),
        "d": 1,
        "w": 2 * np.pi * 1,  # 10 Hz
        "Q": np.array([[1e-10, 0.0], [0.0, 1e-10]]),
        "dt": dt,
    }

    target_firing_rate = 5
    global C, b
    C, b = simulate_neuron_parameters(
        cycle_info_1, 50, target_SNR, target_firing_rate * dt
    )

    await asyncio.gather(
        trajectory_sending_loop(ms_to_ns(0.2), verbose=True),
        spike_sending_loop(ms_to_ns(0.2), TRAJECTORY_1, SPIKES_1, verbose=True),
        dynamical_system_loop(
            cycle_info_1, TRAJECTORY_1, PHASE_DIFF_1, visualize=False
        ),
        dynamical_system_loop(
            cycle_info_2, TRAJECTORY_2, PHASE_DIFF_2, visualize=False
        ),
    )
    transport.close()  # Clean up serve endpoint


if __name__ == "__main__":
    asyncio.run(init_main())

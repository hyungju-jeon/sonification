import queue
import time
import random
import torch
import asyncio

from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import AsyncIOOSCUDPServer

from utils.ndlib.vislib import *
from utils.ndlib.dynlib import *


def max_to_python_osc_handler(address, *args):
    exec("global " + address[1:])
    # print(f"{address[1:]}= {args[0]}")
    exec(address[1:] + "[0] = args[0]")
    # fixed_args = args[0]
    # user_input = args
    print(f"Received message {address}: {args}")


async def packet_sending_loop(duration):
    global traj, phase
    while True:
        start = time.perf_counter_ns()
        while True:
            await asyncio.sleep(0)  # Yield control to allow other tasks to run
            if time.perf_counter_ns() - start >= duration:
                break
        # print(f"Current x: {x[0]}")

        py_to_max_OscSender.send_message(
            "/spike",
            traj[-1, 0],
        )

        py_to_max_OscSender.send_message(
            "/trajectory/latent1",
            traj[-1, 1],
        )
        py_to_max_OscSender.send_message(
            "/trajectory/latent2",
            traj[-1, 3],
        )
        # print(f"Sending message")


# ------------------ OSC ips / ports ------------------ #
# connection parameters
ip = "127.0.0.1"
receiving_from_max_port = 1415
sending_to_max_port = 1123

# ------------------- Global Arrays ----------------------
x_input = [0]
y_input = [0]
user_input = [0]
traj = np.zeros((200, 4))
phase = np.zeros((200, 2))

# ------------------ OSC Receiver from max ------------------ #
# create an instance of the osc_sender class above
py_to_max_OscSender = SimpleUDPClient(ip, sending_to_max_port)


async def dynamical_system_loop(visualize=True):
    global traj, phase
    obs_noise = 1e-5
    dt = 1e-3
    cycle_info = {
        "x0": torch.tensor([1.5, 0]),
        "d": 2,
        "w": (2 * np.pi / (dt)) / 1000,
        "Q": torch.tensor([[obs_noise, 0.0], [0.0, obs_noise]]),
        "dt": dt,
    }
    reference_cycle = LimitCircle(**cycle_info)
    perturb_cycle = LimitCircle(**cycle_info)
    twoC = TwoLimitCycle(reference_cycle, perturb_cycle, dt=1e-3)
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

    while True:

        def transform_input(x):
            return np.max([(x - 3), 0])

        u = user_input[0] * 10 + transform_input(x_input[0]) * 3
        # if mean_x[0] > 5:
        #     twoC.update_state(0.5)
        # elif mean_x[0] < 3:
        #     twoC.update_state(-0.5)
        # else:
        twoC.update_state(u)
        traj[0, :] = twoC.get_state()
        traj = np.roll(traj, -1, axis=0)
        phase_diff = twoC.get_phase_diff()
        phase[0, :] = [np.cos(phase_diff), np.sin(phase_diff)]
        phase = np.roll(phase, -1, axis=0)
        if visualize:
            refTraj.refresh(traj[:, :2])
            pertTraj.refresh(traj[:, 2:])
            phaseTraj.refresh(np.vstack([phase, [0, 0]]))
            fig.canvas.flush_events()

        t = time.time_ns()
        await asyncio.sleep(dt)
        # print(time.time_ns() / (10**9))


async def init_main():
    global user_input
    # Create datagram endpoint and start serving
    # Lists for storing received values
    quitFlag = [False]

    # ---------------------------------------------------------- #

    # ------------------ OSC Receiver from Max ------------------ #
    # dispatcher is used to assign a callback to a received osc message
    # in other words the dispatcher routes the osc message to the right action using the address provided
    dispatcher = Dispatcher()

    # define the handler for quit message message
    def quit_message_handler(address, *args):
        quitFlag[0] = True
        print("QUITTING!")

    # pass the handlers to the dispatcher
    dispatcher.map("/x_input", max_to_python_osc_handler)
    dispatcher.map("/y_input", max_to_python_osc_handler)
    dispatcher.map("/user_input", max_to_python_osc_handler)
    dispatcher.map("/quit*", quit_message_handler)

    # you can have a default_handler for messages that don't have dedicated handlers
    def default_handler(address, *args):
        print(f"No action taken for message {address}: {args}")

    dispatcher.set_default_handler(default_handler)

    # python-osc method for establishing the UDP communication with max
    server = AsyncIOOSCUDPServer(
        (ip, receiving_from_max_port), dispatcher, asyncio.get_event_loop()
    )
    transport, protocol = await server.create_serve_endpoint()
    # ---------------------------------------------------------- #
    await asyncio.gather(
        packet_sending_loop(1e6), dynamical_system_loop(visualize=True)
    )

    transport.close()  # Clean up serve endpoint


if __name__ == "__main__":
    asyncio.run(init_main())

# %%
import time
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
N = 200
INPUT = [0]
OUTPUT_VEC = [np.zeros((N, 1))]
PACKET_RECEIVE = 0
PACKET_SENT = 0

# ------------------ OSC Receiver from max ------------------ #
# create an instance of the osc_sender class above
py_to_max_OscSender = SimpleUDPClient(SERVER_IP, SEND_PORT)


def ms_to_ns(ms):
    return ms * 1e6


def max_to_python_osc_handler(address, *args):
    """
    Handle OSC messages received from Max and update global variables accordingly.

    Parameters:
        address (str): The OSC address of the received message.
                       !! IMPORTANT !!
                       OSC address will be used as variable name. Make sure to match them!
        *args: Variable number of arguments representing the values of the OSC message.
    """
    global PACKET_RECEIVE
    exec("global " + address[1:])
    exec(address[1:] + "[0] = args[0]")
    PACKET_RECEIVE += 1
    print(
        f"Received {PACKET_RECEIVE}/{PACKET_SENT}. Lost {PACKET_SENT - PACKET_RECEIVE} packets, {PACKET_RECEIVE/PACKET_SENT*100}% received"
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


async def python_to_max_handler(interval_ns, outputs):
    global PACKET_SENT
    while True:
        start_t = time.perf_counter_ns()
        py_to_max_OscSender.send_message(
            "/OUTPUT",
            outputs[0][:, 0],
        )
        PACKET_SENT += 1
        iter_duration = time.perf_counter_ns() - start_t
        sleep_duration = np.fmax(interval_ns - iter_duration, 0)
        if sleep_duration == 0:
            print(
                f"Spike Iteration took {iter_duration} which is longer than {interval_ns/1e6} ms"
            )
        await busy_timer(interval_ns)


async def init_main():
    dt = 0.1
    # ---------------------------------------------------------- #
    dispatcher = Dispatcher()
    dispatcher.map("/INPUT", max_to_python_osc_handler)

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
    await asyncio.gather(python_to_max_handler(ms_to_ns(dt), OUTPUT_VEC))
    transport.close()  # Clean up serve endpoint


if __name__ == "__main__":
    asyncio.run(init_main())

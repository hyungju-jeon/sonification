import asyncio

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

from utils.ndlib.dynlib import *
from utils.ndlib.vislib import *

x = [0]
K = 10000
T = np.zeros(K)
i = 0


# ------------------ OSC ips / ports ------------------ #
# connection parameters
ip = "127.0.0.1"
receiving_from_max_port = 1415
sending_to_max_port = 1123

# ------------------ OSC Receiver from max ------------------ #
# create an instance of the osc_sender class above
py_to_max_OscSender = SimpleUDPClient(ip, sending_to_max_port)


def max_to_python_osc_handler(address, fixed_args, *osc_args):
    print(f"T = {fixed_args}")
    py_to_max_OscSender.send_message("/msg", [0])


def elapsed_time_update(address, *args):
    global i
    if i == K:
        return
    T[i] = args[0]
    i += 1


async def timed_message_send(duration):
    global i
    while i < K:
        await asyncio.sleep(0)


async def init_main():
    quitFlag = [False]
    # ------------------ OSC Receiver from Max ------------------ #
    # dispatcher is used to assign a callback to a received osc message
    # in other words the dispatcher routes the osc message to the right action using the address provided
    dispatcher = Dispatcher()

    # define the handler for quit message message
    def quit_message_handler(address, *args):
        quitFlag[0] = True
        print("QUITTING!")

    # pass the handlers to the dispatcher
    dispatcher.map("/spike*", max_to_python_osc_handler, T)
    dispatcher.map("/time*", elapsed_time_update)
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
    await timed_message_send(1e7)  # Enter main loop of program
    transport.close()  # Clean up serve endpoint


if __name__ == "__main__":
    asyncio.run(init_main())

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(T)
    ax1.set(xlabel="Sample", ylabel="Latency (us)")
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.hist(T, bins=100)
    ax2.set(xlabel="Measured Latency (us)", ylabel="Hist")
    plt.suptitle("Latency (us)")
    print(
        f"mean = {np.mean(T)}, std = {np.std(T)} and absolute deviation = {np.mean(np.abs(T - 1.0))} ms"
    )
    plt.show()
    print("init_main done")

import queue
import time
import random

from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient


if __name__ == "__main__":

    # Lists for storing received values
    mean_x = [0]
    mean_y = [0]
    quitFlag = [False]

    # ------------------ OSC ips / ports ------------------ #
    # connection parameters
    ip = "127.0.0.1"
    receiving_from_max_port = 1415
    sending_to_max_port = 1123

    # ----------------------------------------------------------

    # ------------------ OSC Receiver from max ------------------ #
    # create an instance of the osc_sender class above
    py_to_max_OscSender = SimpleUDPClient(ip, sending_to_max_port)
    # ---------------------------------------------------------- #

    # ------------------ OSC Receiver from Max ------------------ #
    # dispatcher is used to assign a callback to a received osc message
    # in other words the dispatcher routes the osc message to the right action using the address provided
    dispatcher = Dispatcher()

    # define the handler for messages starting with /mean_x]
    def mean_x_message_handler(address, *args):
        mean_x[0] = args[0]

    def mean_y_message_handler(address, *args):
        mean_y[0] = args[0]

    # define the handler for quit message message
    def quit_message_handler(address, *args):
        quitFlag[0] = True
        print("QUITTING!")

    # pass the handlers to the dispatcher
    dispatcher.map("/mean_x*", mean_x_message_handler)
    dispatcher.map("/mean_y*", mean_y_message_handler)
    dispatcher.map("/quit*", quit_message_handler)

    # you can have a default_handler for messages that don't have dedicated handlers
    def default_handler(address, *args):
        print(f"No action taken for message {address}: {args}")

    dispatcher.set_default_handler(default_handler)

    # python-osc method for establishing the UDP communication with max
    server = BlockingOSCUDPServer((ip, receiving_from_max_port), dispatcher)
    # ---------------------------------------------------------- #

    # ------------------ NOTE GENERATION  ------------------ #

    while quitFlag[0] is False:
        server.handle_request()
        print(f"Received mean_x value {mean_x[0]}")
        print(f"Received mean_y value {mean_y[0]}")

        # generate a random pitched note with the provided received value
        pitch = int(random.randrange(40, 52))  # 1 octave
        vel = mean_x[0]
        duration = int(random.randrange(0, 1000))

        # Send Notes to max (send pitch last to ensure syncing)
        py_to_max_OscSender.send_message("/test1/mean_x", [vel, vel / 2, vel / 3])
        py_to_max_OscSender.send_message("/test1/pitch", pitch)

        py_to_max_OscSender.send_message("/test2/mean_x", duration)
        py_to_max_OscSender.send_message("/test2/pitch", pitch)

    # ---------------------------------------------------------- #

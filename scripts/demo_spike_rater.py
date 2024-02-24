import sys
import numpy as np
import asyncio
import pyqtgraph as pg
import time
from pyqtgraph.Qt import QtCore
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer

import threading

spike = [np.zeros((50, 1))]
SERVER_IP = "127.0.0.1"
pg.setConfigOptions(useOpenGL=True)
# pg.setConfigOption("antialias", True)
count = 0


def max_to_python_osc_handler(address, *args):
    """
    Handle OSC messages received from Max and update global variables accordingly.

    Parameters:
        address (str): The OSC address of the received message.
                       !! IMPORTANT !!
                       OSC address will be used as variable name. Make sure to match them!
        *args: Variable number of arguments representing the values of the OSC message.
    """
    global spike, count
    exec(address[1:] + "[0] = np.array(args)")
    # print(f"Received message {address}: {args}")


async def init_main():
    # ----------------------------------------------------------- #
    # ------------------ OSC Receiver from Max ------------------ #
    dispatcher = Dispatcher()

    # pass the handlers to the dispatcher
    dispatcher.map("/spike", max_to_python_osc_handler)

    # you can have a default_handler for messages that don't have dedicated handlers
    def default_handler(address, *args):
        # print(f"No action taken for message {address}: {args}")
        pass

    dispatcher.set_default_handler(default_handler)

    # python-osc method for establishing the UDP communication with max
    server = AsyncIOOSCUDPServer(
        (SERVER_IP, 1123), dispatcher, asyncio.get_event_loop()
    )
    transport, protocol = await server.create_serve_endpoint()
    while True:
        try:
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            break


def asyncio_run():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(init_main())


class NeuronVisualizer:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.firing_rates = np.zeros(
            (1000, num_neurons)
        )  # Initialize firing rates for each neuron
        self.decay_factor = 0.99  # Exponential decay factor

        # Create a PyQtGraph window
        self.app = QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, size=(300, 600))
        self.win.setWindowTitle("Firing Rates of Neurons Over Time")
        self.p1 = self.win.addPlot(title="Firing Rates of Neurons")
        self.img = pg.ImageItem()
        self.img.setLevels((0, 100))
        self.p1.addItem(self.img)
        self.p1.hideAxis("bottom")
        self.p1.setLabel("left", "Neuron", "")
        # self.win.setWindowSize(800, 600)  # Set fixed window size
        self.frame = 0

        # Create a custom lookup table (LUT) with green-neon color
        lut = []
        for i in range(256):
            # Use RGB value (0, 255, 65) for green-neon color
            lut.append([0, 255, 65, i])  # Add alpha channel as the last value
        self.img.setLookupTable(lut)

    def animation(self):
        timer = QTimer()
        timer.timeout.connect(self.update_firing_rates)
        timer.start(0)
        self.start()

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QApplication.instance().exec_()

    def update_firing_rates(self):
        self.frame += 1
        stime = time.perf_counter_ns()
        # Update firing rates based on input vector x
        self.firing_rates = np.roll(self.firing_rates, -1, axis=0)
        self.firing_rates[-1, :] = self.firing_rates[-2, :] * self.decay_factor

        for i, firing_event in enumerate(spike[0]):
            if firing_event > 0:  # If there's firing
                self.firing_rates[-1, i] += 5  # Instantly increase firing rate
        if self.frame % 10 == 0:
            self.img.setImage(self.firing_rates, autoLevels=False)
        print(
            f"Current frame {self.frame}, elapsed time: {(time.perf_counter_ns() - stime) / 1e6} ms"
        )


# Start the PyQtGraph event loop
if __name__ == "__main__":
    # asyncio.run(init_main())
    asyncio_thread = threading.Thread(target=asyncio_run)
    asyncio_thread.start()
    # Example usage:
    num_neurons = 50
    neuron_visualizer = NeuronVisualizer(num_neurons)
    neuron_visualizer.animation()

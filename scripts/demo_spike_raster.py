import sys

import numpy as np
import asyncio
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from pyqtgraph.Qt import QtCore
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimerEvent, QTimer
from PyQt5.QtCore import pyqtSignal

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer
from sonification_communication_module import *

import threading

SPIKES_FAST = [np.zeros((50, 1))]
SERVER_IP = "127.0.0.1"
pg.setConfigOptions(useOpenGL=True)
count = 0
num_neurons = 50


class RasterWithTraceVisualizer:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.L = 1000
        self.buffer = 1000
        self.firing_rates = np.zeros(
            (self.L + self.buffer, num_neurons)
        )  # Initialize firing rates for each neuron
        self.decay_factor = 0.99  # Exponential decay factor

        # Create a PyQtGraph window
        self.app = QApplication([])
        self.w = gl.GLViewWidget()
        self.w.setWindowTitle("pyqtgraph example: GLLinePlotItem")
        self.w.setGeometry(0, 110, 800, 600)
        self.w.opts["distance"] = 600
        self.w.show()

        tex1 = pg.makeRGBA(self.firing_rates, levels=(0, 20))[0]
        self.img = gl.GLImageItem(tex1)
        scale_factor = [0.5, 10, 1]
        self.img.scale(*scale_factor)
        self.img.translate(
            -scale_factor[1] * self.num_neurons / 2, -scale_factor[0] * self.L / 2, 50
        )
        self.img.rotate(90, 0, 0, 1)
        self.w.addItem(self.img)
        self.frame = 0

        # Create a custom lookup table (LUT) with green-neon color
        self.lut = []
        for i in range(256):
            # Use RGB value (0, 255, 65) for green-neon color
            self.lut.append([0, 255, 65, i])  # Add alpha channel as the last value

    def animation(self):
        self.prev_count = 0
        self.count = 0
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QApplication.instance().exec_()

    def update_firing_rates(self):
        self.prev_count = count
        self.count += 1
        if self.frame >= self.L + self.buffer:
            self.firing_rates = np.roll(self.firing_rates, -self.buffer, axis=0)
            self.frame = self.L

        if self.frame > 0:
            self.firing_rates[self.frame, :] = (
                self.firing_rates[self.frame - 1, :] * self.decay_factor
            )

        for i, firing_event in enumerate(SPIKES_FAST[0]):
            if firing_event > 0:  # If there's firing
                self.firing_rates[self.frame, i] += 5  # Instantly increase firing rate

        if self.frame % 16 == 0:
            tex1 = pg.makeRGBA(
                self.firing_rates[
                    np.fmax(0, self.frame - self.L) : np.fmax(self.L, self.frame), :
                ],
                levels=(0, 5),
                lut=self.lut,
            )[0]
            self.img.setData(tex1)
        self.frame += 1
        print(f"Frame: {self.frame}, Count: {self.count}")


neuron_visualizer = RasterWithTraceVisualizer(num_neurons)


class SpikePacer(QtCore.QObject):
    trigger = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.trigger.connect(neuron_visualizer.update_firing_rates)

    def max_to_python_osc_handler(self, address, *args):
        global SPIKES_FAST, count
        SPIKES_FAST[0] = np.array(args)
        self.trigger.emit()
        # print(f"Received message count {count}")


spike_pacer = SpikePacer()


async def init_main():
    # ----------------------------------------------------------- #
    # ------------------ OSC Receiver from Max ------------------ #
    dispatcher = Dispatcher()
    # pass the handlers to the dispatcher
    dispatcher.map("/SPIKES_FAST", spike_pacer.max_to_python_osc_handler)

    # python-osc method for establishing the UDP communication with max
    server = AsyncIOOSCUDPServer(
        (SERVER_IP, SPIKE_PORT), dispatcher, asyncio.get_event_loop()
    )
    transport, protocol = await server.create_serve_endpoint()
    while True:
        try:
            await asyncio.sleep(0)
        except KeyboardInterrupt:
            break


def asyncio_run():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(init_main())


# Start the PyQtGraph event loop
if __name__ == "__main__":
    asyncio_thread = threading.Thread(target=asyncio_run)
    asyncio_thread.start()
    neuron_visualizer.animation()

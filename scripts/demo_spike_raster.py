import sys
from matplotlib.pylab import f
import numpy as np
import asyncio
import pyqtgraph as pg
import pyqtgraph.opengl as gl
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
        self.decay_factor = 0.9  # Exponential decay factor

        # Create a PyQtGraph window
        self.app = QApplication([])
        # self.win = pg.GraphicsLayoutWidget(show=True, size=(800, 600))
        # self.win.setWindowTitle("Firing Rates of Neurons Over Time")
        self.w = gl.GLViewWidget()
        self.w.setWindowTitle("pyqtgraph example: GLLinePlotItem")
        self.w.setGeometry(0, 110, 800, 600)
        self.w.opts["distance"] = 200
        self.w.show()
        # self.p1 = self.win.addPlot(title="Firing Rates of Neurons")
        # self.p2 = self.win.addPlot(title="Firing Rates Line Plot")
        # self.img.setLevels((0, 100))
        tex1 = pg.makeRGBA(self.firing_rates, levels=(-0.5, 0.5))[0]
        self.img = gl.GLImageItem(tex1)
        self.img.scale(0.2, 5, 1)
        self.img.translate(-100, -25, 0)
        self.img.rotate(90, 0, 0, 1)
        # self.img.rotate(-90, 0, 1, 0)
        self.w.addItem(self.img)
        # self.p1.hideAxis("bottom")
        # self.p1.setLabel("left", "Neuron", "")
        # self.win.setWindowSize(800, 600)  # Set fixed window size
        self.frame = 0

        # Create a custom lookup table (LUT) with green-neon color
        self.lut = []
        for i in range(256):
            # Use RGB value (0, 255, 65) for green-neon color
            self.lut.append([0, 255, 65, i])  # Add alpha channel as the last value
        # self.img.setLookupTable(lut)

        self.timer = QTimer()
        # self.curve = []
        # for _ in range(num_neurons):
        #     curve = self.p2.plot(pen="g")
        #     self.curve.append(curve)

        # self.phase = 0
        # self.lines = 5
        # self.traces = dict()
        # self.points = 1000
        # self.y = np.linspace(-10, 10, self.lines)
        # self.x = np.linspace(-10, 10, self.points)
        # self.x = np.linspace(-10, 10, 1000)
        # for i, line in enumerate(self.y):
        #     y = np.array([line] * self.points)
        #     d = np.sqrt(self.x**2 + y**2)
        #     sine = 10 * np.sin(d + self.phase)
        #     pts = np.vstack([self.x, y, sine]).transpose()
        #     self.traces[i] = gl.GLLinePlotItem(
        #         pos=pts,
        #         color=pg.glColor((i, self.lines * 1.3)),
        #         width=(i + 1) / 10,
        #         antialias=True,
        #     )
        #     self.w2.addItem(self.traces[i])

    def animation(self):
        self.timer.timeout.connect(self.update_firing_rates)
        self.timer.start(0)
        self.start()
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QApplication.instance().exec_()

    def animation_1(self):
        self.timer.timeout.disconnect()
        self.timer.timeout.connect(self.update_firing_rates)
        self.timer.start(100)
        self.start()

    def start(self):
        pass

    def update_firing_rates(self):
        if self.frame > 1000:
            self.animation_1()
        self.frame += 1
        stime = time.perf_counter_ns()
        # Update firing rates based on input vector x
        self.firing_rates = np.roll(self.firing_rates, -1, axis=0)
        self.firing_rates[-1, :] = self.firing_rates[-2, :] * self.decay_factor

        for i, firing_event in enumerate(spike[0]):
            if firing_event > 0:  # If there's firing
                self.firing_rates[-1, i] += 5  # Instantly increase firing rate

        if self.frame % 10 == 0:
            tex1 = pg.makeRGBA(self.firing_rates, levels=(0, 50), lut=self.lut)[0]
            self.img.setData(tex1)
        # x = np.arange(1000)  # x-axis values (time steps)
        # stime = time.perf_counter_ns()
        # for i, curve in enumerate(self.curve):
        #     curve.setData(
        #         x, self.firing_rates[:, i] + 100 * i
        #     )  # Plot firing rates for each neuron
        # print(f"elapsed time: {(time.perf_counter_ns() - stime) / 1e6} ms")

        # for i, line in enumerate(self.y):
        #     y = np.array([line] * self.points)

        #     amp = 10 / (i + 1)
        #     phase = self.phase * (i + 1) - 10
        #     freq = self.x * (i + 1) / 10

        #     sine = amp * np.sin(freq - phase)
        #     pts = np.vstack([self.x, y, sine]).transpose()

        #     self.set_plotdata(
        #         name=i, points=pts, color=pg.glColor((i, self.lines * 1.3)), width=3
        #     )
        #     self.phase -= 0.0002

    def set_plotdata(self, name, points, color, width):
        self.traces[name].setData(pos=points, color=color, width=width)


# Start the PyQtGraph event loop
if __name__ == "__main__":
    # asyncio.run(init_main())
    asyncio_thread = threading.Thread(target=asyncio_run)
    asyncio_thread.start()
    # Example usage:
    num_neurons = 50
    neuron_visualizer = NeuronVisualizer(num_neurons)
    neuron_visualizer.animation()

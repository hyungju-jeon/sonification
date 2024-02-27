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

SERVER_IP = "127.0.0.1"
pg.setConfigOptions(useOpenGL=True)
packet_count = 0
num_neurons = 100
SPIKES = [np.zeros(num_neurons)]


class RasterWithTraceVisualizer:
    def __init__(self, decay_factor):
        self.num_neurons = num_neurons
        self.L = 1000
        self.buffer = 1000
        self.firing_rates = np.zeros(
            (self.L + self.buffer, num_neurons)
        )  # Initialize firing rates for each neuron
        self.decay_factor = decay_factor  # Exponential decay factor

        # Create a PyQtGraph window
        self.app = QApplication([])
        self.plot_widge = gl.GLViewWidget()
        self.plot_widge.setWindowTitle("pyqtgraph example: GLLinePlotItem")
        self.plot_widge.setGeometry(0, 110, 800, 600)
        self.plot_widge.opts["distance"] = 600
        self.plot_widge.show()

        tex1 = pg.makeRGBA(self.firing_rates, levels=(0, 20))[0]
        self.img = gl.GLImageItem(tex1)
        scale_factor = [0.5, 5, 1]
        self.img.scale(*scale_factor)
        self.img.translate(
            -scale_factor[1] * self.num_neurons / 2, -scale_factor[0] * self.L / 2, 50
        )
        self.img.rotate(90, 0, 0, 1)
        self.plot_widge.addItem(self.img)
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

    def update(self):
        self.count += 1
        if self.frame >= self.L + self.buffer:
            self.firing_rates = np.roll(self.firing_rates, -self.buffer, axis=0)
            self.frame = self.L

        if self.frame > 0:
            self.firing_rates[self.frame, :] = (
                self.firing_rates[self.frame - 1, :] * self.decay_factor
            )

        for i, firing_event in enumerate(SPIKES[0]):
            if firing_event > 0:  # If there's firing
                self.firing_rates[self.frame, i] += 5  # Instantly increase firing rate

        if (self.count > 1000) & (self.frame % 16 == 0):
            tex1 = pg.makeRGBA(
                self.firing_rates[np.fmax(0, self.frame - self.L) : self.frame, :],
                levels=(0, 5),
                lut=self.lut,
            )[0]
            self.img.setData(tex1)
        self.frame += 1


class SpikeBubbleVisualizer:
    def __init__(self):
        self.num_neurons = num_neurons
        # Create a PyQtGraph window
        self.app = QApplication([])
        self.plot_widget = gl.GLViewWidget()
        self.plot_widget.setWindowTitle("pyqtgraph example: GLLinePlotItem")
        self.plot_widget.setGeometry(0, 110, 800, 600)
        self.plot_widget.opts["distance"] = 50
        self.plot_widget.show()

        gx = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        gx.translate(-10, 0, 0)
        self.plot_widget.addItem(gx)
        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, -10, 0)
        self.plot_widget.addItem(gy)
        gz = gl.GLGridItem()
        gz.translate(0, 0, -10)
        self.plot_widget.addItem(gz)

        self.data = np.zeros(num_neurons)
        self.centroid_positions = [
            [np.random.uniform(-10, 10), np.random.uniform(-10, 10)]
            for _ in range(num_neurons)
        ]
        self.centroids = dict()
        self.traces = dict()

    def animation(self):
        self.prev_count = 0
        self.count = 0
        self.frame = 0
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QApplication.instance().exec_()

    def set_plotdata(self, name, points, color, width):
        self.traces[name].setData(pos=points, color=color, width=width)

    def update(self):
        self.data = SPIKES[0]
        # Check and remove all marked circles (finished ripple animations)
        self.remove_marked_circles()

        # Update centroid location and replot centroid indicator
        # self.move_centroids()
        # self.update_centroid_indicator()

        if self.frame % 16 == 0:
            if any(self.data > 0):
                self.trigger_spike(np.where(self.data > 0)[0])
        print(f"Frame: {self.frame}, Count: {packet_count}")
        self.frame += 1

    def trigger_spike(self, index):
        """
        Triggers a spike animation at the specified index.

        Args:
            index (int): The index of the spike to trigger.
        """
        pos = [self.centroid_positions[idx] for idx in index]

        # Add the ripple background glow element
        # ripple_bg = gl.GLScatterPlotItem()
        # ripple_bg.setData(pos=pos, size=30, color=(0, 1, 0.25, 0.8))
        # self.plot_widget.addItem(ripple_bg)
        # QTimer.singleShot(1, lambda: self.expand_circle((ripple_bg, 0)))

        # Add the ripple element
        ripple = gl.GLScatterPlotItem()
        ripple.setData(pos=pos, size=10, color=(0.0, 1, 0.25, 1))
        self.plot_widget.addItem(ripple)
        QTimer.singleShot(1, lambda: self.expand_circle((ripple, 0)))

        # Add the zapping effect element
        zap_effect = gl.GLScatterPlotItem()
        zap_effect.setData(
            pos=pos,
            size=10,
            color=(1, 1, 1, 1),
        )
        self.plot_widget.addItem(zap_effect)
        QTimer.singleShot(1, lambda: self.create_zap_effect(zap_effect))

    def create_zap_effect(self, circle):
        """
        Creates a zap effect for a circle by making it invisible after a delay.

        Args:
            circle (ScatterPlotItem): The circle to create the zap effect for.
        """
        QTimer.singleShot(100, lambda: circle.setData(color=(0, 0, 0, 0)))

    def expand_circle(self, circle_radius_tuple):
        """
        Expands a circle in the animation by gradually increasing its size.

        Args:
            circle (tuple): A tuple containing the circle item and its current radius.
        """
        circle, iter = circle_radius_tuple
        current_color = list(circle.color)
        radius = circle.size + 15
        if iter < 50 and current_color[-1] > 0.05 and radius < 1000:
            current_color[-1] *= 0.9
            circle.setData(size=radius, color=current_color)
            QTimer.singleShot(16, lambda: self.expand_circle((circle, iter + 1)))
        else:
            circle.setData(color=(0, 0, 0, 0))

    def remove_marked_circles(self):
        """
        Removes marked circles from the animation.
        """
        scatter_items = [
            item
            for item in self.plot_widget.items
            if isinstance(item, gl.GLScatterPlotItem)
        ]
        for item in scatter_items:
            if item.color[-1] == 0:
                self.plot_widget.removeItem(item)


# visualizer = RasterWithTraceVisualizer(0)
visualizer = SpikeBubbleVisualizer()


class SpikePacer(QtCore.QObject):
    trigger = pyqtSignal()

    def __init__(self, fcn):
        super().__init__()
        self.trigger.connect(fcn)

    def max_to_python_osc_handler(self, address, *args):
        global SPIKES, packet_count

        SPIKES[0][:50] = np.array(args[0])
        SPIKES[0][50:] = np.array(args[0])
        packet_count += 1
        self.trigger.emit()


spike_pacer = SpikePacer(visualizer.update)


async def init_main():
    dispatcher = Dispatcher()
    dispatcher.map("/SPIKES", spike_pacer.max_to_python_osc_handler)

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
    visualizer.animation()

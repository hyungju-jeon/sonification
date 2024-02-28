import sys

import numpy as np
import asyncio
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from pyqtgraph.Qt import QtCore
from PyQt5.QtWidgets import QApplication, QGraphicsView
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
DISC_RADIUS_INC = [20]
SEQ_TRIGGER = [0]


class RasterWithTrace3DVisualizer:
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


class SpikeBubble3DVisualizer:
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
        self.velocity = np.random.uniform(-1, 1, (num_neurons, 2))

    def animation(self):
        self.prev_count = 0
        self.count = 0
        self.frame = 0
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QApplication.instance().exec_()

    def update(self):
        self.data = SPIKES[0]
        # Check and remove all marked circles (finished ripple animations)
        self.remove_marked_circles()

        # Update centroid location and replot centroid indicator
        # self.update_centroid_indicator()

        if self.frame % 1 == 0:
            self.estimate_velocity()
            if any(self.data > 0):
                self.trigger_spike(np.where(self.data > 0)[0])
        print(f"Frame: {self.frame}, Count: {packet_count}")
        self.frame += 1

    def estimate_velocity(self):
        self.velocity = np.random.uniform(-1, 1, (num_neurons, 2))
        for i, center in enumerate(self.centroid_positions):
            self.centroid_positions[i] += 0.1 * self.velocity[i]

    def update_centroid_indicator(self):
        """
        Updates the centroid indicator positions in the animation.
        """
        pass
        # x, y = self.indicators.getData()
        # # get first two columns of pos
        # pos = np.column_stack((x, y))
        # # if new position is out of bounds, reverse the velocity
        # for i, center in enumerate(pos):
        #     self.velocity[i] = np.clip(self.velocity[i], -1, 1)
        #     if center[0] > 10 or center[0] < -10:
        #         self.velocity[i, 0] *= -1
        #     if center[1] > 10 or center[1] < -10:
        #         self.velocity[i, 1] *= -1
        # new_pos = pos + self.velocity * 0.1
        # for i, trace in enumerate(self.indicator_traces):
        #     trace.setData(
        #         x=[pos[i][0], new_pos[i][0]],
        #         y=[pos[i][1], new_pos[i][1]],
        #     )

        # self.indicators.setData(pos=new_pos, skipFiniteCheck=True)
        # self.centroid_positions = new_pos
        # print(f"msec : {(time.perf_counter_ns() - stime) /1e9}")

    def trigger_spike(self, index):
        """
        Triggers a spike animation at the specified index.

        Args:
            index (int): The index of the spike to trigger.
        """
        pos = [self.centroid_positions[idx] for idx in index]

        # Add the ripple element
        ripple = gl.GLScatterPlotItem()
        ripple.setData(pos=pos, size=10, color=(0.5, 1, 0.5, 0.9))
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

    def expand_circle(self, circle_iter_tuple):
        """
        Expands a circle in the animation by gradually increasing its size.

        Args:
            circle (tuple): A tuple containing the circle item and its current radius.
        """
        circle, iter = circle_iter_tuple
        current_color = list(circle.color)
        radius = circle.size + 5
        # print(circle.size)
        if current_color[-1] > 0.05 and radius < 100:
            current_color[-1] *= 0.9
            circle.setData(size=radius, color=current_color)
            QTimer.singleShot(10, lambda: self.expand_circle((circle, iter + 1)))
        else:
            # QTimer.singleShot(10, lambda: circle.setData(color=(0, 0, 0, 0)))
            circle.setData(size=0, color=(0, 0, 0, 0))

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


class SpikeDisc2DVisualizer:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setYRange(-10, 10)
        self.plot_widget.setXRange(-10, 10)
        self.plot_widget.setGeometry(0, 110, 1024, 768)
        # remove axis
        self.plot_widget.hideAxis("bottom")
        self.plot_widget.hideAxis("left")
        self.plot_widget.show()
        self.plot_widget.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)

        self.centroid_positions = np.array(
            [
                [np.random.uniform(-10, 10), np.random.uniform(-10, 10)]
                for _ in range(num_neurons)
            ]
        )
        self.draw_centroids()

        # Add function that will be called when the timer times out
        self.frame = 0

    def animation(self):
        self.prev_count = 0
        self.count = 0
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QApplication.instance().exec_()

    def draw_centroids(self):
        """
        Draws the centroid indicators in the animation.

        Returns:
            list: The centroid indicator items in the animation.
        """
        indicators = pg.ScatterPlotItem()
        indicators.setData(
            pos=self.centroid_positions,
            size=5,
            symbol="o",
            pen=pg.mkPen(width=0, color=(51, 255, 51, 0)),
            brush=(51, 255, 51, 0),
        )
        self.plot_widget.addItem(indicators)
        self.indicators = indicators
        self.size = np.ones(num_neurons) * 5
        self.color = np.repeat(
            np.array([51, 255, 51, 100])[np.newaxis, :], num_neurons, axis=0
        )

    def update(self):
        """
        Updates the animation by removing marked circles, updating the binary vector,
        animating points, moving centroids, and stopping the timer when the animation is complete.
        """
        # Check and remove all marked circles (finished ripple animations)
        # self.remove_marked_circles()

        # Update centroid location and replot centroid indicator
        # self.update_centroid_indicator()

        if self.frame % 10 == 0:
            self.shrink_circle()
            # self.estimate_velocity()
            # self.move_centroids()
            if any(SPIKES[0] > 0):
                self.trigger_spike(np.where(SPIKES[0] > 0)[0])

        # print(f"Frame: {self.frame}, Count: {packet_count}")
        self.frame += 1

    def trigger_spike(self, index):
        """
        Triggers a spike animation at the specified index.

        Args:
            index (int): The index of the spike to trigger.
        """
        self.size[index] += DISC_RADIUS_INC[0]
        self.color[index, -1] = 200

        # Add the zapping effect element
        zap_effect = pg.ScatterPlotItem()
        zap_effect.setData(
            pos=self.centroid_positions[index],
            size=self.size[index],
            symbol="o",
            pen=pg.mkPen(width=0, color=(230, 230, 230, 0)),
            brush=(230, 230, 230, 0),
        )
        self.plot_widget.addItem(zap_effect)
        QTimer.singleShot(30, lambda: self.plot_widget.removeItem(zap_effect))

    def shrink_circle(self):
        """
        Expands a circle in the animation by gradually increasing its size.

        Args:
            circle (tuple): A tuple containing the circle item and its current radius.
        """
        self.size *= 0.98
        self.color[:, -1] = 0.98 * self.color[:, -1]
        self.indicators.setData(
            size=self.size, pos=self.centroid_positions, brush=self.color
        )


# visualizer = RasterWithTraceVisualizer(0)
# visualizer = SpikeBubble3DVisualizer()
visualizer = SpikeDisc2DVisualizer()


class SpikePacer(QtCore.QObject):
    trigger = pyqtSignal()

    def __init__(self, fcn):
        super().__init__()
        self.trigger.connect(fcn)

    def spike_osc_handler(self, address, *args):
        global SPIKES, packet_count

        SPIKES[0][:50] = np.array(args[0])
        SPIKES[0][50:] = np.array(args[0])
        packet_count += 1
        self.trigger.emit()

    def max_control_osc_handler(self, address, *args):
        exec("global " + address[1:])
        exec(address[1:] + "[0] = args[0]")


spike_pacer = SpikePacer(visualizer.update)


async def init_main():
    dispatcher_spike = Dispatcher()
    dispatcher_spike.map("/SPIKES", spike_pacer.spike_osc_handler)
    dispatcher_max = Dispatcher()
    dispatcher_max.map("/DISC_RADIUS_INC", spike_pacer.max_control_osc_handler)

    server_spike = AsyncIOOSCUDPServer(
        (SERVER_IP, SPIKE_PORT), dispatcher_spike, asyncio.get_event_loop()
    )
    server_max = AsyncIOOSCUDPServer(
        (SERVER_IP, MAX_CONTROL_PORT), dispatcher_max, asyncio.get_event_loop()
    )
    transport, protocol = await server_spike.create_serve_endpoint()
    transport, protocol = await server_max.create_serve_endpoint()
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
    # visualizer2.animation()

import sys

import numpy as np
import asyncio
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import pyqtgraph.exporters

from pyqtgraph.Qt import QtCore, QtGui
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
LATENT = [np.zeros(8)]
DISC_RADIUS_INC = [10]
DECAY_FACTOR = [0.9]
SEQUENCE_TRIGGER = [0]


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
        # self.update_centroid_indicator()

        if self.frame % 1 == 0:
            self.update_centroid()
            if any(self.data > 0):
                self.trigger_spike(np.where(self.data > 0)[0])
        print(f"Frame: {self.frame}, Count: {packet_count}")
        self.frame += 1

    def update_centroid(self):
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
        self.plot_widget.setYRange(-12, 12)
        self.plot_widget.setXRange(-12, 12)
        self.plot_widget.setGeometry(0, 110, 640, 480)
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
        self.target_positions = np.zeros((num_neurons, 2))
        for i in range(10):
            self.target_positions[10 * (i) : 10 * (i + 1), :] = [
                9 * np.cos(2 * np.pi * i / 10),
                9 * np.sin(2 * np.pi * i / 10),
            ]

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
        if self.frame % 10 == 0:
            self.shrink_circle()
            self.estimate_velocity()
            # self.move_centroids()
            if any(SPIKES[0] > 0):
                self.trigger_spike(np.where(SPIKES[0] > 0)[0])
        if self.frame > 5 * 1e3 * 10:
            SEQUENCE_TRIGGER[0] = 1

        if self.frame > 30 * 1e3 * 10:
            self.app.quit()
        print(f"Frame: {self.frame}, Count: {packet_count}")
        self.frame += 1
        # Save the frame for video generation

    def trigger_spike(self, index):
        """
        Triggers a spike animation at the specified index.

        Args:
            index (int): The index of the spike to trigger.
        """
        self.size[index] += DISC_RADIUS_INC[0]
        self.color[index, -1] = 200
        if SEQUENCE_TRIGGER[0] == 1:
            self.velocity = self.centroid_positions - self.target_positions
            self.centroid_positions[index] -= 0.3 * self.velocity[index]

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

    def estimate_velocity(self):
        self.velocity = np.random.uniform(-1, 1, (num_neurons, 2))
        self.centroid_positions += 0.03 * self.velocity

    def save_frame(self):
        """
        Saves the current frame of the animation to a file.
        """
        exporter = pg.exporters.ImageExporter(self.plot_widget.scene())
        exporter.parameters()["width"] = 1920
        exporter.export(
            "/Users/hyungju/Desktop/hyungju/Project/sonification/results/animation/spike/frame"
            + str(self.frame)
            + ".png"
        )


class SpikeRaster2DVisualizer:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setYRange(0, 100)
        self.plot_widget.setXRange(0, 1000)
        self.plot_widget.setGeometry(0, 110, 640, 480)
        # remove axis
        self.plot_widget.hideAxis("bottom")
        self.plot_widget.hideAxis("left")
        self.plot_widget.show()
        self.plot_widget.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)

        self.L = 1000
        self.buffer = 1000
        self.firing_rates = np.zeros(
            (self.L + self.buffer, num_neurons)
        )  # Initialize firing rates for each neuron
        self.decay_factor = DECAY_FACTOR[0]
        self.num_neurons = num_neurons

        self.img = pg.ImageItem()
        self.img.setLevels((0, 20))
        scale_factor = [640 / self.L, 480 / self.num_neurons]
        # tr = QtGui.QTransform()  # prepare ImageItem transformation:
        # tr.scale(scale_factor[0], scale_factor[1])
        # tr.translate(
        #     -self.L / 2, -self.num_neurons / 2
        # )  # move 3x3 image to locate center at axis origin
        # self.img.setTransform(tr)  # assign transform

        self.plot_widget.addItem(self.img)

        # Create a custom lookup table (LUT) with green-neon color
        self.lut = []
        for i in range(256):
            self.lut.append([51, 255, 51, i])  # Add alpha channel as the last value
        self.img.setLookupTable(self.lut)
        self.frame = 0

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
            self.img.setImage(
                self.firing_rates[np.fmax(0, self.frame - self.L) : self.frame, :],
                autoLevels=False,
            )
            print(f"Frame: {self.count}, Count: {packet_count}")
        self.frame += 1


class LatentOrbitVisualizer:
    def __init__(self, x_index, y_index):

        # 'distance': 10.0,         ## distance of camera from center
        # 'fov':  60,               ## horizontal field of view in degrees
        # 'elevation': 30,          ## camera's angle of elevation in degrees
        # 'azimuth': 45,            ## camera's azimuthal angle in degrees
        # Create a PyQtGraph window
        self.app = QApplication([])
        self.plot_widget = gl.GLViewWidget()
        self.plot_widget.setGeometry(0, 110, 640, 480)
        self.plot_widget.opts["distance"] = 30
        self.plot_widget.opts["fov"] = 90
        self.plot_widget.opts["elevation"] = -5
        self.plot_widget.opts["azimuth"] = 45
        self.plot_widget.show()

        gridSize = 25
        scale = 2
        gx = gl.GLGridItem(QtGui.QVector3D(gridSize, gridSize, 1))
        gx.rotate(90, 0, 1, 0)
        gx.scale(scale, scale, scale)
        gx.translate(-gridSize, 0, gridSize / 2)
        self.plot_widget.addItem(gx)
        gy = gl.GLGridItem(QtGui.QVector3D(gridSize, gridSize, 1))
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, -gridSize, gridSize / 2)
        gy.scale(scale, scale, scale)
        self.plot_widget.addItem(gy)
        gz = gl.GLGridItem(QtGui.QVector3D(gridSize, gridSize, 1))
        gz.translate(0, 0, -gridSize + gridSize / 2)
        gz.scale(scale, scale, scale)
        self.plot_widget.addItem(gz)

        self.L = 1000
        self.buffer = 1000
        self.latent = np.zeros(
            (self.L + self.buffer, 8)
        )  # Initialize firing rates for each neuron
        self.decay_factor = DECAY_FACTOR[0]
        self.x = x_index
        self.y = y_index
        self.traces = dict()
        self.heads = dict()

        color = np.repeat(
            np.array([51, 255, 51, 255])[np.newaxis, :] / 255, self.L, axis=0
        )
        for i in range(1, self.L):
            color[i][-1] = color[i - 1][-1] * self.decay_factor
        self.L = np.where(np.array([x[-1] for x in color]) < 0.1)[0][0]
        color = color[: self.L]
        color = color[::-1]
        self.data = np.zeros((8, self.L + self.buffer))
        # self.L = 1000

        # self.glColor = [pg.glColor(x) for x in color]

        self.z = [x for x in range(8) if x not in [self.x, self.y]]

        for i in range(6):
            self.traces[i] = gl.GLLinePlotItem(
                pos=np.zeros((self.L, 3)),
                color=color,
                width=5,
                antialias=True,
            )
            self.heads[i] = gl.GLScatterPlotItem(
                pos=np.zeros((1, 3)),
                size=0,
            )
            self.plot_widget.addItem(self.traces[i])
            self.plot_widget.addItem(self.heads[i])
        self.frame = 0

    def animation(self):
        self.prev_count = 0
        self.count = 0
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QApplication.instance().exec_()

    def update(self):
        self.count += 1
        if self.frame >= self.L + self.buffer:
            self.data = np.roll(self.data, -self.buffer, axis=1)
            self.frame = self.L
        if self.frame > 0:
            self.data[:, self.frame] = LATENT[0]

        self.data = np.roll(self.data, -1, axis=1)
        self.data[:, -1] = LATENT[0]

        if self.frame % 10 == 0:
            for i in range(6):
                pts = np.vstack(
                    [
                        self.data[i, :] * 5,
                        self.data[i + 1, :] * 5,
                        self.data[i + 2, :] * 5,
                    ]
                ).transpose()
                self.heads[i].setData(pos=pts[-1, :], size=10)
                self.traces[i].setData(
                    pos=pts,
                )
            # for i in range(6):
            #     pts = np.vstack(
            #         [
            #             self.data[self.x] * 5,
            #             self.data[self.y] * 5,
            #             self.data[self.z[i], :] * 3,
            #         ]
            #     ).transpose()
            #     self.traces[i].setData(
            #         pos=pts,
            #     )
            # if self.frame >= self.L + self.buffer:
            #     self.data = np.roll(self.data, -self.buffer, axis=0)
            #     self.frame = self.L
            print(f"Frame: {self.count}, Count: {packet_count}")
        self.frame += 1


# visualizer = RasterWithTrace3DVisualizer(0)
# visualizer = SpikeBubble3DVisualizer()
# visualizer = SpikeDisc2DVisualizer()
# visualizer = SpikeRaster2DVisualizer()
visualizer = LatentOrbitVisualizer(0, 1)


class SpikePacer(QtCore.QObject):
    # spike_trigger = pyqtSignal()
    latent_trigger = pyqtSignal()

    def __init__(self, fcn):
        super().__init__()
        # self.spike_trigger.connect(fcn)
        self.latent_trigger.connect(fcn)

    def spike_osc_handler(self, address, *args):
        global SPIKES, packet_count

        SPIKES[0][:50] = np.array(args[0])
        SPIKES[0][50:] = np.array(args[1])
        packet_count += 1
        # self.spike_trigger.emit()

    def latent_osc_handler(self, address, *args):
        global LATENT, packet_count

        LATENT[0] = np.array(args)
        packet_count += 1
        self.latent_trigger.emit()
        # print(f"Received update for LATENT : {args}")
        # print(
        #     f"Received update for LATENT : {args}, total packet count: {packet_count}"
        # )

    def max_control_osc_handler(self, address, *args):
        exec("global " + address[1:])
        exec(address[1:] + "[0] = args[0]")


spike_pacer = SpikePacer(visualizer.update)


async def init_main():
    dispatcher_python = Dispatcher()
    dispatcher_python.map("/SPIKES", spike_pacer.spike_osc_handler)
    dispatcher_python.map("/trajectory", spike_pacer.latent_osc_handler)
    dispatcher_max = Dispatcher()
    dispatcher_max.map("/DISC_RADIUS_INC", spike_pacer.max_control_osc_handler)
    dispatcher_max.map("/SEQUENCE_TRIGGER", spike_pacer.max_control_osc_handler)

    server_spike = AsyncIOOSCUDPServer(
        (SERVER_IP, SPIKE_PORT), dispatcher_python, asyncio.get_event_loop()
    )
    server_latent = AsyncIOOSCUDPServer(
        (SERVER_IP, LATENT_PORT), dispatcher_python, asyncio.get_event_loop()
    )
    server_max = AsyncIOOSCUDPServer(
        (SERVER_IP, MAX_CONTROL_PORT), dispatcher_max, asyncio.get_event_loop()
    )
    transport, protocol = await server_spike.create_serve_endpoint()
    transport, protocol = await server_latent.create_serve_endpoint()
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

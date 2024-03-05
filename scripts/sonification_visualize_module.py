import asyncio
import signal
import sys
import threading
from tkinter import SE

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtCore import QTimer, QTimerEvent, pyqtSignal
from PyQt5.QtWidgets import QApplication, QDesktopWidget, QGraphicsView, QGridLayout
from pyqtgraph.Qt import QtCore, QtGui
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer
from matplotlib.colors import LinearSegmentedColormap
from sonification_communication_module import *

cdict = {
    "red": [[0.0, 0.2, 0.2], [1.0, 1.0, 1.0]],
    "green": [[0.0, 1.0, 1.0], [1.0, 0.7, 0.7]],
    "blue": [[0.0, 0.2, 0.2], [1.0, 0.0, 0.0]],
}
newcmp = LinearSegmentedColormap("testCmap", segmentdata=cdict, N=256)
RGBA = np.round(newcmp(np.linspace(0, 1, 256)) * 255).astype(int)

packet_count = 0
num_neurons = 100
SPIKES = [np.zeros(num_neurons)]
LATENT = [np.zeros(8)]

ASPECT_RATIO = 0.7
GRID_SIZE_WIDTH = 100
GRID_SIZE_HEIGHT = 50

DISC_RADIUS_INC = [5]
DISC_DECAY_FACTOR = [0.90]
LATENT_DECAY_FACTOR = [0.99]
RASTER_DECAY_FACTOR = [0.1]

WALL_RASTER_LEFT = [0]
WALL_RASTER_RIGHT = [0]
WALL_RASTER_TOP = [0]
WALL_RASTER_BOTTOM = [1]
WALL_SPIKE = [1]
WALL_TRUE_LATENT = [0]
WALL_INFERRED_LATENT = [0]

CEILING_RASTER = [1]
CEILING_TRUE_LATENT = [0]
CEILING_INFERRED_LATENT = [0]
CEILING_SPIKE = [0]
SPIKE_ORGANIZATION = [1]

COLOR_INDEX = [0]
TRUE_LATENT_COLOR = [51, 255, 51, 255]
INFERRED_LATENT_COLOR = [255, 176, 0, 255]

SCALE_FACTOR = GRID_SIZE_HEIGHT / 4
pg.setConfigOptions(useOpenGL=True)


class SpikeDisc2DVisualizer:

    def __init__(self, visible=False, widget=None):
        if widget is None:
            self.app = QApplication(sys.argv)
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setYRange(-12, 12)
            self.plot_widget.setXRange(-12, 12)
            self.plot_widget.setGeometry(0, 110, 640, 480)
            # remove axis
            self.plot_widget.hideAxis("bottom")
            self.plot_widget.hideAxis("left")
            self.plot_widget.show()
        else:
            self.plot_widget = widget

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
        self.prev_count = 0
        self.count = 0

    def animation(self):
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

        # print(f"Frame: {self.frame}, Count: {packet_count}")
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

    def __init__(self, visible=False, widget=None):
        if widget is None:
            self.app = QApplication(sys.argv)
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setXRange(0, num_neurons)
            self.plot_widget.setYRange(0, 1000)
            self.plot_widget.setGeometry(0, 110, 640, 480)
            # remove axis
            self.plot_widget.hideAxis("bottom")
            self.plot_widget.hideAxis("left")
            self.plot_widget.show()
        else:
            self.plot_widget = widget

        self.L = 1000
        self.buffer = 1000
        self.firing_rates = np.zeros(
            (self.L + self.buffer, num_neurons)
        )  # Initialize firing rates for each neuron
        self.decay_factor = DISC_DECAY_FACTOR[0]
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
            # print(f"Frame: {self.count}, Count: {packet_count}")
        self.frame += 1


class RasterWithTrace3DVisualizer:
    def __init__(
        self, orientation, visible=True, max_level=5, separate=False, widget=None
    ):
        # Create a PyQtGraph window
        if widget is None:
            self.app = QApplication([])
            self.plot_widget = gl.GLViewWidget()
            self.plot_widget.setGeometry(0, 0, 1900, 1200)
            self.plot_widget.opts["distance"] = 600
            self.plot_widget.show()
        else:
            self.plot_widget = widget
            self.plot_widget.show()

        self.num_neurons = 25 if separate else 100
        self.L = 1000
        self.buffer = 1000
        self.firing_rates = np.zeros((self.L + self.buffer, num_neurons))
        self.decay_factor = RASTER_DECAY_FACTOR[0]  # Exponential decay factor
        self.max_level = max_level
        self.visible = visible

        raster_texture = pg.makeRGBA(self.firing_rates, levels=(0, max_level))[0]
        self.img = gl.GLImageItem(raster_texture)

        if orientation == "left":
            self.slicer = slice(50, 75) if separate else slice(0, 100)
            scale_factor = [
                -GRID_SIZE_WIDTH / self.L,
                -GRID_SIZE_HEIGHT / self.num_neurons,
                1,
            ]
            self.img.scale(*scale_factor)
            self.img.rotate(90, 1, 0, 0)
            self.img.translate(
                +GRID_SIZE_WIDTH / 2 + GRID_SIZE_HEIGHT / 2,
                -GRID_SIZE_WIDTH / 2,
                GRID_SIZE_HEIGHT / 2 + 0,
            )
        elif orientation == "right":
            self.slicer = slice(0, 25) if separate else slice(0, 100)
            scale_factor = [
                -GRID_SIZE_WIDTH / self.L,
                -GRID_SIZE_HEIGHT / self.num_neurons,
                1,
            ]
            self.img.scale(*scale_factor)
            self.img.rotate(-90, 1, 0, 0)
            self.img.translate(
                +GRID_SIZE_WIDTH / 2 + GRID_SIZE_HEIGHT / 2,
                +GRID_SIZE_WIDTH / 2,
                -GRID_SIZE_HEIGHT / 2 + 0,
            )
        elif orientation == "top":
            self.slicer = slice(25, 50) if separate else slice(0, 100)
            scale_factor = [
                -GRID_SIZE_WIDTH / self.L,
                -GRID_SIZE_WIDTH / self.num_neurons,
                1,
            ]
            self.img.scale(*scale_factor)
            self.img.translate(
                +GRID_SIZE_WIDTH / 2 + GRID_SIZE_HEIGHT / 2,
                +GRID_SIZE_WIDTH / 2,
                +GRID_SIZE_HEIGHT / 2 + 0,
            )
        elif orientation == "bot":
            self.slicer = slice(75, 100) if separate else slice(0, 100)
            scale_factor = [
                -GRID_SIZE_WIDTH / self.L,
                GRID_SIZE_WIDTH / self.num_neurons,
                1,
            ]
            self.img.scale(*scale_factor)
            self.img.translate(
                +GRID_SIZE_WIDTH / 2 + GRID_SIZE_HEIGHT / 2,
                -GRID_SIZE_WIDTH / 2,
                -GRID_SIZE_HEIGHT / 2 + 0,
            )
        else:
            self.slicer = slice(0, 100) if separate else slice(0, 100)
            scale_factor = [
                GRID_SIZE_HEIGHT / self.L,
                GRID_SIZE_WIDTH / self.num_neurons,
                1,
            ]
            self.img.scale(*scale_factor)
            self.img.rotate(90, 0, 1, 0)
            self.img.translate(
                -GRID_SIZE_WIDTH / 2 + GRID_SIZE_HEIGHT / 2,
                -GRID_SIZE_WIDTH / 2,
                GRID_SIZE_HEIGHT / 2,
            )

        self.img.setVisible(self.visible)
        self.plot_widget.addItem(self.img)

        self.frame = 0
        self.count = 0

        # Create a custom lookup table (LUT) with green-neon color
        self.lut = []
        for i in range(256):
            self.lut.append([51, 255, 51, i])  # Add alpha channel as the last value

    def animation(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QApplication.instance().exec_()

    def update_lut(self, color):
        for i in range(256):
            self.lut[i] = [color[0], color[1], color[2], i]

    def update(self):
        if self.frame >= self.L + self.buffer:
            self.firing_rates = np.roll(self.firing_rates, -self.buffer, axis=0)
            self.frame = self.L

        if self.frame > 0:
            self.firing_rates[self.frame, :] = (
                self.firing_rates[self.frame - 1, :] * self.decay_factor
            )

        for i, firing_event in enumerate(SPIKES[0][self.slicer]):
            if firing_event > 0:  # If there's firing
                self.firing_rates[self.frame, i] += 5  # Instantly increase firing rate

        if self.frame % 10 == 0 and self.visible:
            if self.count >= 1000:
                raster_texture = pg.makeRGBA(
                    self.firing_rates[np.fmax(0, self.frame - self.L) : self.frame, :],
                    levels=(0, self.max_level),
                    lut=self.lut,
                )[0]
            else:
                rolled_rate = self.firing_rates[: self.L, :]
                rolled_rate = np.roll(rolled_rate, 1000 - self.count, axis=0)
                raster_texture = pg.makeRGBA(
                    rolled_rate,
                    levels=(0, self.max_level),
                    lut=self.lut,
                )[0]
            self.img.setData(raster_texture)

        self.count += 1
        self.frame += 1


class SpikeBall3DVisualizer:
    def __init__(self, target_location=None, visible=False, widget=None):
        self.num_neurons = num_neurons
        # Create a PyQtGraph window
        if widget is None:
            self.app = QApplication([])
            self.plot_widget = gl.GLViewWidget()
            self.plot_widget.setGeometry(0, 110, 640, 480)
            self.plot_widget.opts["distance"] = 30
            self.plot_widget.opts["fov"] = 90
            self.plot_widget.opts["elevation"] = -5
            self.plot_widget.opts["azimuth"] = 45
            self.plot_widget.show()

        else:
            self.plot_widget = widget

        self.data = np.zeros(num_neurons)
        self.centroid_positions = [
            [
                np.random.uniform(-10, 10),
                np.random.uniform(-10, 10) * ASPECT_RATIO,
                np.random.uniform(-10, 10),
            ]
            for _ in range(num_neurons)
        ]
        if target_location is None:
            self.target_positions = np.zeros((num_neurons, 3))
            for i in range(10):
                self.target_positions[10 * (i) : 10 * (i + 1), :] = [
                    9 * np.cos(2 * np.pi * i / 10),
                    9 * np.sin(2 * np.pi * i / 10),
                    3 * np.random.uniform(-10, 10),
                ]
        else:
            self.target_positions = target_location
            # self.target_positions[:, 1, :] *= ASPECT_RATIO
        self.velocity = np.random.uniform(-1, 1, (num_neurons, 2))

        self.draw_centroids()
        # Add function that will be called when the timer times out
        self.frame = 0
        self.prev_count = 0
        self.count = 0
        self.visible = visible

    def animation(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QApplication.instance().exec_()

    def draw_centroids(self):
        indicators = gl.GLScatterPlotItem()
        indicators.setData(
            pos=self.centroid_positions,
            size=5,
            color=(0.2, 1, 0.2, 0),
        )
        self.plot_widget.addItem(indicators)
        self.indicators = indicators
        self.size = np.ones(num_neurons) * 5
        self.color = np.repeat(
            np.array([0.2, 1, 0.2, 0.4])[np.newaxis, :], num_neurons, axis=0
        )
        self.color[raster_sort_idx % 50 > 25] = [0.7, 0.7, 0.7, 0.4]

    def update_color(self, color):
        self.color[:, :3] = color[:3] / 255
        print(self.color[0, :3])
        self.color[raster_sort_idx % 50 > 25] = [0.7, 0.7, 0.7, 0.4]

    def update(self):
        if self.frame % 10 == 0:
            self.shrink_circle()
            self.estimate_velocity()

            if any(SPIKES[0] > 0):
                self.trigger_spike(np.where(SPIKES[0] > 0)[0])
        # print(f"Frame: {self.frame}, Count: {packet_count}")
        self.frame += 1
        self.count += 1

    def trigger_spike(self, index):
        """
        Triggers a spike animation at the specified index.

        Args:
            index (int): The index of the spike to trigger.
        """
        if WALL_SPIKE[0] == 1:
            self.size[index] += DISC_RADIUS_INC[0] * 1.0
        else:
            self.size[index] = 0
        self.color[index, -1] = 0.9
        if SPIKE_ORGANIZATION[0]:
            self.velocity = self.centroid_positions - self.target_positions
            self.centroid_positions[index] -= 0.05 * self.velocity[index]

        pos = [self.centroid_positions[idx] for idx in index]

    def shrink_circle(self):
        """
        Expands a circle in the animation by gradually increasing its size.

        Args:
            circle (tuple): A tuple containing the circle item and its current radius.
        """
        self.size *= 0.98
        self.color[:, -1] = 0.98 * self.color[:, -1]
        if self.visible:
            self.indicators.setData(
                size=self.size, pos=self.centroid_positions, color=self.color
            )

    def estimate_velocity(self):
        self.velocity = np.random.uniform(-1, 1, (num_neurons, 3))
        self.velocity[0] = 0
        if not SPIKE_ORGANIZATION[0]:
            self.velocity *= 3
        self.centroid_positions += 0.05 * self.velocity
        self.centroid_positions -= 0.0001 * self.centroid_positions


class LatentCycleVisualizer:
    def __init__(self, x_index, y_index, widget=None):
        # Create a PyQtGraph window
        if widget is None:
            self.app = QApplication([])
            self.plot_widget = gl.GLViewWidget()
        else:
            self.plot_widget = widget
        self.plot_widget.setGeometry(0, 110, 640, 480)
        self.plot_widget.opts["distance"] = 30
        self.plot_widget.opts["fov"] = 90
        self.plot_widget.opts["elevation"] = -5
        self.plot_widget.opts["azimuth"] = 45
        self.plot_widget.show()

        self.L = 1000
        self.buffer = 1000
        self.latent = np.zeros(
            (self.L + self.buffer, 8)
        )  # Initialize firing rates for each neuron
        self.decay_factor = DISC_DECAY_FACTOR[0]
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

        self.z = [x for x in range(8) if x not in [self.x, self.y]]
        for i in range(6):
            self.traces[i] = gl.GLLinePlotItem(
                pos=np.zeros((self.L, 3)),
                color=color,
                width=5,
                antialias=True,
            )
            self.plot_widget.addItem(self.traces[i])
        self.frame = 0
        self.prev_count = 0
        self.count = 0

    def animation(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QApplication.instance().exec_()

    def update(self):
        self.count += 1
        if self.frame >= self.L + self.buffer:
            self.data = np.roll(self.data, -self.buffer, axis=1)
            self.frame = self.L
        if self.frame > 0:
            self.data[:, self.frame] = LATENT[0]
        if self.frame % 10 == 0 and self.frame > 0:
            slice_window = slice(np.fmax(0, self.frame - self.L), self.frame)
            for i in range(6):
                pts = np.vstack(
                    [
                        self.data[i, slice_window] * 5,
                        self.data[i + 1, slice_window] * 5,
                        self.data[i + 2, slice_window] * 5,
                    ]
                ).transpose()
                self.traces[i].setData(
                    pos=pts,
                )
            # print(f"Frame: {self.count}, Count: {packet_count}")
        self.frame += 1


class LatentOrbitCeilingVisualizer:
    def __init__(self, x_index, y_index, color, visible=False, widget=None):
        # Create a PyQtGraph window
        if widget is None:
            self.app = QApplication([])
            self.plot_widget = gl.GLViewWidget()
            self.plot_widget.setGeometry(0, 110, 640, 480)
            self.plot_widget.opts["distance"] = 30
            self.plot_widget.opts["fov"] = 90
            self.plot_widget.opts["elevation"] = -5
            self.plot_widget.opts["azimuth"] = 0
            self.plot_widget.show()
        else:
            self.plot_widget = widget

        self.L = 1000
        self.buffer = 1000
        self.latent = np.zeros(
            (self.L + self.buffer, 8)
        )  # Initialize firing rates for each neuron
        self.decay_factor = LATENT_DECAY_FACTOR[0]
        self.x = x_index
        self.y = y_index
        self.traces = dict()
        self.heads = dict()
        self.color = np.repeat(np.array(color)[np.newaxis, :] / 255, self.L, axis=0)

        for i in range(1, self.L):
            self.color[i][-1] = self.color[i - 1][-1] * self.decay_factor
        self.L = np.where(np.array([x[-1] for x in self.color]) < 0.1)[0][0]
        self.color = self.color[: self.L]
        self.color = self.color[::-1]
        self.data = np.zeros((8, self.L + self.buffer))

        self.z = [x for x in range(8) if x not in [self.x]]
        for i in range(7):
            self.traces[i] = gl.GLLinePlotItem(
                pos=np.zeros((self.L, 3)),
                color=self.color,
                width=5,
                antialias=True,
            )
            self.traces[i].setVisible(visible)
            self.plot_widget.addItem(self.traces[i])
        self.frame = 0
        self.prev_count = 0
        self.count = 0
        self.bias = 0
        self.visible = visible

    def animation(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QApplication.instance().exec_()

    def update(self):
        self.count += 1
        if self.frame >= self.L + self.buffer:
            self.data = np.roll(self.data, -self.buffer, axis=1)
            self.frame = self.L
        if self.frame > 0:
            self.data[:, self.frame] = LATENT[0]
        if self.frame % 10 == 0 and self.frame > 0 and self.visible:
            slice_window = slice(np.fmax(0, self.frame - self.L), self.frame)
            for i in range(0, 6):
                pts = np.vstack(
                    [
                        np.ones_like(self.data[0, slice_window])
                        * (-GRID_SIZE_WIDTH / 2 + GRID_SIZE_HEIGHT / 2),
                        (self.data[i, slice_window] * SCALE_FACTOR + self.bias),
                        self.data[i + 1, slice_window] * SCALE_FACTOR,
                    ]
                ).transpose()
                self.traces[i].setData(
                    pos=pts,
                )
            # print(f"Frame: {self.count}, Count: {packet_count}")
        self.frame += 1


class LatentOrbitVisualizer:
    def __init__(self, x_index, y_index, color, visible=False, widget=None):
        # Create a PyQtGraph window
        if widget is None:
            self.app = QApplication([])
            self.plot_widget = gl.GLViewWidget()
            self.plot_widget.setGeometry(0, 110, 640, 480)
            self.plot_widget.opts["distance"] = 30
            self.plot_widget.opts["fov"] = 90
            self.plot_widget.opts["elevation"] = -5
            self.plot_widget.opts["azimuth"] = 0
            self.plot_widget.show()
        else:
            self.plot_widget = widget

        self.L = 1000
        self.buffer = 1000
        self.latent = np.zeros(
            (self.L + self.buffer, 8)
        )  # Initialize firing rates for each neuron
        self.decay_factor = LATENT_DECAY_FACTOR[0]
        self.x = x_index
        self.y = y_index
        self.traces = dict()
        self.heads = dict()
        self.color = np.repeat(np.array(color)[np.newaxis, :] / 255, self.L, axis=0)

        for i in range(1, self.L):
            self.color[i][-1] = self.color[i - 1][-1] * self.decay_factor
        self.L = np.where(np.array([x[-1] for x in self.color]) < 0.1)[0][0]
        self.color = self.color[: self.L]
        self.color = self.color[::-1]
        self.data = np.zeros((8, self.L + self.buffer))

        self.z = [x for x in range(8) if x not in [self.x]]
        for i in range(7):
            self.traces[i] = gl.GLLinePlotItem(
                pos=np.zeros((self.L, 3)),
                color=self.color,
                width=5,
                antialias=True,
            )
            self.traces[i].setVisible(visible)
            self.plot_widget.addItem(self.traces[i])
        self.frame = 0
        self.prev_count = 0
        self.count = 0
        self.bias = 0
        self.visible = visible

    def animation(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QApplication.instance().exec_()

    def update(self):
        self.count += 1
        if self.frame >= self.L + self.buffer:
            self.data = np.roll(self.data, -self.buffer, axis=1)
            self.frame = self.L
        if self.frame > 0:
            self.data[:, self.frame] = LATENT[0]
        if self.frame % 10 == 0 and self.frame > 0 and self.visible:
            slice_window = slice(np.fmax(0, self.frame - self.L), self.frame)
            for i in range(0, 6):
                pts = np.vstack(
                    [
                        np.ones_like(self.data[0, slice_window])
                        * (-GRID_SIZE_WIDTH / 2 + GRID_SIZE_HEIGHT / 2),
                        (self.data[i, slice_window] * SCALE_FACTOR + self.bias)
                        * ASPECT_RATIO,
                        self.data[i + 1, slice_window] * SCALE_FACTOR,
                    ]
                ).transpose()
                self.traces[i].setData(
                    pos=pts,
                )
            # print(f"Frame: {self.count}, Count: {packet_count}")
        self.frame += 1


# -------------------------------------------------------------------------------
loading_matrix_fast_name = "./data/loading_matrix_fast.npz"
loading_matrix_slow_name = "./data/loading_matrix_slow.npz"

param = np.load(loading_matrix_fast_name, allow_pickle=True)
C_fast, b_fast = np.hstack(param["C"]), param["b"]
param = np.load(loading_matrix_slow_name, allow_pickle=True)
C_slow, b_slow = np.hstack(param["C"]), param["b"]

C = np.hstack([C_fast, C_slow])
b = np.vstack([b_fast, b_slow]).flatten()
theta = np.arctan2(C[1, :], C[0, :])

raster_sort_idx = np.argsort(theta)
theta = theta[raster_sort_idx]
target_location = np.vstack(
    [
        np.ones_like(theta) * (-GRID_SIZE_WIDTH / 2 + GRID_SIZE_HEIGHT / 2),
        np.cos(theta) * GRID_SIZE_HEIGHT / 2,
        np.sin(theta) * GRID_SIZE_HEIGHT / 2,
    ]
).T
target_location[:, 1:] *= 0.7
target_location[:, 1] *= ASPECT_RATIO
target_location[raster_sort_idx < 50, 1] += GRID_SIZE_HEIGHT / 2
target_location[raster_sort_idx >= 50, 1] -= GRID_SIZE_HEIGHT / 2

app = QApplication([])
# -----------------------------Visualization on the wall-----------------------------
parent_widget = gl.GLViewWidget()
parent_widget.setGeometry(0, 0, 1920, 1200)
layout = QGridLayout()
parent_widget.setLayout(layout)
monitor = QDesktopWidget().screenGeometry(1)
parent_widget.move(monitor.left(), monitor.top())
# parent_widget.showFullScreen()

wall_plot_widget = gl.GLViewWidget()
wall_plot_widget.setGeometry(0, 0, 1920, 1200)
wall_plot_widget.opts["center"] = QtGui.QVector3D(-30, 0, 0)
wall_plot_widget.opts["distance"] = 75
wall_plot_widget.opts["fov"] = 90
wall_plot_widget.opts["elevation"] = 0
wall_plot_widget.opts["azimuth"] = 0

g_center = gl.GLGridItem(QtGui.QVector3D(GRID_SIZE_HEIGHT, GRID_SIZE_WIDTH, 1))
g_center.rotate(90, 0, 1, 0)
g_center.translate(-GRID_SIZE_WIDTH / 2 + GRID_SIZE_HEIGHT / 2, 0, 0)
wall_plot_widget.addItem(g_center)
g_left = gl.GLGridItem(QtGui.QVector3D(GRID_SIZE_WIDTH, GRID_SIZE_HEIGHT, 1))
g_left.rotate(90, 1, 0, 0)
g_left.translate(GRID_SIZE_HEIGHT / 2, -GRID_SIZE_WIDTH / 2, 0)
wall_plot_widget.addItem(g_left)
g_floor = gl.GLGridItem(QtGui.QVector3D(GRID_SIZE_WIDTH, GRID_SIZE_WIDTH, 1))
g_floor.translate(GRID_SIZE_HEIGHT / 2, 0, -GRID_SIZE_HEIGHT / 2 + 0)
wall_plot_widget.addItem(g_floor)
g_right = gl.GLGridItem(QtGui.QVector3D(GRID_SIZE_WIDTH, GRID_SIZE_HEIGHT, 1))
g_right.rotate(-90, 1, 0, 0)
g_right.translate(GRID_SIZE_HEIGHT / 2, GRID_SIZE_WIDTH / 2, 0)
wall_plot_widget.addItem(g_right)

layout.addWidget(wall_plot_widget, 1, 0)
parent_widget.show()

vis_wall_raster_left = RasterWithTrace3DVisualizer(
    orientation="left", visible=WALL_RASTER_LEFT[0], widget=wall_plot_widget
)
vis_wall_raster_right = RasterWithTrace3DVisualizer(
    orientation="right", visible=WALL_RASTER_RIGHT[0], widget=wall_plot_widget
)
vis_wall_raster_top = RasterWithTrace3DVisualizer(
    orientation="top", visible=WALL_RASTER_TOP[0], widget=wall_plot_widget
)
vis_wall_raster_bottom = RasterWithTrace3DVisualizer(
    orientation="bot", visible=WALL_RASTER_BOTTOM[0], widget=wall_plot_widget
)
vis_wall_spike = SpikeBall3DVisualizer(
    target_location=target_location, visible=WALL_SPIKE[0], widget=wall_plot_widget
)
vis_wall_true_latent = LatentOrbitVisualizer(
    0, 1, color=TRUE_LATENT_COLOR, visible=WALL_TRUE_LATENT[0], widget=wall_plot_widget
)
vis_wall_inferred_latent = LatentOrbitVisualizer(
    1,
    1,
    color=INFERRED_LATENT_COLOR,
    visible=WALL_INFERRED_LATENT[0],
    widget=wall_plot_widget,
)

# ---------------------------Visualization on the Ceiling-----------------------------
ceiling_plot_widget = gl.GLViewWidget()
ceiling_plot_widget.setGeometry(0, 0, 1920, 1200)
ceiling_plot_widget.opts["center"] = QtGui.QVector3D(-30, 0, 0)
ceiling_plot_widget.opts["distance"] = 55
ceiling_plot_widget.opts["fov"] = 90
ceiling_plot_widget.opts["elevation"] = 0
ceiling_plot_widget.opts["azimuth"] = 0

monitor = QDesktopWidget().screenGeometry(0)
ceiling_plot_widget.move(monitor.left(), monitor.top())
# ceiling_plot_widget.showFullScreen()
ceiling_plot_widget.show()

vis_ceiling_true_latent = LatentOrbitCeilingVisualizer(
    0,
    1,
    color=TRUE_LATENT_COLOR,
    visible=CEILING_TRUE_LATENT[0],
    widget=ceiling_plot_widget,
)
vis_ceiling_inferred_latent = LatentOrbitCeilingVisualizer(
    0,
    1,
    color=TRUE_LATENT_COLOR,
    visible=CEILING_INFERRED_LATENT[0],
    widget=ceiling_plot_widget,
)
vis_ceiling_raster = RasterWithTrace3DVisualizer(
    orientation="center", visible=CEILING_RASTER[0], widget=ceiling_plot_widget
)
# ---------------------------OSC mapping-----------------------------


class SpikePacer(QtCore.QObject):
    spike_trigger = pyqtSignal()
    latent_trigger = pyqtSignal()

    def __init__(self, spike_fcn, latent_fcn):
        super().__init__()
        for fcn in spike_fcn:
            self.spike_trigger.connect(fcn)
        for fcn in latent_fcn:
            self.latent_trigger.connect(fcn)

    def spike_osc_handler(self, address, *args):
        global SPIKES, packet_count
        SPIKES[0] = np.array(args)
        SPIKES[0] = SPIKES[0][raster_sort_idx]
        packet_count += 1
        self.spike_trigger.emit()

    def latent_osc_handler(self, address, *args):
        global LATENT, packet_count

        LATENT[0] = np.array(args)
        # packet_count += 1
        self.latent_trigger.emit()

    def max_switch_wall_raster_L(self, address, *args):
        vis_wall_raster_left.visible = args[0]
        vis_wall_raster_left.img.setVisible(args[0])

    def max_switch_wall_raster_R(self, address, *args):
        vis_wall_raster_right.visible = args[0]
        vis_wall_raster_right.img.setVisible(args[0])

    def max_switch_wall_raster_T(self, address, *args):
        vis_wall_raster_top.visible = args[0]
        vis_wall_raster_top.img.setVisible(args[0])

    def max_switch_wall_raster_B(self, address, *args):
        vis_wall_raster_bottom.visible = args[0]
        vis_wall_raster_bottom.img.setVisible(args[0])

    def max_switch_wall_spike(self, address, *args):
        vis_wall_spike.visible = args[0]
        vis_wall_spike.indicators.setVisible(args[0])

    def max_switch_wall_true_latent(self, address, *args):
        vis_wall_true_latent.visible = args[0]
        for trace in vis_wall_true_latent.traces.items():
            trace[1].setVisible(args[0])
        if args[0]:
            vis_wall_inferred_latent.bias = GRID_SIZE_HEIGHT / 2
        else:
            vis_wall_inferred_latent.bias = 0

    def max_switch_ceiling_true_latent(self, address, *args):
        vis_ceiling_true_latent.visible = args[0]
        for trace in vis_ceiling_true_latent.traces.items():
            trace[1].setVisible(args[0])
        if args[0]:
            vis_ceiling_inferred_latent.bias = GRID_SIZE_HEIGHT / 2
        else:
            vis_ceiling_inferred_latent.bias = 0

    def max_switch_wall_inferred_latent(self, address, *args):
        vis_wall_inferred_latent.visible = args[0]
        for trace in vis_wall_inferred_latent.traces.items():
            trace[1].setVisible(args[0])
        if args[0]:
            vis_wall_true_latent.bias = -GRID_SIZE_HEIGHT / 2
        else:
            vis_wall_true_latent.bias = 0

    def max_switch_ceiling_inferred_latent(self, address, *args):
        vis_ceiling_inferred_latent.visible = args[0]
        for trace in vis_ceiling_inferred_latent.traces.items():
            trace[1].setVisible(args[0])
        if args[0]:
            vis_ceiling_true_latent.bias = -GRID_SIZE_HEIGHT / 2
        else:
            vis_ceiling_true_latent.bias = 0

    def max_switch_ceiling_raster(self, address, *args):
        vis_ceiling_raster.visible = args[0]
        vis_ceiling_raster.img.setVisible(args[0])

    def max_control_osc_handler(self, address, *args):
        exec("global " + address[1:])
        exec(address[1:] + "[0] = args[0]")

    def max_control_color(self, address, *args):
        COLOR_INDEX[0] = args[0]
        color = RGBA[int(COLOR_INDEX[0])]
        vis_wall_raster_left.update_lut(color)
        vis_wall_raster_right.update_lut(color)
        vis_wall_raster_top.update_lut(color)
        vis_wall_raster_bottom.update_lut(color)
        vis_ceiling_raster.update_lut(color)
        vis_wall_spike.update_color(color)
        # vis_wall_true_latent.update_lut(color)
        # vis_ceiling_true_latent.update_lut(color)


spike_pacer = SpikePacer(
    spike_fcn=[
        vis_wall_raster_left.update,
        vis_wall_raster_bottom.update,
        vis_wall_raster_right.update,
        vis_wall_raster_top.update,
        vis_wall_spike.update,
        vis_ceiling_raster.update,
    ],
    latent_fcn=[
        vis_wall_true_latent.update,
        vis_wall_inferred_latent.update,
        vis_ceiling_true_latent.update,
        vis_ceiling_inferred_latent.update,
    ],
)


async def init_main():
    dispatcher_python = Dispatcher()
    dispatcher_python.map("/SPIKES", spike_pacer.spike_osc_handler)
    dispatcher_python.map("/TRAJECTORY", spike_pacer.latent_osc_handler)

    dispatcher_max = Dispatcher()
    dispatcher_max.map("/COLOR_INDEX", spike_pacer.max_control_color)
    dispatcher_max.map("/DISC_RADIUS_INC", spike_pacer.max_control_osc_handler)
    dispatcher_max.map("/SPIKE_ORGANIZATION", spike_pacer.max_control_osc_handler)
    dispatcher_max.map("/WALL_SPIKE", spike_pacer.max_switch_wall_spike)
    dispatcher_max.map("/WALL_RASTER_L", spike_pacer.max_switch_wall_raster_L)
    dispatcher_max.map("/WALL_RASTER_R", spike_pacer.max_switch_wall_raster_R)
    dispatcher_max.map("/WALL_RASTER_T", spike_pacer.max_switch_wall_raster_T)
    dispatcher_max.map("/WALL_RASTER_B", spike_pacer.max_switch_wall_raster_B)
    dispatcher_max.map("/WALL_TRUE_LATENT", spike_pacer.max_switch_wall_true_latent)
    dispatcher_max.map(
        "/WALL_INFERRED_LATENT", spike_pacer.max_switch_wall_inferred_latent
    )

    dispatcher_max.map("/CEILING_RASTER", spike_pacer.max_switch_ceiling_raster)
    dispatcher_max.map(
        "/CEILING_TRUE_LATENT", spike_pacer.max_switch_ceiling_true_latent
    )
    dispatcher_max.map(
        "/CEILING_INFERRED_LATENT", spike_pacer.max_switch_ceiling_inferred_latent
    )

    server_spike = AsyncIOOSCUDPServer(
        (LOCAL_SERVER, SPIKE_VISUALIZE_PORT),
        dispatcher_python,
        asyncio.get_event_loop(),
    )
    server_latent = AsyncIOOSCUDPServer(
        (LOCAL_SERVER, TRUE_LATENT_PORT), dispatcher_python, asyncio.get_event_loop()
    )
    server_max = AsyncIOOSCUDPServer(
        (LOCAL_SERVER, MAX_CONTROL_PORT), dispatcher_max, asyncio.get_event_loop()
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


def start_post_recording():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    asyncio_thread = threading.Thread(target=asyncio_run)
    asyncio_thread.start()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        QApplication.instance().exec_()


# Start the PyQtGraph event loop
if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    asyncio_thread = threading.Thread(target=asyncio_run)
    asyncio_thread.start()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        QApplication.instance().exec_()

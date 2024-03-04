import sys

import signal
from tkinter import SE
import numpy as np
import asyncio
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QGraphicsView, QDesktopWidget
from PyQt5.QtCore import QTimerEvent, QTimer
from PyQt5.QtCore import pyqtSignal

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer
from sonification_communication_module import *

import threading

packet_count = 0
num_neurons = 100
SPIKES = [np.zeros(num_neurons)]
LATENT = [np.zeros(8)]

GRID_SIZE_WIDTH = 50
GRID_SIZE_HEIGHT = 30

DISC_RADIUS_INC = [20]
DISC_DECAY_FACTOR = [0.99]
LATENT_DECAY_FACTOR = [0.99]
RASTER_DECAY_FACTOR = [0.1]

WALL_SPIKE = [1]
WALL_TRUE_LATENT = [1]
WALL_INFERRED_LATENT = [1]
CEILING_RASTER = [1]
CEILING_TRUE_LATENT = [1]
CEILING_INFERRED_LATENT = [1]
CEILING_SPIKE = [1]
SPIKE_ORGANIZATION = [1]

COLOR_INDEX = [0]
TRUE_LATENT_COLOR = [51, 255, 51, 255]
INFERRED_LATENT_COLOR = [255, 176, 0, 255]
pg.setConfigOptions(useOpenGL=True)


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

        self.img.setVisible(self.visible)
        self.plot_widget.addItem(self.img)

        self.frame = 0
        self.count = 0

        # Create a custom lookup table (LUT) with green-neon color
        self.lut = []
        for i in range(256):
            # Use RGB value (0, 255, 65) for green-neon color
            self.lut.append([0, 255, 65, i])  # Add alpha channel as the last value

    def animation(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QApplication.instance().exec_()

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

        if self.frame % 10 == 0:
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


class SpikeBubble3DVisualizer:

    def __init__(self, widget=None):
        self.num_neurons = num_neurons
        # Create a PyQtGraph window
        if widget is None:
            self.app = QApplication([])
            self.plot_widget = gl.GLViewWidget()
        else:
            self.plot_widget = widget
        self.plot_widget.setWindowTitle("pyqtgraph example: GLLinePlotItem")
        self.plot_widget.setGeometry(0, 110, 800, 600)
        self.plot_widget.opts["distance"] = 50
        self.plot_widget.show()

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

        if self.frame % 15 == 0:
            self.update_centroid()
            if any(self.data > 0):
                self.trigger_spike(np.where(self.data > 0)[0])
        # print(f"Frame: {self.frame}, Count: {packet_count}")
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


class SpikeBall3DVisualizer:
    def __init__(self, target_location=None, widget=None):
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
                np.random.uniform(-10, 10),
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
        self.velocity = np.random.uniform(-1, 1, (num_neurons, 2))

        self.draw_centroids()
        # Add function that will be called when the timer times out
        self.frame = 0
        self.prev_count = 0
        self.count = 0

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
        self.color[raster_sort_idx % 50 > 25] = [1, 0.7, 0.0, 0.4]

    def update(self):
        if self.frame % 10 == 0:
            self.shrink_circle()
            self.estimate_velocity()
            # self.move_centroids()
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
            self.size[index] += DISC_RADIUS_INC[0]
        else:
            self.size[index] = 0
        self.color[index, -1] = 0.9
        if self.count > 10000:
            SPIKE_ORGANIZATION[0] = 1
        if SPIKE_ORGANIZATION[0] == 1:
            self.velocity = self.centroid_positions - self.target_positions
            self.centroid_positions[index] -= 0.3 * self.velocity[index]

        # # Add the zapping effect element
        # zap_effect = gl.GLScatterPlotItem()
        # zap_effect.setData(
        #     pos=self.centroid_positions[index], size=20, color=(0.8, 0.8, 0.8, 0.3)
        # )
        # self.plot_widget.addItem(zap_effect)
        # QTimer.singleShot(30, lambda: self.plot_widget.removeItem(zap_effect))

        pos = [self.centroid_positions[idx] for idx in index]

    def shrink_circle(self):
        """
        Expands a circle in the animation by gradually increasing its size.

        Args:
            circle (tuple): A tuple containing the circle item and its current radius.
        """
        self.size *= 0.98
        self.color[:, -1] = 0.98 * self.color[:, -1]
        self.indicators.setData(
            size=self.size, pos=self.centroid_positions, color=self.color
        )

    def estimate_velocity(self):
        self.velocity = np.random.uniform(-1, 1, (num_neurons, 3))
        self.centroid_positions += 0.05 * self.velocity
        self.centroid_positions -= 0.0001 * self.centroid_positions


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


class LatentOrbitVisualizer:
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


class LatentOrbitVisualizer_2:
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
        for i in range(6):
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
            for i in range(0, 6):
                pts = np.vstack(
                    [
                        self.data[self.x, slice_window] * GRID_SIZE_HEIGHT / 4,
                        self.data[self.z[i], slice_window] * GRID_SIZE_HEIGHT / 4,
                        self.data[self.z[i + 1], slice_window] * GRID_SIZE_HEIGHT / 4,
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
target_location = np.vstack([np.random.rand(100), np.cos(theta), np.sin(theta)]).T * 7

app = QApplication([])
# -----------------------------Visualization on the wall-----------------------------
wall_plot_widget = gl.GLViewWidget()
wall_plot_widget.setGeometry(0, 0, 1920, 1200)
wall_plot_widget.opts["center"] = QtGui.QVector3D(-GRID_SIZE_WIDTH / 2, 0, 0)
wall_plot_widget.opts["distance"] = 50
wall_plot_widget.opts["fov"] = 100
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
monitor = QDesktopWidget().screenGeometry(1)
wall_plot_widget.move(monitor.left(), monitor.top())
wall_plot_widget.showFullScreen()


vis_wall_raster_left = RasterWithTrace3DVisualizer(
    orientation="left", visible=1, widget=wall_plot_widget
)
vis_wall_raster_right = RasterWithTrace3DVisualizer(
    orientation="right", visible=1, widget=wall_plot_widget
)
vis_wall_raster_top = RasterWithTrace3DVisualizer(
    orientation="top", visible=1, widget=wall_plot_widget
)
vis_wall_raster_bottom = RasterWithTrace3DVisualizer(
    orientation="bot", visible=1, widget=wall_plot_widget
)
vis_wall_spike = SpikeBall3DVisualizer(
    target_location=target_location, widget=wall_plot_widget
)
vis_wall_true_latent = LatentOrbitVisualizer_2(
    0, 1, color=TRUE_LATENT_COLOR, widget=wall_plot_widget
)
vis_wall_inferred_latent = LatentOrbitVisualizer_2(
    1, 1, color=INFERRED_LATENT_COLOR, widget=wall_plot_widget
)

# ---------------------------Visualization on the Ceiling-----------------------------
# visualizer = SpikeDisc2DVisualizer()
# visualizer = SpikeRaster2DVisualizer()


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

        SPIKES[0][:50] = np.array(args[0])
        SPIKES[0][50:] = np.array(args[1])
        SPIKES[0] = SPIKES[0][raster_sort_idx]
        packet_count += 1
        self.spike_trigger.emit()

    def latent_osc_handler(self, address, *args):
        global LATENT, packet_count

        LATENT[0] = np.array(args)
        # packet_count += 1
        self.latent_trigger.emit()

    def max_switch_wall_raster_L(self, address, *args):
        vis_wall_raster_left.img.setVisible(args[0])

    def max_switch_wall_raster_R(self, address, *args):
        vis_wall_raster_right.img.setVisible(args[0])

    def max_switch_wall_raster_T(self, address, *args):
        vis_wall_raster_top.img.setVisible(args[0])

    def max_switch_wall_raster_B(self, address, *args):
        vis_wall_raster_bottom.img.setVisible(args[0])

    def max_switch_wall_spike(self, address, *args):
        vis_wall_spike.indicators.setVisible(args[0])

    def max_switch_wall_true_latent(self, address, *args):
        print(args)
        for trace in vis_wall_true_latent.traces.items():
            trace[1].setVisible(args[0])

    def max_switch_wall_inferred_latent(self, address, *args):
        for trace in vis_wall_true_latent.traces.items():
            trace[1].setVisible(args[0])

    def max_control_osc_handler(self, address, *args):
        exec("global " + address[1:])
        exec(address[1:] + "[0] = args[0]")


spike_pacer = SpikePacer(
    spike_fcn=[
        vis_wall_raster_left.update,
        vis_wall_raster_bottom.update,
        vis_wall_raster_right.update,
        vis_wall_raster_top.update,
        vis_wall_spike.update,
    ],
    latent_fcn=[vis_wall_true_latent.update, vis_wall_inferred_latent.update],
)


async def init_main():
    dispatcher_python = Dispatcher()
    dispatcher_python.map("/SPIKES", spike_pacer.spike_osc_handler)
    dispatcher_python.map("/trajectory", spike_pacer.latent_osc_handler)

    dispatcher_max = Dispatcher()
    dispatcher_max.map("/DISC_RADIUS_INC", spike_pacer.max_control_osc_handler)
    dispatcher_max.map("/SEQUENCE_TRIGGER", spike_pacer.max_control_osc_handler)
    dispatcher_max.map("/WALL_RASTER_L", spike_pacer.max_switch_wall_raster_L)
    dispatcher_max.map("/WALL_RASTER_R", spike_pacer.max_switch_wall_raster_R)
    dispatcher_max.map("/WALL_RASTER_T", spike_pacer.max_switch_wall_raster_T)
    dispatcher_max.map("/WALL_RASTER_B", spike_pacer.max_switch_wall_raster_B)
    dispatcher_max.map("/WALL_TRUE_LATENT", spike_pacer.max_switch_wall_true_latent)
    dispatcher_max.map(
        "/WALL_INFERRED_LATENT", spike_pacer.max_switch_wall_inferred_latent
    )
    dispatcher_max.map("/WALL_SPIKE", spike_pacer.max_switch_wall_spike)

    server_spike = AsyncIOOSCUDPServer(
        (LOCAL_SERVER, SPIKE_PORT), dispatcher_python, asyncio.get_event_loop()
    )
    server_latent = AsyncIOOSCUDPServer(
        (LOCAL_SERVER, TRUE_LATENT_PORT), dispatcher_python, asyncio.get_event_loop()
    )
    server_max = AsyncIOOSCUDPServer(
        ("0.0.0.0", MAX_CONTROL_PORT), dispatcher_max, asyncio.get_event_loop()
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

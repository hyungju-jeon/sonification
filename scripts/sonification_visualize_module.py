import asyncio
import signal
import sys
import threading

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QDesktopWidget, QGridLayout
from pyqtgraph.Qt import QtCore, QtGui
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from sonification_communication_module import *

cdict = {
    "red": [[0.0, 1, 1], [1.0, 0.2, 0.2]],
    "green": [[0.0, 0.7, 0.7], [1.0, 1, 1]],
    "blue": [[0.0, 0, 0], [1.0, 0.2, 0.2]],
}
AMGR_CMAP = LinearSegmentedColormap("AmGr", segmentdata=cdict, N=1024)
RAINBOW_CMAP = plt.get_cmap("turbo", 1024).reversed()
RGBA = np.round(AMGR_CMAP(np.linspace(0, 1, 256)) * 255).astype(int)

num_neurons = 100
SPIKES = [np.zeros(num_neurons)]
LATENT = [np.zeros(4)]
INFERRED = [np.zeros(4)]

ASPECT_RATIO = 1
WIDGET_SIZE = 1024
VIS_DEPTH = -25
VIS_RADIUS = 10
VIS_RADIUS_LATENT = 15
GRID_SIZE_WIDTH = 100
GRID_SIZE_HEIGHT = 50

LATENT_DECAY_FACTOR = [1]
INFERRED_DECAY_FACTOR = [0.95]

DISC_RADIUS_INC = [10]
DISC_DECAY_FACTOR = [0.90]
RASTER_DECAY_FACTOR = [0.9]

SWITCH_WALL_RASTER_LEFT = [0]
SWITCH_WALL_RASTER_RIGHT = [0]
SWITCH_WALL_RASTER_TOP = [0]
SWITCH_WALL_RASTER_BOTTOM = [0]
SWITCH_WALL_SPIKE = [0]
SWITCH_WALL_TRUE_LATENT = [1]
SWITCH_WALL_INFERRED_LATENT = [0]

SWITCH_CEILING_RASTER = [0]
SWITCH_CEILING_SPIKE = [1]

SPIKE_ORGANIZATION = [0]
GW_COLOR = [1]
SWITCH_GRID = [0]

COLOR_INDEX = [0]
WHITE_COLOR = [200, 200, 200, 255]
WHITE_CEILING_COLOR = [255, 255, 255, 255]
GREEN_COLOR = [51, 255, 51, 255]
AMBER_COLOR = [255, 176, 0, 255]

SCALE_FACTOR = GRID_SIZE_HEIGHT / 4
pg.setConfigOptions(useOpenGL=False)


def compute_decay_factor(d):
    return np.exp(np.log(0.1) / d)


class RasterPlaneVisualizer:
    def __init__(
        self, orientation, visible=True, max_level=10, separate=False, widget=None
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

        self.num_neurons = num_neurons // 2 if separate else num_neurons
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
        elif orientation == "ceiling":
            self.slicer = slice(0, 100) if separate else slice(0, 100)
            scale_factor = [
                GRID_SIZE_HEIGHT / self.L,
                GRID_SIZE_WIDTH / self.num_neurons,
                1,
            ]
            self.img.scale(*scale_factor)
            self.img.rotate(-90, 0, 1, 0)
            self.img.translate(
                -GRID_SIZE_WIDTH / 2 + GRID_SIZE_HEIGHT / 2,
                -GRID_SIZE_WIDTH / 2,
                -GRID_SIZE_HEIGHT / 2,
            )

        self.img.setVisible(self.visible)
        self.plot_widget.addItem(self.img)

        self.frame = 0
        self.count = 0

        # Create a custom lookup table (LUT) with green-neon color
        self.lut = [[*WHITE_COLOR[:3], i] for i in range(256)]
        if orientation == "ceiling":
            self.lut = [[*WHITE_CEILING_COLOR[:3], i] for i in range(256)]


    def animation(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QApplication.instance().exec_()

    def update_lut(self, color):
        self.lut = [[*color[:3], i] for i in range(256)]

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
                self.firing_rates[
                    self.frame, i
                ] += self.max_level//3  # Instantly increase firing rate

        if self.frame % 10 == 0 and self.visible:
            if self.count >= self.L:
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


class SpikeBallVisualizer:
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
        self.centroid_positions = np.random.uniform(
            -VIS_RADIUS, VIS_RADIUS, (num_neurons, 3)
        )
        self.centroid_positions[:, 0] += VIS_DEPTH
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
        self.visible = visible

        self.MAX_OSCsender = SimpleUDPClient(MAX_SERVER, MAX_OUTPUT_PORT)

    def animation(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QApplication.instance().exec_()

    def draw_centroids(self):
        indicators = gl.GLScatterPlotItem()
        indicators.setData(
            pos=self.centroid_positions,
            size=0,
            color=(0.2, 1, 0.2, 0),
        )
        self.plot_widget.addItem(indicators)
        self.indicators = indicators
        self.size = np.ones(num_neurons) * 5
        self.color = np.repeat(
            np.array([*np.array(WHITE_COLOR[:3]) / 255, 0.4])[np.newaxis, :],
            num_neurons,
            axis=0,
        )
        self.color[raster_sort_idx > 50] = np.array(
            [*np.array(AMBER_COLOR[:3]) / 255, 0.4]
        )

    def update_color(self, color):
        self.color[raster_sort_idx < 50] = np.array([*np.array(color[:3]) / 255, 0.4])

    def update(self):
        if self.frame % 10 == 0:
            self.shrink_circle()
            self.estimate_velocity()
            if any(SPIKES[0] > 0):
                self.trigger_spike(np.where(SPIKES[0] > 0)[0])

            self.theta = (
                (
                    np.arctan2(
                        self.centroid_positions[:, 2], self.centroid_positions[:, 1]
                    )
                    + np.pi
                )
                * 360
                / (2 * np.pi)
            )
            self.MAX_OSCsender.send_message(
                "/SPIKE_POSITION",
                self.theta[raster_sort_idx_inv].tolist(),
            )
        self.frame += 1
        self.count += 1

    def trigger_spike(self, index):
        """
        Triggers a spike animation at the specified index.

        Args:
            index (int): The index of the spike to trigger.
        """
        if self.visible == 1:
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


class TrueLatentVisualizer:
    def __init__(self, ref_color, var_color, visible=False, widget=None):
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

        self.num_latent = LATENT[0].shape[0]
        self.L_ref = 1500
        self.L_var = 5000
        self.buffer = 5000
        self.max_length = 20000 + self.buffer
        self.latent = np.zeros((self.max_length, self.num_latent))
        self.decay_ref = compute_decay_factor(self.L_ref)
        self.decay_var = compute_decay_factor(self.L_var)
        self.traces = dict()

        self.color_ref = np.repeat(
            np.array(ref_color)[np.newaxis, :] / 255, self.L_ref, axis=0
        )
        if isinstance(var_color, LinearSegmentedColormap) or isinstance(
            var_color, ListedColormap
        ):
            self.colormap = var_color
            self.color_var = var_color(np.linspace(0, 1, self.L_var))
        else:
            self.colormap = None
            self.color_var = np.repeat(
                np.array(var_color)[np.newaxis, :] / 255, self.L_var, axis=0
            )

        for i in range(1, self.L_ref):
            self.color_ref[i][-1] = self.color_ref[i - 1][-1] * self.decay_ref

        for i in range(1, self.L_var):
            self.color_var[i][-1] = self.color_var[i - 1][-1] * self.decay_var

        self.color_ref = self.color_ref[::-1]
        self.color_var = self.color_var[::-1]
        self.data = np.zeros((self.num_latent, self.max_length))

        for i in range(self.num_latent // 2):
            if i == 0:
                self.traces[i] = gl.GLLinePlotItem(
                    pos=np.zeros((self.L_ref, 3)),
                    color=self.color_ref,
                    width=5,
                    antialias=False,
                )
            else:
                self.traces[i] = gl.GLLinePlotItem(
                    pos=np.zeros((self.L_var, 3)),
                    color=self.color_var,
                    width=5,
                    antialias=False,
                )
            self.traces[i].setVisible(visible)
            self.plot_widget.addItem(self.traces[i])
        self.last_frame = 0
        self.visible = visible

    def animation(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QApplication.instance().exec_()

    def update_color(self, color):
        self.color_ref[:, :3] = np.repeat(
            np.array(color)[np.newaxis, :3] / 255, self.L_ref, axis=0
        )
        self.traces[0].color = self.color_ref

    def update_trail_length(self, L_var):
        self.L_var = L_var
        self.decay_var = compute_decay_factor(self.L_var)
        self.color_var = self.colormap(np.linspace(0, 1, self.L_var))
        for i in range(1, self.L_var):
            self.color_var[i][-1] = self.color_var[i - 1][-1] * self.decay_var
        self.color_var = self.color_var[::-1]
        self.traces[1].color = self.color_var

    def update(self):
        if self.last_frame >= self.max_length:
            self.data = np.roll(self.data, -self.buffer, axis=1)
            self.last_frame -= self.buffer
        if self.last_frame > 0:
            self.data[:, self.last_frame] = LATENT[0]
        if self.last_frame % 10 == 0 and self.last_frame > 0 and self.visible:
            for i in range(self.num_latent // 2):
                if i == 0:
                    slice_window = slice(
                        np.fmax(0, self.last_frame - self.L_ref), self.last_frame
                    )
                else:
                    slice_window = slice(
                        np.fmax(0, self.last_frame - self.L_var), self.last_frame
                    )
                    if self.last_frame < self.L_var:
                        self.traces[i].color = self.color_var[
                            self.L_var - self.last_frame : self.L_var, :
                        ]

                pts = np.vstack(
                    [
                        np.ones_like(self.data[0, slice_window]) * VIS_DEPTH,
                        self.data[2 * i, slice_window] * VIS_RADIUS_LATENT ,
                        self.data[2 * i + 1, slice_window] * VIS_RADIUS_LATENT,
                    ]
                ).transpose()
                self.traces[i].setData(
                    pos=pts,
                )
        self.last_frame += 1


class InferredLatentVisualizer:
    def __init__(self, ref_color, var_color, visible=False, widget=None):
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

        self.num_latent = LATENT[0].shape[0]
        self.L_ref = 75
        self.L_var = 250
        self.buffer = 5000
        self.max_length = 20000 + self.buffer
        self.latent = np.zeros((self.max_length, self.num_latent))
        self.decay_ref = compute_decay_factor(self.L_ref)
        print(self.decay_ref)
        self.decay_var = compute_decay_factor(self.L_var)
        self.traces = dict()

        self.color_ref = np.repeat(
            np.array(ref_color)[np.newaxis, :] / 255, self.L_ref, axis=0
        )
        if isinstance(var_color, LinearSegmentedColormap) or isinstance(
            var_color, ListedColormap
        ):
            self.colormap = var_color
            self.color_var = var_color(np.linspace(0, 1, self.L_var))
        else:
            self.colormap = None
            self.color_var = np.repeat(
                np.array(var_color)[np.newaxis, :] / 255, self.L_var, axis=0
            )

        for i in range(1, self.L_ref):
            self.color_ref[i][-1] = self.color_ref[i - 1][-1] * self.decay_ref

        for i in range(1, self.L_var):
            self.color_var[i][-1] = self.color_var[i - 1][-1] * self.decay_var

        self.color_ref = self.color_ref[::-1]
        self.color_var = self.color_var[::-1]
        self.data = np.zeros((self.num_latent, self.max_length))

        for i in range(self.num_latent // 2):
            if i == 0:
                self.traces[i] = gl.GLLinePlotItem(
                    pos=np.zeros((self.L_ref, 3)),
                    color=self.color_ref,
                    width=1,
                    antialias=False,
                )
            else:
                self.traces[i] = gl.GLLinePlotItem(
                    pos=np.zeros((self.L_var, 3)),
                    color=self.color_var,
                    width=1,
                    antialias=False,
                )
            self.traces[i].setVisible(visible)
            self.plot_widget.addItem(self.traces[i])
        self.last_frame = 0
        self.visible = visible

    def animation(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QApplication.instance().exec_()

    def update_color(self, color):
        self.color_ref[:, :3] = np.repeat(
            np.array(color)[np.newaxis, :3] / 255, self.L_ref, axis=0
        )
        self.traces[0].color = self.color_ref

    def update_trail_length(self, L_var):
        self.L_var = L_var // 20
        self.decay_var = compute_decay_factor(self.L_var)
        self.color_var = self.colormap(np.linspace(0, 1, self.L_var))
        for i in range(1, self.L_var):
            self.color_var[i][-1] = self.color_var[i - 1][-1] * self.decay_var
        self.color_var = self.color_var[::-1]
        self.traces[1].color = self.color_var

    def update(self):
        if self.last_frame >= self.max_length:
            self.data = np.roll(self.data, -self.buffer, axis=1)
            self.last_frame -= self.buffer
        if self.last_frame > 0:
            self.data[:, self.last_frame] = INFERRED[0]
        if self.last_frame % 1 == 0 and self.last_frame > 0 and self.visible:
            for i in range(self.num_latent // 2):
                if i == 0:
                    slice_window = slice(
                        np.fmax(0, self.last_frame - self.L_ref), self.last_frame
                    )
                else:
                    slice_window = slice(
                        np.fmax(0, self.last_frame - self.L_var), self.last_frame
                    )
                    if self.last_frame < self.L_var:
                        self.traces[i].color = self.color_var[
                            self.L_var - self.last_frame : self.L_var, :
                        ]

                pts = np.vstack(
                    [
                        np.ones_like(self.data[0, slice_window]) * VIS_DEPTH,
                        self.data[2 * i, slice_window] * VIS_RADIUS_LATENT,
                        self.data[2 * i + 1, slice_window] * VIS_RADIUS_LATENT,
                    ]
                ).transpose()
                self.traces[i].setData(
                    pos=pts,
                )
        self.last_frame += 1


# -------------------------------------------------------------------------------
loading_matrix_slow_name = "./data/loading_matrix_slow.npz"

param = np.load(loading_matrix_slow_name, allow_pickle=True)
C_slow, b_slow = np.hstack(param["C"]), param["b"]

C = np.hstack([C_slow])
b = np.vstack([b_slow]).flatten()
theta = np.arctan2(C[1, :], C[0, :])

raster_sort_idx = np.argsort(theta)
raster_sort_idx_inv = np.argsort(raster_sort_idx)
theta = theta[raster_sort_idx]
target_location = np.vstack(
    [
        np.ones_like(theta) * VIS_DEPTH,
        np.cos(theta) * VIS_RADIUS * 2,
        np.sin(theta) * VIS_RADIUS * 2,
    ]
).T

app = QApplication([])
# -----------------------------Visualization on the wall-----------------------------
wall_parent_widget = gl.GLViewWidget()
wall_parent_widget.setGeometry(0, 0, 1920, 1200)
layout = QGridLayout()
wall_parent_widget.setLayout(layout)
monitor = QDesktopWidget().screenGeometry(2)
wall_parent_widget.move(monitor.left(), monitor.top())

wall_widget = gl.GLViewWidget(parent=wall_parent_widget)
wall_widget.setGeometry(0, 0, 1920, 1200)
wall_widget.opts["center"] = QtGui.QVector3D(-30, 0, 0)
wall_widget.opts["distance"] = 75
wall_widget.opts["fov"] = 90
wall_widget.opts["elevation"] = 0
wall_widget.opts["azimuth"] = 0
wall_parent_widget.showMaximized()
wall_parent_widget.showFullScreen()

#g_center = gl.GLGridItem(QtGui.QVector3D(GRID_SIZE_HEIGHT, GRID_SIZE_WIDTH, 1))
#g_center.rotate(90, 0, 1, 0)
#g_center.translate(-GRID_SIZE_WIDTH / 2 + GRID_SIZE_HEIGHT / 2, 0, 0)
#wall_widget.addItem(g_center)

#g_left = gl.GLGridItem(QtGui.QVector3D(GRID_SIZE_WIDTH, GRID_SIZE_HEIGHT, 1))
#g_left.rotate(90, 1, 0, 0)
#g_left.translate(GRID_SIZE_HEIGHT / 2, -GRID_SIZE_WIDTH / 2, 0)
#wall_widget.addItem(g_left)

#g_floor = gl.GLGridItem(QtGui.QVector3D(GRID_SIZE_WIDTH, GRID_SIZE_WIDTH, 1))
#g_floor.translate(GRID_SIZE_HEIGHT / 2, 0, -GRID_SIZE_HEIGHT / 2 + 0)
#wall_widget.addItem(g_floor)

#g_right = gl.GLGridItem(QtGui.QVector3D(GRID_SIZE_WIDTH, GRID_SIZE_HEIGHT, 1))
#g_right.rotate(-90, 1, 0, 0)
#g_right.translate(GRID_SIZE_HEIGHT / 2, GRID_SIZE_WIDTH / 2, 0)
#wall_widget.addItem(g_right)

layout.addWidget(wall_widget, 1, 0)
wall_parent_widget.show()

vis_wall_raster_left = RasterPlaneVisualizer(
    orientation="left", visible=SWITCH_WALL_RASTER_LEFT[0], widget=wall_widget
)
vis_wall_raster_right = RasterPlaneVisualizer(
    orientation="right", visible=SWITCH_WALL_RASTER_RIGHT[0], widget=wall_widget
)
vis_wall_raster_top = RasterPlaneVisualizer(
    orientation="top", visible=SWITCH_WALL_RASTER_TOP[0], widget=wall_widget
)
vis_wall_raster_bottom = RasterPlaneVisualizer(
    orientation="bot", visible=SWITCH_WALL_RASTER_BOTTOM[0], widget=wall_widget
)

vis_wall_true_latent = TrueLatentVisualizer(
    ref_color=WHITE_COLOR,
    var_color=RAINBOW_CMAP,
    visible=SWITCH_WALL_TRUE_LATENT[0],
    widget=wall_widget,
)
vis_wall_inferred_latent = InferredLatentVisualizer(
    ref_color=AMBER_COLOR,
    var_color=AMGR_CMAP,
    visible=SWITCH_WALL_INFERRED_LATENT[0],
    widget=wall_widget,
)

# ---------------------------Visualization on the Ceiling-----------------------------
ceiling_parent_widget = gl.GLViewWidget()
ceiling_parent_widget.setGeometry(0, 0, 1920, 1200)
layout = QGridLayout()
ceiling_parent_widget.setLayout(layout)
monitor = QDesktopWidget().screenGeometry(1)
ceiling_parent_widget.move(monitor.left(), monitor.top())
ceiling_parent_widget.show()

ceiling_widget = gl.GLViewWidget(parent=ceiling_parent_widget)
ceiling_widget.setGeometry(0, 0, 1920, 1200)
ceiling_widget.opts["center"] = QtGui.QVector3D(-30, 0, 0)
ceiling_widget.opts["distance"] = 55
ceiling_widget.opts["fov"] = 90
ceiling_widget.opts["elevation"] = 0
ceiling_widget.opts["azimuth"] = 0
ceiling_parent_widget.showMaximized()
ceiling_parent_widget.showFullScreen()


vis_ceiling_spike = SpikeBallVisualizer(
    target_location=target_location,
    visible=SWITCH_CEILING_SPIKE[0],
    widget=ceiling_widget,
)
vis_ceiling_raster = RasterPlaneVisualizer(
    orientation="ceiling",
    visible=SWITCH_CEILING_RASTER[0],
    widget=ceiling_widget,
)
# ---------------------------OSC mapping-----------------------------


class SpikePacer(QtCore.QObject):
    spike_trigger = pyqtSignal()
    latent_trigger = pyqtSignal()
    inferred_trigger = pyqtSignal()

    def __init__(self, spike_fcns, latent_fcns, inferred_fcns):
        super().__init__()
        for fcn in spike_fcns:
            self.spike_trigger.connect(fcn)
        for fcn in latent_fcns:
            self.latent_trigger.connect(fcn)
        for fcn in inferred_fcns:
            self.inferred_trigger.connect(fcn)

    def spike_osc_handler(self, address, *args):
        global SPIKES
        SPIKES[0] = np.array(args)
        SPIKES[0] = SPIKES[0][raster_sort_idx]
        self.spike_trigger.emit()

    def latent_osc_handler(self, address, *args):
        global LATENT

        LATENT[0] = np.array(args)
        self.latent_trigger.emit()

    def inferred_osc_handler(self, address, *args):
        global INFERRED

        INFERRED[0] = np.array(args)
        self.inferred_trigger.emit()

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
        # vis_wall_spike.visible = args[0]
        # vis_wall_spike.indicators.setVisible(args[0])
        pass

    def max_switch_wall_true_latent(self, address, *args):
        vis_wall_true_latent.visible = args[0]
        for trace in vis_wall_true_latent.traces.items():
            trace[1].setVisible(args[0])

    def max_switch_wall_inferred_latent(self, address, *args):
        vis_wall_inferred_latent.visible = args[0]
        for trace in vis_wall_inferred_latent.traces.items():
            trace[1].setVisible(args[0])

    def max_switch_ceiling_raster(self, address, *args):
        vis_ceiling_raster.visible = args[0]
        vis_ceiling_raster.img.setVisible(args[0])

    def max_switch_ceiling_spike(self, address, *args):
        vis_ceiling_spike.visible = args[0]
        vis_ceiling_spike.indicators.setVisible(args[0])

    def max_control_true_trail_length(self, address, *args):
        vis_wall_true_latent.update_trail_length(args[0])

    def max_control_inferred_trail_length(self, address, *args):
        vis_wall_inferred_latent.update_trail_length(args[0])

    def max_control_GW_color(self, address, *args):
        if args[0] == 0:
            color = WHITE_COLOR
        else:
            color = GREEN_COLOR
        vis_wall_raster_left.update_lut(color)
        vis_wall_raster_right.update_lut(color)
        vis_wall_raster_top.update_lut(color)
        vis_wall_raster_bottom.update_lut(color)
        vis_ceiling_raster.update_lut(color)
        # vis_wall_spike.update_color(color)
        vis_ceiling_spike.update_color(color)
        vis_wall_true_latent.update_color(color)

    def max_control_osc_handler(self, address, *args):
        exec("global " + address[1:])
        exec(address[1:] + "[0] = args[0]")


spike_pacer = SpikePacer(
    spike_fcns=[
        vis_wall_raster_left.update,
        vis_wall_raster_bottom.update,
        vis_wall_raster_right.update,
        vis_wall_raster_top.update,
        # vis_wall_spike.update,
        vis_ceiling_raster.update,
        vis_ceiling_spike.update,
    ],
    latent_fcns=[
        vis_wall_true_latent.update,
    ],
    inferred_fcns=[
        vis_wall_inferred_latent.update,
    ],
)


async def init_main():
    dispatcher_python = Dispatcher()
    dispatcher_python.map("/SPIKES", spike_pacer.spike_osc_handler)
    dispatcher_python.map("/TRAJECTORY", spike_pacer.latent_osc_handler)
    dispatcher_python.map("/INFERRED_TRAJECTORY", spike_pacer.inferred_osc_handler)

    dispatcher_max = Dispatcher()
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
    dispatcher_max.map("/CEILING_SPIKE", spike_pacer.max_switch_ceiling_spike)

    dispatcher_max.map("/GW_COLOR", spike_pacer.max_control_GW_color)
    dispatcher_max.map("/TRUE_LENGTH", spike_pacer.max_control_true_trail_length)
    dispatcher_max.map("/INFERRED_LENGTH", spike_pacer.max_control_inferred_trail_length)

    server_spike = AsyncIOOSCUDPServer(
        (LOCAL_SERVER, SPIKE_VISUALIZE_PORT),
        dispatcher_python,
        asyncio.get_event_loop(),
    )
    server_latent = AsyncIOOSCUDPServer(
        (LOCAL_SERVER, LATENT_PORT), dispatcher_python, asyncio.get_event_loop()
    )

    server_max = AsyncIOOSCUDPServer(
        (LOCAL_SERVER, MAX_CONTROL_PORT), dispatcher_max, asyncio.get_event_loop()
    )
    await server_latent.create_serve_endpoint()
    await server_spike.create_serve_endpoint()
    await server_max.create_serve_endpoint()
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

import asyncio
import signal
import sys
import threading

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtCore import pyqtSignal
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

num_neurons = 100
SPIKES = [np.zeros(num_neurons)]
LATENT = [np.zeros(4)]
INFERRED = [np.zeros(4)]

ASPECT_RATIO = 1
WIDGET_SIZE = 1024
VIS_DEPTH = -25
VIS_RADIUS = 10

DISC_RADIUS_INC = [10]
DISC_DECAY_FACTOR = [0.90]
RASTER_DECAY_FACTOR = [0.1]

TRUE_LATENT_LENGTH = [5000]
INFERRED_LATENT_LENGTH = [5000]

SWITCH_RASTER_VISUALIZE = [1]
SWITCH_TRUE_LATENT_VISUALIZE = [1]
SWITCH_INFERRED_LATENT_VISUALIZE = [1]
SWITCH_SPIKE_VISUALIZE = [1]
SWITCH_SPIKE_ORGANIZATION = [0]
SWITCH_GW_COLOR = [0]

COLOR_INDEX = [0]
WHITE_COLOR = [200, 200, 200, 255]
GREEN_COLOR = [51, 255, 51, 255]
AMBER_COLOR = [255, 176, 0, 255]

pg.setConfigOptions(useOpenGL=True)


def compute_decay_factor(d):
    return np.exp(np.log(0.1) / d)


class RasterCircularVisualizer:
    def __init__(self, visible=True, num_images=30, max_level=5, widget=None):
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

        self.num_neurons = num_neurons
        self.num_images = num_images
        self.neuron_per_image = self.num_neurons // self.num_images
        self.L = 1000
        self.buffer = 1000
        self.firing_rates = np.zeros((self.L + self.buffer, num_neurons))
        self.decay_factor = RASTER_DECAY_FACTOR[0]  # Exponential decay factor
        self.max_level = max_level
        self.visible = visible

        raster_texture = pg.makeRGBA(
            self.firing_rates[:, : self.neuron_per_image], levels=(0, max_level)
        )[0]
        self.imgs = [gl.GLImageItem(raster_texture) for _ in range(self.num_images)]

        # Arrange each image in a circle
        for i in range(self.num_images):
            self.slicer = slice(0, 100)
            scale_factor = [
                -0.4,
                2
                * (WIDGET_SIZE / 4)
                * np.tan(np.pi / self.num_images)
                / self.neuron_per_image,
                1,
            ]
            tr_scale = pg.Transform3D()
            tr_scale.scale(*scale_factor)
            tr_rotate = pg.Transform3D()
            tr_rotate.rotate(
                360 / self.num_images * i + 90 - (360 / self.num_images) / 2, 1, 0, 0
            )
            tr_translate = pg.Transform3D()
            tr_translate.translate(
                0,
                -(WIDGET_SIZE / 4) * np.cos(2 * np.pi * i / self.num_images),
                -(WIDGET_SIZE / 4) * np.sin(2 * np.pi * i / self.num_images),
            )
            self.imgs[i].applyTransform(tr_translate * tr_rotate * tr_scale, local=True)
            self.imgs[i].setVisible(self.visible)
            self.plot_widget.addItem(self.imgs[i])

        self.frame = 0
        self.count = 0

        # Create a custom lookup table (LUT) with green-neon color
        self.lut = [[*GREEN_COLOR[:3], i] for i in range(256)]

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
                ] += self.max_level  # Instantly increase firing rate

        if self.frame % 10 == 0 and self.visible:
            if self.count >= self.L:
                for n_img in range(self.num_images):
                    raster_texture = pg.makeRGBA(
                        self.firing_rates[
                            np.fmax(0, self.frame - self.L) : self.frame,
                            n_img
                            * self.neuron_per_image : (n_img + 1)
                            * self.neuron_per_image,
                        ],
                        levels=(0, self.max_level),
                        lut=self.lut,
                    )[0]
                    self.imgs[n_img].setData(raster_texture)
            else:
                rolled_rate = self.firing_rates[: self.L, :]
                rolled_rate = np.roll(rolled_rate, 1000 - self.count, axis=0)

                for n_img in range(self.num_images):
                    raster_texture = pg.makeRGBA(
                        rolled_rate[
                            :,
                            n_img
                            * self.neuron_per_image : (n_img + 1)
                            * self.neuron_per_image,
                        ],
                        levels=(0, self.max_level),
                        lut=self.lut,
                    )[0]
                    self.imgs[n_img].setData(raster_texture)

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

    def update_color(self, color):
        self.color[:, :3] = np.array(color[:3]) / 255

    def update(self):
        if self.frame % 10 == 0:
            self.shrink_circle()
            self.estimate_velocity()

            if any(SPIKES[0] > 0):
                self.trigger_spike(np.where(SPIKES[0] > 0)[0])
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
        if SWITCH_SPIKE_ORGANIZATION[0]:
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
        if not SWITCH_SPIKE_ORGANIZATION[0]:
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
        self.traces[-1].color = self.color_var

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
                        self.data[2 * i, slice_window] * VIS_RADIUS,
                        self.data[2 * i + 1, slice_window] * VIS_RADIUS,
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
        self.L_var = L_var // 20
        self.decay_var = compute_decay_factor(self.L_var)
        self.color_var = self.colormap(np.linspace(0, 1, self.L_var))
        for i in range(1, self.L_var):
            self.color_var[i][-1] = self.color_var[i - 1][-1] * self.decay_var
        self.color_var = self.color_var[::-1]
        self.traces[-1].color = self.color_var

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
                        self.data[2 * i, slice_window] * VIS_RADIUS,
                        self.data[2 * i + 1, slice_window] * VIS_RADIUS,
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
theta = theta[raster_sort_idx]
target_location = np.vstack(
    [
        np.ones_like(theta) * VIS_DEPTH,
        np.cos(theta) * VIS_RADIUS,
        np.sin(theta) * VIS_RADIUS,
    ]
).T

app = QApplication([])
# -----------------------------Visualization on the floor-----------------------------
parent_widget = gl.GLViewWidget()
parent_widget.setGeometry(0, 0, 1920, 1200)
layout = QGridLayout()
parent_widget.setLayout(layout)
monitor = QDesktopWidget().screenGeometry(0)
parent_widget.move(monitor.left(), monitor.top())
parent_widget.show()

floor_widget = gl.GLViewWidget(parent=parent_widget)
floor_widget.setGeometry(360, 0, 1200, 1200)
floor_widget.opts["center"] = QtGui.QVector3D(-30, 0, 0)
floor_widget.opts["distance"] = 55
floor_widget.opts["fov"] = 90
floor_widget.opts["elevation"] = 0
floor_widget.opts["azimuth"] = 0

vis_spike = SpikeBallVisualizer(
    target_location=target_location,
    visible=SWITCH_SPIKE_VISUALIZE[0],
    widget=floor_widget,
)
vis_true_latent = TrueLatentVisualizer(
    ref_color=GREEN_COLOR,
    var_color=RAINBOW_CMAP,
    visible=SWITCH_TRUE_LATENT_VISUALIZE[0],
    widget=floor_widget,
)
vis_inferred_latent = InferredLatentVisualizer(
    ref_color=AMBER_COLOR,
    var_color=AMGR_CMAP,
    visible=SWITCH_INFERRED_LATENT_VISUALIZE[0],
    widget=floor_widget,
)
vis_raster = RasterCircularVisualizer(
    visible=SWITCH_RASTER_VISUALIZE[0], widget=floor_widget
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

    def max_switch_spike_vis(self, address, *args):
        vis_spike.visible = args[0]
        vis_spike.indicators.setVisible(args[0])

    def max_switch_true_latent_vis(self, address, *args):
        vis_true_latent.visible = args[0]
        for trace in vis_true_latent.traces.items():
            trace[1].setVisible(args[0])

    def max_switch_inferred_latent_vis(self, address, *args):
        vis_inferred_latent.visible = args[0]
        for trace in vis_inferred_latent.traces.items():
            trace[1].setVisible(args[0])

    def max_switch_raster_vis(self, address, *args):
        vis_raster.visible = args[0]
        for img in vis_raster.imgs:
            img.setVisible(args[0])

    def max_control_osc_handler(self, address, *args):
        exec("global " + address[1:])
        exec(address[1:] + "[0] = args[0]")

    def max_control_true_trail_length(self, address, *args):
        vis_true_latent.update_trail_length(args[0])

    def max_control_inferred_trail_length(self, address, *args):
        vis_inferred_latent.update_trail_length(args[0])

    def max_control_GW_color(self, address, *args):
        if args[0] == 0:
            color = WHITE_COLOR
        else:
            color = GREEN_COLOR
        vis_raster.update_lut(color)
        vis_spike.update_color(color)
        vis_true_latent.update_color(color)


spike_pacer = SpikePacer(
    spike_fcns=[
        vis_spike.update,
        vis_raster.update,
    ],
    latent_fcns=[
        vis_true_latent.update,
    ],
    inferred_fcns=[
        vis_inferred_latent.update,
    ],
)


async def init_main():
    dispatcher_python = Dispatcher()
    dispatcher_python.map("/SPIKES", spike_pacer.spike_osc_handler)
    dispatcher_python.map("/TRAJECTORY", spike_pacer.latent_osc_handler)
    dispatcher_python.map("/INFERRED_TRAJECTORY", spike_pacer.inferred_osc_handler)

    dispatcher_max = Dispatcher()
    dispatcher_max.map("/GW_COLOR", spike_pacer.max_control_GW_color)
    dispatcher_max.map("/TRUE_LENGTH", spike_pacer.max_control_true_trail_length)
    dispatcher_max.map("/INFERRED_LENGTH", spike_pacer.max_control_true_trail_length)
    dispatcher_max.map("/DISC_RADIUS_INC", spike_pacer.max_control_osc_handler)
    dispatcher_max.map("/SPIKE_ORGANIZATION", spike_pacer.max_control_osc_handler)
    dispatcher_max.map("/SPIKE", spike_pacer.max_switch_spike_vis)
    dispatcher_max.map("/TRUE_LATENT", spike_pacer.max_switch_true_latent_vis)
    dispatcher_max.map("/INFERRED_LATENT", spike_pacer.max_switch_inferred_latent_vis)
    dispatcher_max.map("/RASTER", spike_pacer.max_switch_raster_vis)

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
    await server_spike.create_serve_endpoint()
    await server_latent.create_serve_endpoint()
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

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
from matplotlib.colors import LinearSegmentedColormap
from sonification_communication_module import *

cdict = {
    "red": [[0.0, 0.2, 0.2], [1.0, 1.0, 1.0]],
    "green": [[0.0, 1.0, 1.0], [1.0, 0.7, 0.7]],
    "blue": [[0.0, 0.2, 0.2], [1.0, 0.0, 0.0]],
}
GRADIENT_CMAP = LinearSegmentedColormap("testCmap", segmentdata=cdict, N=1024)
RGBA = np.round(GRADIENT_CMAP(np.linspace(0, 1, 256)) * 255).astype(int)

packet_count = 0
num_neurons = 100
SPIKES = [np.zeros(num_neurons)]
LATENT = [np.zeros(4)]
INFERRED = [np.zeros(4)]

ASPECT_RATIO = 1
WIDGET_SIZE = 1024
VIS_DEPTH = -25
VIS_RADIUS = 12.5

DISC_RADIUS_INC = [10]
DISC_DECAY_FACTOR = [0.90]
LATENT_DECAY_FACTOR = [1]
INFERRED_DECAY_FACTOR = [0.95]
RASTER_DECAY_FACTOR = [0.1]

SWITCH_RASTER = [1]
SWITCH_TRUE_LATENT = [1]
SWITCH_INFERRED_LATENT = [1]
SWITCH_SPIKE = [1]
SWITCH_SPIKE_ORGANIZATION = [1]

COLOR_INDEX = [0]
TRUE_LATENT_COLOR = [51, 255, 51, 255]
INFERRED_LATENT_COLOR = [255, 176, 0, 255]

pg.setConfigOptions(useOpenGL=True)


def compute_decay_factor(d):
    pass


class RasterCircularVisualizer:
    def __init__(self, visible=True, max_level=5, widget=None):
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
        self.num_image = 30
        self.neuron_per_image = self.num_neurons // self.num_image
        self.L = 1000
        self.buffer = 1000
        self.firing_rates = np.zeros((self.L + self.buffer, num_neurons))
        self.decay_factor = RASTER_DECAY_FACTOR[0]  # Exponential decay factor
        self.max_level = max_level
        self.visible = visible

        raster_texture = pg.makeRGBA(
            self.firing_rates[:, : self.neuron_per_image], levels=(0, max_level)
        )[0]
        self.imgs = [gl.GLImageItem(raster_texture) for _ in range(self.num_image)]

        # Arrange each image in a circle
        for i in range(self.num_image):
            self.slicer = slice(0, 100)
            scale_factor = [
                -0.5,
                2
                * (WIDGET_SIZE / 2)
                * np.tan(np.pi / self.num_image)
                / self.neuron_per_image,
                1,
            ]
            tr_scale = pg.Transform3D()
            tr_scale.scale(*scale_factor)
            tr_rotate = pg.Transform3D()
            tr_rotate.rotate(
                360 / self.num_image * i + 90 - (360 / self.num_image) / 2, 1, 0, 0
            )
            tr_translate = pg.Transform3D()
            tr_translate.translate(
                -WIDGET_SIZE / 2,
                -(WIDGET_SIZE / 2) * np.cos(2 * np.pi * i / self.num_image),
                -(WIDGET_SIZE / 2) * np.sin(2 * np.pi * i / self.num_image),
            )
            # self.imgs[i].applyTransform(, local=True)
            self.imgs[i].applyTransform(tr_translate * tr_rotate * tr_scale, local=True)
            self.imgs[i].setVisible(self.visible)
            self.plot_widget.addItem(self.imgs[i])

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
                for n_img in range(self.num_image):
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

                for n_img in range(self.num_image):
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
        if SWITCH_SPIKE[0] == 1:
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

        self.L = 5000
        self.buffer = 1500
        self.latent = np.zeros(
            (self.L + self.buffer, 4)
        )  # Initialize firing rates for each neuron
        self.decay_factor = LATENT_DECAY_FACTOR[0]
        self.x = x_index
        self.y = y_index
        self.traces = dict()
        self.heads = dict()
        if isinstance(color, LinearSegmentedColormap):
            self.colormap = color
            self.color = color(np.linspace(0, 1, self.L))
        else:
            self.colormap = None
            self.color = np.repeat(np.array(color)[np.newaxis, :] / 255, self.L, axis=0)

        for i in range(1, self.L):
            self.color[i][-1] = self.color[i - 1][-1] * self.decay_factor
        # self.L = np.where(np.array([x[-1] for x in self.color]) < 0.1)[0][0]
        self.color = self.color[: self.L]
        self.color = self.color[::-1]
        self.data = np.zeros((4, self.L + self.buffer))

        self.z = [x for x in range(4) if x not in [self.x]]
        for i in range(4):
            self.traces[i] = gl.GLLinePlotItem(
                pos=np.zeros((self.L, 3)),
                color=self.color,
                width=5,
                antialias=False,
            )
            self.traces[i].setVisible(visible)
            self.plot_widget.addItem(self.traces[i])
        self.frame = 0
        self.prev_count = 0
        self.count = 0
        self.traj_pos_bias = 0
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
            for i in range(0, 3, 2):
                pts = np.vstack(
                    [
                        np.ones_like(self.data[0, slice_window]) * VIS_DEPTH,
                        (self.data[i, slice_window] * VIS_RADIUS + self.traj_pos_bias)
                        * ASPECT_RATIO,
                        self.data[i + 1, slice_window] * VIS_RADIUS,
                    ]
                ).transpose()
                self.traces[i].setData(
                    pos=pts,
                )
        self.frame += 1


class InferredLatentVisualizer:
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

        self.L = 1500
        self.buffer = 1500
        self.latent = np.zeros(
            (self.L + self.buffer, 4)
        )  # Initialize firing rates for each neuron
        self.decay_factor = INFERRED_DECAY_FACTOR[0]
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
        self.data = np.zeros((4, self.L + self.buffer))

        self.z = [x for x in range(4) if x not in [self.x]]
        for i in range(4):
            self.traces[i] = gl.GLLinePlotItem(
                pos=np.zeros((self.L, 3)),
                color=self.color,
                width=5,
                antialias=False,
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
            self.data[:, self.frame] = INFERRED[0]
        if self.frame % 1 == 0 and self.frame > 0 and self.visible:
            slice_window = slice(np.fmax(0, self.frame - self.L), self.frame)
            for i in range(0, 3, 2):
                pts = np.vstack(
                    [
                        np.ones_like(self.data[0, slice_window]) * VIS_DEPTH,
                        (self.data[i, slice_window] * VIS_RADIUS + self.bias)
                        * ASPECT_RATIO,
                        self.data[i + 1, slice_window] * VIS_RADIUS,
                    ]
                ).transpose()
                self.traces[i].setData(
                    pos=pts,
                )
            # print(f"Frame: {self.count}, Count: {packet_count}")
        self.frame += 1


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
target_location[:, 1:] *= 0.7
target_location[:, 1] *= ASPECT_RATIO

app = QApplication([])
# -----------------------------Visualization on the floor-----------------------------
parent_widget = gl.GLViewWidget()
parent_widget.setGeometry(0, 0, 1920, 1200)
layout = QGridLayout()
parent_widget.setLayout(layout)
monitor = QDesktopWidget().screenGeometry(0)
parent_widget.move(monitor.left(), monitor.top())


floor_widget = gl.GLViewWidget()
floor_widget.setGeometry(0, 0, 1024, 1024)
floor_widget.opts["center"] = QtGui.QVector3D(-30, 0, 0)
floor_widget.opts["distance"] = 55
floor_widget.opts["fov"] = 90
floor_widget.opts["elevation"] = 0
floor_widget.opts["azimuth"] = 0

layout.addWidget(floor_widget, 1, 0)
parent_widget.show()


vis_spike = SpikeBall3DVisualizer(
    target_location=target_location, visible=SWITCH_SPIKE[0], widget=floor_widget
)
vis_true_latent = TrueLatentVisualizer(
    0,
    1,
    color=GRADIENT_CMAP,
    visible=SWITCH_TRUE_LATENT[0],
    widget=floor_widget,
)
vis_inferred_latent = InferredLatentVisualizer(
    0,
    1,
    color=INFERRED_LATENT_COLOR,
    visible=SWITCH_INFERRED_LATENT[0],
    widget=floor_widget,
)
vis_raster = RasterCircularVisualizer(visible=SWITCH_RASTER[0], widget=floor_widget)
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

    def inferred_osc_handler(self, address, *args):
        global INFERRED, packet_count

        INFERRED[0] = np.array(args)
        # packet_count += 1
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
        vis_spike.visible = args[0]
        vis_spike.indicators.setVisible(args[0])

    def max_switch_wall_true_latent(self, address, *args):
        vis_wall_true_latent.visible = args[0]
        for trace in vis_wall_true_latent.traces.items():
            trace[1].setVisible(args[0])
        if args[0]:
            # vis_wall_inferred_latent.bias = GRID_SIZE_HEIGHT / 2
            vis_wall_inferred_latent.bias = 0
        else:
            vis_wall_inferred_latent.bias = 0

    def max_switch_ceiling_true_latent(self, address, *args):
        vis_true_latent.visible = args[0]
        for trace in vis_true_latent.traces.items():
            trace[1].setVisible(args[0])
        if args[0]:
            # vis_ceiling_inferred_latent.bias = GRID_SIZE_HEIGHT / 2
            vis_inferred_latent.bias = 0
        else:
            vis_inferred_latent.bias = 0

    def max_switch_wall_inferred_latent(self, address, *args):
        vis_wall_inferred_latent.visible = args[0]
        for trace in vis_wall_inferred_latent.traces.items():
            trace[1].setVisible(args[0])
        if args[0]:
            vis_wall_true_latent.traj_pos_bias = 0
        else:
            vis_wall_true_latent.traj_pos_bias = 0

    def max_switch_ceiling_inferred_latent(self, address, *args):
        vis_inferred_latent.visible = args[0]
        for trace in vis_inferred_latent.traces.items():
            trace[1].setVisible(args[0])
        if args[0]:
            vis_true_latent.bias = 0
        else:
            vis_true_latent.bias = 0

    def max_switch_ceiling_raster(self, address, *args):
        vis_raster.visible = args[0]
        vis_raster.img.setVisible(args[0])

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
        vis_raster.update_lut(color)
        vis_spike.update_color(color)
        # vis_wall_true_latent.update_lut(color)
        # vis_ceiling_true_latent.update_lut(color)


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
    server_inferred_latent = AsyncIOOSCUDPServer(
        (LOCAL_SERVER, INFERRED_LATENT_PORT),
        dispatcher_python,
        asyncio.get_event_loop(),
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

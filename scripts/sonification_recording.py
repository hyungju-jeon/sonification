# %%
import time
import os
import random
import subprocess
import sys
from os.path import dirname
from os.path import join as pjoin
import pyqtgraph.opengl as gl

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
import scipy.io as sio
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsView,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QGroupBox,
)
from pyqtgraph.Qt import QtCore, QtGui


# %%
# ---------------------------------------------------------------- #
# Common Parameters
random.seed(0)
dt = 1e-3  # 1ms for dynamic system update
frame_time = 0
data_folder = os.path.join(dirname(__file__), "../data")

mat_contents = sio.loadmat(os.path.join(data_folder, "array_5.mat"))
spk_times = mat_contents["spk_times"]
num_neurons = spk_times.shape[0]
orientation_condition = mat_contents["ori"][0]
trial_length = 2560

sequence_csv = os.path.join(data_folder, "sequence_GrafV1_op1.csv")
sequence_data = np.genfromtxt(sequence_csv, delimiter=",")
orientation_sequence = sequence_data[0, :]
trial_sequence = sequence_data[1, :]

latent_data = np.load(os.path.join(data_folder, "latent_trajectories.npy"))

tuning_curve = np.genfromtxt(
    os.path.join(data_folder, "tuning_curves_5.csv"), delimiter=","
)


def create_spike_train_from_sequence(orientation, trial):
    orientation_index = np.where(orientation_condition == orientation)[0]
    trial_index = orientation_index[trial]

    spike_events = spk_times[:, trial_index]
    spike_train = np.zeros((num_neurons, 2560))
    for i in range(num_neurons):
        idx = np.round(spike_events[i]).astype(int)
        idx[idx >= 2560] = 2559
        spike_train[i, idx] = 1

    return spike_train


smooth_tuning_curve = np.zeros((360, num_neurons))
for i in range(num_neurons):
    # interpolate the tuning curve
    smooth_tuning_curve[:, i] = np.interp(
        np.linspace(0, 360, 360), np.arange(0, 360, 5), tuning_curve[:, i]
    )


def generate_target_location(tuning_curve):
    # Tuning curve is matrix by [orientation, neuron]
    # Find the peak of the tuning curve and map to find preferred orientation assign target location for each neuron along unit circle based on preferred orientation
    target_location = np.zeros((num_neurons, 2))
    for i in range(num_neurons):
        target_orientation = np.argmax(tuning_curve[:, i])
        target_location[i, :] = [
            10 * np.cos(target_orientation / tuning_curve.shape[0] * 2 * np.pi),
            10 * np.sin(target_orientation / tuning_curve.shape[0] * 2 * np.pi),
        ]
    return target_location


target_location = generate_target_location(smooth_tuning_curve)

packet_count = 0
SPIKES = [np.zeros(num_neurons)]
DISC_RADIUS_INC = [10]
DECAY_FACTOR = [0.1]
SEQUENCE_TRIGGER = [0]
LATENT = [np.zeros(8)]

# %% Trajectory
# Concatenate along first dimension of latent_data to create (84x127) X 8 array
total_latent_data = np.concatenate(latent_data, axis=0)
num_latent = total_latent_data.shape[1]
# smooth trajectory
smooth_latent = np.zeros((2560 * 84, num_latent))
for i in range(num_latent):
    # interpolate the tuning curve
    smooth_latent[:, i] = np.interp(
        np.linspace(0, total_latent_data.shape[0] * 20, 215040),
        np.arange(0, total_latent_data.shape[0] * 20, 20),
        total_latent_data[:, i],
    )


# %%
class SpikeDisc2DVisualizer:

    def __init__(self):
        self.app = QApplication([])
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground((0, 0, 0, 255))
        self.plot_widget.setYRange(-15, 15)
        self.plot_widget.setXRange(-15 * 1.6, 15 * 1.6)
        self.plot_widget.setGeometry(0, 0, 1920, 1200)
        # remove axis
        self.plot_widget.hideAxis("bottom")
        self.plot_widget.hideAxis("left")
        self.plot_widget.show()
        self.plot_widget.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)

        self.centroid_positions = np.array(
            [
                [np.random.uniform(-16, 16), np.random.uniform(-10, 10)]
                for _ in range(num_neurons)
            ]
        )
        self.target_positions = target_location

        self.draw_centroids()

        # Add function that will be called when the timer times out
        self.frame = 0
        self.prev_count = 0
        self.count = 0

    def animation(self):
        self.prev_count = 0
        self.count = 0

        # QApplication.instance().exec_()

    def draw_centroids(self):
        """
        Draws the centroid indicators in the animation.

        Returns:
            list: The centroid indicator items in the animation.
        """
        indicators = pg.ScatterPlotItem()
        indicators.setData(
            pos=self.centroid_positions,
            size=0,
            symbol="o",
            pen=pg.mkPen(width=0, color=(51, 255, 51, 0)),
            brush=(51, 255, 51, 0),
        )
        self.plot_widget.addItem(indicators)
        self.indicators = indicators
        self.size = np.zeros(num_neurons)
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
        if self.frame % 10 == 0:
            self.save_frame()
        self.frame += 1
        print(f"Current Frame: {self.frame}")

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
            self.centroid_positions[index] -= 0.1 * self.velocity[index]

        # # Add the zapping effect element
        # zap_effect = pg.ScatterPlotItem()
        # zap_effect.setData(
        #     pos=self.centroid_positions[index],
        #     size=self.size[index],
        #     symbol="o",
        #     pen=pg.mkPen(width=0, color=(230, 230, 230, 0)),
        #     brush=(230, 230, 230, 0),
        # )
        # self.plot_widget.addItem(zap_effect)

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
        self.centroid_positions += 0.05 * self.velocity

    def save_frame(self):
        """
        Saves the current frame of the animation to a file.
        """
        exporter = pg.exporters.ImageExporter(self.plot_widget.scene())
        # exporter.parameters()["width"] = 1048
        exporter.export(
            "/Users/hyungju/Desktop/hyungju/Project/sonification/results/animation/recording/spike/frame"
            + f"{self.frame:010}"
            + ".png"
        )


class SpikeRaster2DVisualizer:

    def __init__(self):
        self.app = QApplication([])
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground((40, 40, 40, 255))
        self.plot_widget.setXRange(0, 1000)
        self.plot_widget.setYRange(0, num_neurons)
        self.plot_widget.setGeometry(0, 0, 1920, 500)
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
        # self.decay_factor = DECAY_FACTOR[0]
        self.decay_factor = 0
        self.num_neurons = num_neurons

        self.img = pg.ImageItem()
        self.img.setLevels((0, 5))
        scale_factor = [1920 / self.L, 1200 / self.num_neurons]
        self.count = 0
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

        if (self.count < 1000) & (self.count % 10 == 0):
            rolled_rate = self.firing_rates[0:1000, :]
            rolled_rate = np.roll(rolled_rate, 1000 - self.count, axis=0)
            self.img.setImage(rolled_rate, autoLevels=False)
            self.save_frame()

        if (self.count >= 1000) & (self.count % 10 == 0):
            self.img.setImage(
                self.firing_rates[np.fmax(0, self.frame - self.L) : self.frame, :],
                autoLevels=False,
            )
            self.save_frame()
            print(f"Frame: {self.count}, Count: {packet_count}")
        self.frame += 1

    def save_frame(self):
        """
        Saves the current frame of the animation to a file.
        """
        exporter = pg.exporters.ImageExporter(self.plot_widget.scene())
        # exporter.parameters()["width"] = 1048
        exporter.export(
            "/Users/hyungju/Desktop/hyungju/Project/sonification/results/animation/recording/h_raster/frame"
            + f"{self.count:06}"
            + ".png"
        )
        # self.bufferImg = exporter.export(toBytes=True)


class SpikeRaster2DVerticalVisualizer:

    def __init__(self):
        self.app = QApplication([])
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground((40, 40, 40, 255))
        self.plot_widget.setYRange(0, 1000)
        self.plot_widget.setXRange(0, num_neurons)
        self.plot_widget.setGeometry(0, 0, 1920, 500)
        # remove axis
        self.plot_widget.hideAxis("bottom")
        self.plot_widget.hideAxis("left")
        self.plot_widget.show()
        # self.plot_widget.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)

        self.L = 1000
        self.buffer = 1000
        self.firing_rates = np.zeros(
            (self.L + self.buffer, num_neurons)
        )  # Initialize firing rates for each neuron
        # self.decay_factor = DECAY_FACTOR[0]
        self.decay_factor = 0.5
        self.num_neurons = num_neurons

        self.img = pg.ImageItem()
        self.img.setLevels((0, 5))
        scale_factor = [1920 / self.L, 1200 / self.num_neurons]
        self.count = 0
        self.plot_widget.addItem(self.img)

        tr = QtGui.QTransform()  # prepare ImageItem transformation:
        tr.rotate(90)  # rotate 90 degrees

        # assign transform

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

        if (self.count < 1000) & (self.count % 10 == 0):
            rolled_rate = self.firing_rates[0:1000, :]
            rolled_rate = np.roll(rolled_rate, 1000 - self.count, axis=0)
            self.img.setImage(rolled_rate.T, autoLevels=False)
            self.save_frame()

        if (self.count >= 1000) & (self.count % 10 == 0):
            self.img.setImage(
                self.firing_rates[np.fmax(0, self.frame - self.L) : self.frame, :].T,
                autoLevels=False,
            )
            self.save_frame()
            print(f"Frame: {self.count}, Count: {packet_count}")
        self.frame += 1

    def save_frame(self):
        """
        Saves the current frame of the animation to a file.
        """
        exporter = pg.exporters.ImageExporter(self.plot_widget.scene())
        # exporter.parameters()["width"] = 1048
        exporter.export(
            "/Users/hyungju/Desktop/hyungju/Project/sonification/results/animation/recording/v_raster/frame"
            + f"{self.count:06}"
            + ".png"
        )
        # self.bufferImg = exporter.export(toBytes=True)


class LatentOrbitVisualizer:
    def __init__(self, x_index, y_index):

        self.app = QApplication([])
        self.plot_widget = gl.GLViewWidget()
        self.plot_widget.setGeometry(0, 110, 960, 600)
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
        self.buffer = 0
        self.latent = np.zeros(
            (self.L + self.buffer, 8)
        )  # Initialize firing rates for each neuron
        self.decay_factor = DECAY_FACTOR[0]
        self.x = x_index
        self.y = y_index
        self.traces = dict()

        color = np.repeat(
            np.array([51, 255, 51, 255])[np.newaxis, :] / 255, self.L, axis=0
        )
        for i in range(1, self.L - 2):
            color[self.L - i - 1][-1] = color[self.L - i][-1] * 0.99
        self.L = 1000 - np.where(np.array([x[-1] for x in color]) < 0.1)[0][-1]
        self.data = np.zeros((8, self.L + self.buffer))
        color = color[-self.L :]
        self.z = [x for x in range(8) if x not in [self.x, self.y]]

        for i in range(6):
            self.traces[i] = gl.GLLinePlotItem(
                pos=np.zeros((self.L, 3)),
                color=color,
                width=10,
                antialias=True,
            )
            self.plot_widget.addItem(self.traces[i])
        self.frame = 0
        self.prev_count = 0
        self.count = 0

    def animation(self):
        self.prev_count = 0
        self.count = 0
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QApplication.instance().exec_()

    def update(self):
        self.count += 1
        self.data = np.roll(self.data, -1, axis=1)
        self.data[:, -1] = LATENT[0]

        if self.frame % 10 == 0:
            for i in range(6):
                pts = np.vstack(
                    [
                        self.data[i, :] * 15,
                        self.data[i + 1, :] * 15,
                        self.data[i + 2, :] * 10,
                    ]
                ).transpose()
                self.traces[i].setData(
                    pos=pts,
                )
            self.save_frame()
            print(f"Frame: {self.count}, Count: {packet_count}")
        self.frame += 1

    def save_frame(self):
        """
        Saves the current frame of the animation to a file.
        """
        fname = f"/Users/hyungju/Desktop/hyungju/Project/sonification/results/animation/recording/latent_4/frame{self.count:06}.png"
        self.plot_widget.grabFramebuffer().save(fname)


# %%
visualizer_vertical_raster = SpikeRaster2DVerticalVisualizer()
# visualizer_raster = SpikeRaster2DVisualizer()
# visualizer = SpikeDisc2DVisualizer()
# visualizer_latent = LatentOrbitVisualizer(2, 4)
iter = 0
smooth_latent = smooth_latent - np.mean(smooth_latent, axis=0)
for orientation, trial in zip(orientation_sequence, trial_sequence):
    spike_train = create_spike_train_from_sequence(int(orientation), int(trial) - 1)
    current_latent = smooth_latent[iter * 2560 : (iter + 1) * 2560, :] / 20
    if iter < 36:
        for i in range(trial_length):
            SPIKES[0] = spike_train[:, i]

            # visualizer_raster.update()
            visualizer_vertical_raster.update()
    # if iter >= 36:
    #     if iter == 40:
    #         SEQUENCE_TRIGGER[0] = 1
    #     for i in range(trial_length):
    #         SPIKES[0] = spike_train[:, i]
    #         # visualizer.update()
    # if iter >= 48:
    #     for i in range(trial_length):
    #         LATENT[0] = current_latent[i, :]
    #         visualizer_latent.update()
    iter += 1

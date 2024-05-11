# %%
import os
import random
import subprocess
import signal
import sys
import time
from os.path import dirname
from os.path import join as pjoin

import multiprocessing
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from pythonosc.udp_client import SimpleUDPClient

from utils.ndlib.vislib import *

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
DECAY_FACTOR = [0.9]
SEQUENCE_TRIGGER = [0]

# %% Trajectory
# Concatenate along first dimension of latent_data to create (84x127) X 8 array
total_latent_data = np.concatenate(latent_data, axis=0)
num_latent = total_latent_data.shape[1]
# smooth trajectory
smooth_latent = np.zeros((total_latent_data.shape[0] * 20, num_latent))
for i in range(num_latent):
    # interpolate the tuning curve
    smooth_latent[:, i] = np.interp(
        np.arange(0, total_latent_data.shape[0] * 20, 1),
        np.arange(0, total_latent_data.shape[0] * 20, 20),
        total_latent_data[:, i],
    )


two_trials = slice(0, 254)
first_stimulus = slice(0, 127)
second_stimulus = slice(254, 381)

from sklearn.decomposition import PCA

# Run PCA on total_latent_data
pca = PCA(n_components=8)
pca.fit(total_latent_data)
pca_latent_data = pca.transform(total_latent_data - np.mean(total_latent_data, axis=0))

# Run SVD on total_latent_data
U, S, V = np.linalg.svd(total_latent_data)
svd_latent_data = U[:, :8]

# %%
orientation_series = np.repeat(orientation_sequence, 64) % 180
stim_indicator = np.repeat(np.repeat([1, 0], 64)[:127], 84)
stim_latent = np.abs(total_latent_data[stim_indicator == 1, :])
mean_stim_latent = np.zeros_like(stim_latent)
for i in range(84):
    mean_stim_latent[i * 64 : (i + 1) * 64, :] = np.mean(
        np.abs(stim_latent[i * 64 : (i + 1) * 64, :]), axis=0
    )

# normalize per column
mean_stim_latent = (stim_latent - np.mean(stim_latent, axis=0)) / np.std(
    stim_latent, axis=0
)

pca = PCA(n_components=8)
pca.fit(mean_stim_latent)
pca_latent_data = pca.transform(smooth_latent)

# %% Plot
plot_data = pca_latent_data - np.mean(pca_latent_data, axis=0)
# plot_data /= 10

subprocesses = None


def worker(file):
    global subprocesses
    subprocesses = subprocess.Popen(["python", file])


prcesses = multiprocessing.Process(
    target=worker("./scripts/sonification_visualize_module.py")
)
# prcesses.start()

SERVER_IP = "127.0.0.1"
LATENT_PORT = 1113

OSCsender = SimpleUDPClient(SERVER_IP, LATENT_PORT)
for i in range(plot_data.shape[0]):
    try:
        OSCsender.send_message("/trajectory", plot_data[i, :].tolist())
        time.sleep(0.001)
        i += 1
    except KeyboardInterrupt:
        print("Terminating!")
        subprocesses.kill()
        os.kill(os.getpid(), signal.SIGKILL)
subprocesses.kill()

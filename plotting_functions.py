# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:20:18 2024

@author: abel_
"""

import glob
import os, sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir) 

import numpy as np
from scipy.integrate import solve_ivp
import scipy
import math

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from matplotlib import transforms
import matplotlib.patches as mpatches
import matplotlib.colors as mplcolors
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
from pylab import rcParams
from sklearn.decomposition import PCA

from tqdm import tqdm

def generate_data(i):
    theta = np.linspace(0, 2 * np.pi, 100)
    x = np.sin(theta + i * np.pi / 10)
    y = np.cos(theta + i * np.pi / 10)
    z = theta
    return x, y, z

def coupled_oscillators(t, y, k, sigma=0.):
    """
    Define the system of first-order ODEs for coupled oscillators.

    Parameters:
    - t: Time
    - y: Array of positions and velocities of oscillators
    - masses: List of masses of oscillators
    - k: Spring constant

    Returns:
    - dydt: Array of derivatives of positions and velocities
    """

    num_oscillators = len(k)
    dydt = np.zeros_like(y)

    # Extract positions and velocities from y
    positions = y[:num_oscillators]
    velocities = y[num_oscillators:]

    # Calculate derivatives of positions and velocities
    eps = np.random.normal(0, sigma, num_oscillators)
    for i in range(num_oscillators):

        acceleration = k[i] *  - positions[i] + eps[i]
        dydt[i] = velocities[i]
        dydt[num_oscillators + i] = acceleration

    return dydt

# Parameters
k = [1.0, 2.]  # Spring constants

def plot_trajectory_3d(trajs, lims=[], ax=None,
                                   cam_view = (20., -35., 0.),
                                   cmap = cmx.get_cmap("gist_yarg"), 
                                   facecolor='black'):
    """
    

    Parameters
    ----------
    traj : numpy.array
        Trajectories of shape(num_of_trajs, T, hidden_dim).
    lims : TYPE, optional
        Limits of the axes. The default is [].
    ax : TYPE, optional
        Axis of subplot of a figure. The default is None.
    view : TYPE, optional
        View of camera. The default is (20., -35., 0.).
    cmap : TYPE, optional
        Colormap for different trajectories. The default is cmx.get_cmap("gist_yarg").

    Returns
    -------
    None.

    """
    
    num_of_trajs, T, hidden_dim = trajs.shape
    elev, azim, roll = cam_view
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", facecolor=facecolor)
    norm = mplcolors.Normalize(vmin=-.5,vmax=.5)
    norm = norm(np.linspace(-.5, .5, num=T, endpoint=True))

    for trial_i in range(num_of_trajs):
        ax.plot(trajs[trial_i,:,0], trajs[trial_i,:,1], zs=trajs[trial_i,:,2],
                zdir='z', color=cmap(norm[trial_i]))
        
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.axis('off')

    if lims!=[]:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        ax.set_zlim(lims[2])
        
def load_data():   
    # Initial conditions
    x0 = [0.5, 1]  # Initial positions
    v0 = [.5, 0]  # Initial velocities
    y0 = np.concatenate([x0, v0])
    
    # Time span
    T = 115
    t_span = (0, T)  # Time interval for integration
    sol = solve_ivp(coupled_oscillators, t_span, y0, args=(k,), t_eval=np.linspace(0, T, 10000))
    trajs = np.expand_dims(sol.y.T, axis=0)
    
    tuned_responses = np.load("tuned_responses.npy")
    z_values=tuned_responses.shape[1]
    
    pref_idx = np.load("pref_oris_idx.npy")



#ANIMATION
def make_trajectory_video(traj, shift, seq_i, lims,
                          fs=(5,5), dpi=400, elev=30, azim=0., roll=0, move_camera_along=None,
                          folder='trajectory_videos'):
    
    fig = plt.figure(figsize=fs, dpi=dpi)
    fig.set_size_inches(fs[0], fs[0], True)
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    ax.axis('off')
    ax.view_init(elev=elev, azim=azim, roll=0)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    traj_ext = np.ones((traj.shape[0]+shift+1, traj.shape[1])) * traj[0]
    traj_ext[shift+1:, :] = traj
    
    # lims = [[-1.1,1.1], [-1.1,1.1], [-1.1,1.1]]
    # ax.set_xlim([traj[:,0].min(), traj[:,0].max()])
    # ax.set_ylim([traj[:,1].min(), traj[:,1].max()])
    # ax.set_zlim([traj[:,2].min(), traj[:,2].max()])
    
    ax.set_xlim([lims[0][0], lims[0][1]])
    ax.set_ylim([lims[1][0], lims[1][1]])
    ax.set_zlim([lims[2][0], lims[2][1]])

    # Initialize the plot
    line, = ax.plot([], [], [], lw=2, c='#33FF33')

    def update(frame):
        # x, y, z = generate_data(frame)
        
        x = traj_ext[frame:shift+frame,0]
        y = traj_ext[frame:shift+frame,1]
        z = traj_ext[frame:shift+frame,2]
        
        line.set_data(x, y)
        line.set_3d_properties(z)
        
        # ax.view_init(azim=frame+azim)
        
        # if move_camera_along=='elev':  
            # ax.view_init(globals()[move_camera_along]=frame)
        if move_camera_along=='elev':  
            ax.view_init(elev=frame)

        elif move_camera_along=='azim':  
            ax.view_init(azim=frame)
        
        return line,

    ani = FuncAnimation(fig, update, frames=range(0, traj.shape[0]+1, 1), blit=True)

    ani.save(folder+f"/traj_{seq_i}.mp4", fps=50)
    
    
def make_trajectory_video_pyrecorder(traj, shift, seq_i, lims, fs=(5,5), dpi=400, fps=10, codec='mp4v',
                                     elev=30, azim=0., roll=0,
                                     move_camera_along=None):

    converter = Matplotlib(dpi=dpi)
    writer = Video(f"trajectory_videos/traj_{seq_i}.mp4", fps=fps, codec=codec)
    rec = Recorder(writer, converter=converter)
    
    traj_ext = np.zeros((traj.shape[0]+shift, traj.shape[1]))
        
    for t in range(traj.shape[0]):

        fig = plt.figure(figsize=fs, dpi=dpi)
        fig.set_size_inches(15, 15, True)
        ax = fig.add_subplot(111, projection='3d', facecolor='black')
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

        line, = ax.plot([], [], [], lw=2, c='#33FF33')

        ax.axis('off')
        ax.view_init(elev=elev, azim=azim, roll=0)
        
        # lims = [[-1.1,1.1], [-1.1,1.1], [-1.1,1.1]]
        ax.set_xlim([lims[0][0], lims[0][1]])
        ax.set_ylim([lims[1][0], lims[1][1]])
        ax.set_zlim([lims[2][0], lims[2][1]])
        
        x = traj_ext[t:shift+t,0]
        y = traj_ext[t:shift+t,1]
        z = traj_ext[t:shift+t,2]
        
        line.set_data(x, y)
        line.set_3d_properties(z)
        
        rec.record(fig=fig)

    rec.close()
    
# binned_spike_array = np.load("binned_spike_array.npy")
# binned_spike_tensor = torch.tensor(binned_spike_array, dtype=default_dtype)
# sonified_binned_spike_array = np.zeros((sequence_GrafV1_op1.shape[1], binned_spike_array.shape[1], nNeurons))
# for i, orientation in enumerate(sequence_GrafV1_op1[0]):
#     trial_i = int(sequence_GrafV1_op1[1][i]-1)
#     sonified_binned_spike_array[i] = binned_spike_array[np.where(ori==orientation)[1]][trial_i]
# sonified_binned_spike_tensor = torch.tensor(sonified_binned_spike_array, dtype=default_dtype)
# loss, z_s, stats = ssm(binned_spike_tensor, n_samples)
# all_m_f_array = stats['m_f'].detach().numpy()
# X = all_m_f_array[:,64:,:].reshape((-1, 8))
# X = all_m_f_array[:,:,:].reshape((-1, 8))
# all_m_f_array_transformed = transformed_data.reshape((3600,-1,3))
# traj = all_m_f_array_transformed[0,...]
# plot_trajectory_3d(all_m_f_array_transformed[:1,...])
# make_trajectory_video(trajs, shift=10, dpi=10)
    
def make_tunings_video(trajs, shift, elev=30, azim=0., roll=0, move_camera_along=None):
    fig = plt.figure()
    fig.set_size_inches(15, 15, True)
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    ax.axis('off')
    ax.set_xlim([0, 360])
    ax.set_ylim([0, 72])
    ax.set_zlim([0, 1])
    x = np.arange(0, 360, 5)
    
    def update2(frame):
        lines = [ax.plot([], [], [], c='w')[0] for _ in range(z_values)]
        for i, line in enumerate(lines):
            y = np.ones(72)*i
            z = scipy.ndimage.gaussian_filter(tuned_responses[:,pref_idx][:, i]/np.max(tuned_responses[:,pref_idx][:, i]), sigma=2)
            line.set_data(x, y)
            line.set_3d_properties(z)
        
        ax.view_init(elev=frame, azim=90, roll=0)
        
        return lines, 
        
########Super jumpy:
# def make_gabor_video(T, fps=100, orientation = 0, fs=(10,10), dpi=400):
    
#     fig = plt.figure(figsize=fs, dpi=dpi)
#     ax = fig.add_subplot(111, facecolor='black')
#     ax.axis('off')


#     size = 990    # Size of the grating
#     spatial_frequency = 2.2  # Spatial frequency of the grating (cycles/pixel)
#       # Orientation of the grating (radians)
    
#     # Create a grid of coordinates
#     x = np.linspace(-1, 1, size)
#     y = np.linspace(-1, 1, size)
#     X, Y = np.meshgrid(x, y)

#     grating = np.sin(2 * np.pi  * spatial_frequency * (np.cos(orientation) * X + np.sin(orientation) * Y))
#     im = ax.imshow(grating, cmap='gist_gray_r')     # Initialize the plot

#     def update(frame):
#         grating = np.sin(frame*np.pi / 500 + 2 * np.pi * spatial_frequency * (np.cos(orientation) * X + np.sin(orientation) * Y))
#         im.set_array(grating)
#         return im,

#     ani = FuncAnimation(fig, update, frames=range(1, T, 1),
#                                frames = T * fps,
#                                interval = 1000 / fps, # in ms, 
#                                blit=True)

#     ani.save('gabor_animation.mp4', fps=100)



from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video
from pyrecorder.converters.matplotlib import Matplotlib
from matplotlib.colors import ListedColormap

fs = (5,5)
dpi = 100

def make_gabor_video(orientations, T=256, spatial_frequency=9/2, drift_frequency=6.25, 
                     size=990, fps=100, fs=(10,10), dpi=100, codec='mp4v', seq_i=0,
                     folder='gratings_w'):
    
    # Size of the grating
    # Spatial frequency of the grating (cycles/pixel)
    # Orientation of the grating (radians)
    
    # Create a custom colormap using ListedColormap
    colors = [(i/255, i/255, i/255) for i in range(40, 256)]
    custom_cmap = ListedColormap(colors)
    # Create a grid of coordinates
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((size,size))
    Z[np.where(X**2+Y**2<1)]=1
     
    # initialize the converter which is creates an image when `record()` is called
    converter = Matplotlib(dpi=dpi)
    writer = Video(folder+f"/sinusoid_sonified_{seq_i}_{orientations[0]}.mp4", fps=fps, codec=codec)
    rec = Recorder(writer, converter=converter)
    
    for orientation in tqdm(orientations):
        radians = -math.radians(orientation)

        for t in range(T):
        
            fig = plt.figure(figsize=fs, dpi=dpi)
            ax = fig.add_subplot(111, facecolor='black')
            ax.axis('off')
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
            # ax.set_rasterized = True
            
            if t<128:
                grating = np.sin(-drift_frequency*t*2*np.pi/100 + 2 * np.pi * spatial_frequency * (np.cos(radians) * X + np.sin(radians) * Y))
                grating = np.multiply(grating, Z)
                ax.matshow(grating, cmap=custom_cmap, interpolation='nearest')     

            else:
                grating = np.ones((1,1))*148
                ax.matshow(grating, cmap=custom_cmap, vmin=40, vmax=256)     

            ax.set_xlim=(-1, 1)
            ax.set_ylim=(-1, 1)
            rec.record(fig=fig)
    
    rec.close()
    
def plot_2d(latents_pca, fs=(10,10), dpi=400, folder='plots'):
    norm = mplcolors.Normalize(vmin=-0,vmax=355)

    cmap = plt.get_cmap('hsv')
    fig = plt.figure(figsize=fs, dpi=dpi)
    fig.set_size_inches(fs[0], fs[0], True)
    ax = fig.add_subplot(111, facecolor='black')
    ax.axis('off')
    for trial_i in range(latents_pca.shape[0]):
        if sequence_GrafV1_op1[0][trial_i] == 0. or sequence_GrafV1_op1[0][trial_i] == 90.:
            ax.plot(latents_pca[trial_i,:,2], latents_pca[trial_i,:,3], color=cmap(norm(sequence_GrafV1_op1[0][trial_i])))
    plt.savefig(folder+'/090_latents_pc34.pdf', bbox_inches="tight")
    
    
    
def pca_transform(X, number_of_pcs):
    X_standardized = (X - np.mean(X, axis=0))
    covariance_matrix = np.cov(X_standardized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    top_eigenvectors = eigenvectors[:, :number_of_pcs]
    transformed_data = np.dot(X_standardized, top_eigenvectors)
    
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    return top_eigenvectors, transformed_data
    
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

if __name__ == '__main__':
    makedirs('plots')
    makedirs('trajectory_videos')

    latents_full = np.load("latent_trajectories_full.npy")
    latents_full = (latents_full - np.mean(latents_full, axis=0))
    n_latents = latents_full.shape[-1]

    n_pca = 4
    top_eigenvectors, all_latents_pca = pca_transform(latents_full.reshape((-1,n_latents)), n_pca)
    
    latents = np.load("latent_trajectories.npy")
    latents = (latents - np.mean(latents, axis=0))

    transformed_latents = np.dot(latents.reshape((-1,n_latents)), top_eigenvectors)
    latents_pca = transformed_latents.reshape((latents.shape[0],-1,n_pca))
    
    # pca = PCA(n_components=3)
    # pca.fit(latents_full.reshape((-1,8)))
    # latents_pca = pca.transform(latents.reshape((-1,8))).reshape((latents.shape[0],-1,3))
    
    eps = 20
    lims = [[], [], []]
    lims[0] = [latents_pca[:,0].min()-eps, latents_pca[:,0].max()+eps]
    lims[1] = [latents_pca[:,1].min()-eps, latents_pca[:,1].max()+eps]
    lims[2] = [latents_pca[:,2].min()-eps, latents_pca[:,2].max()+eps]
    
    sequence_GrafV1_op1 = np.loadtxt("sequence_GrafV1_op1.csv", delimiter=',')
    
    # plot_2d(latents_pca)

    for seq_i, seq in enumerate(sequence_GrafV1_op1[0]):
        print(seq_i)
        # if seq_i>17:
        #     make_gabor_video(orientations=[seq], T=256, spatial_frequency=9/2, drift_frequency=6.25, 
        #                          size=1280, fs=(10,10), dpi=100, seq_i=seq_i)
        
        # make_trajectory_video_pyrecorder(latents_pca[seq_i], lims=lims, seq_i=seq_i, shift=20)
        azim=20 #seq_i*128
        make_trajectory_video(latents_pca[seq_i], lims=lims, seq_i=seq_i, shift=20,
                              elev=0, azim=azim, roll=0)

        
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.pylab import f
from matplotlib import cm


def plot_latents(n_latents, n_samples, blues, z_c, z_gt, epoch):
    
    fig, axs = plt.subplots(1, n_latents, figsize=(20, 5))
    [
        axs[i].plot(z_c[j, 0, :, i], color=blues(j), alpha=0.5)
        for i in range(n_latents)
        for j in range(n_samples)
    ]
    [
        axs[i].plot(z_gt[0, :, i], color="black", alpha=0.7, label="true")
        for i in range(n_latents)
    ]
    [axs[i].set_box_aspect(1.0) for i in range(n_latents)]
    [axs[i].set_title(f"dim {i}") for i in range(n_latents)]

    fig.suptitle(f'epoch {epoch}')

    plt.savefig(f"results/epoch_{epoch}.png")

    plt.close()


def animate_motion_energy(
        frame_index, motion_energy_x, motion_energy_xs, motion_energy_y, motion_energy_ys, ts, axs
        ):
    '''
    This function is called periodically from FuncAnimation in the plot function
    in the Motion Energy class.
    '''

    # Add x and y to lists
    motion_energy_xs.append(motion_energy_x)
    motion_energy_ys.append(motion_energy_y)
    ts.append(frame_index)

    # Limit x and y lists to 20 items
    #xs = xs[-20:]
    #ys = ys[-20:]

    # Draw x and y lists
    axs.clear()
    axs.plot(ts, motion_energy_xs, color='b', label='motion_energy_x')
    axs.plot(ts, motion_energy_ys, color='y', label='motion_energy_y')

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('Motion Energy Evolution')
    plt.ylabel('Optical Flow')








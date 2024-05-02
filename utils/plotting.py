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






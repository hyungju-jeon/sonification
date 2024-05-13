import matplotlib.pyplot as plt
import numpy as np

from matplotlib.pylab import f
from matplotlib import cm

from sympy.physics.vector import ReferenceFrame
from sympy.physics.vector import curl

from PIL import Image as im


#flow_x = np.load('./flow_arrays/flow_x.npy')
#flow_y = np.load('./flow_arrays/flow_y.npy')


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


def plot_vector_field(flow_x, flow_y):

    flow_x = np.array(flow_x)
    flow_y = np.array(flow_y)

    # 1D arrays 
    x = np.arange(0 , flow_x.shape[1], 1) 
    y = np.arange(0 , flow_x.shape[0], 1) 

    # Meshgrid (2D arrays)
    X, Y = np.meshgrid(x, y)

    # Assign vector directions
    # U = arr if dir == 'Y' else np.ones(arr.shape) 
    # V = arr if dir == 'X' else np.ones(arr.shape)

    U = flow_x
    V = flow_y

    # Depict illustration 
    plt.figure(figsize=(10, 10)) 
    plt.streamplot(X, Y, U, V, density=1.4, linewidth=None, color='#A23BEC') 

    # Show plot with grid 
    plt.grid() 
    plt.show()


def calc_divergence(flow_x, flow_y):

    flow_x = np.array(flow_x)
    flow_y = np.array(flow_y)
    
    x = np.arange(0 , flow_x.shape[1], 1) 
    y = np.arange(0 , flow_x.shape[0], 1) 

    X, Y = np.meshgrid(x, y)

    U = flow_x
    V = flow_y

    dU = np.gradient(U, x, axis=1)
    dV = np.gradient(V, y, axis=0)

    div = dU + dV

    return div
'''

def calc_curl(flow_x, flow_y):

    flow_x = np.array(flow_x)
    flow_y = np.array(flow_y)
    
    x = np.arange(0 , flow_x.shape[1], 1) 
    y = np.arange(0 , flow_x.shape[0], 1) 

    X, Y = np.meshgrid(x, y)

    U = flow_x
    V = flow_y

    dU = np.gradient(U, x, axis=1)
    dV = np.gradient(V, y, axis=0)

    R = ReferenceFrame('R')

    F = R[1]**2 * R[2] * R.x - R[0]*R[1] * R.y + R[2]**2 * R.z

    c = curl(F, R)  

    return c
    '''



# Define the partial derivatives of Fx and Fy with respect to x and y
def partial_x(Fx, h=1e-6):
    return (np.roll(Fx, -1, axis=1) - np.roll(Fx, 1, axis=1)) / (2 * h)

def partial_y(Fy, h=1e-6):
    return (np.roll(Fy, -1, axis=0) - np.roll(Fy, 1, axis=0)) / (2 * h)

# Calculate the curl of the vector field
def calc_curl(flow_x, flow_y):

    flow_x = np.array(flow_x)
    flow_y = np.array(flow_y)

    # Create a grid of points (x, y)
    x = np.arange(0 , flow_x.shape[1], 1) 
    y = np.arange(0 , flow_x.shape[0], 1) 

    X, Y = np.meshgrid(x, y)

    U = flow_x
    V = flow_y

    curl_x = partial_y(V) - partial_x(U)
    curl_y = partial_x(U) - partial_y(V)
    '''
    curl_x = (curl_x - curl_x.min()) / (curl_x.max() - curl_x.min())
    curl_y = (curl_y - curl_y.min()) / (curl_y.max() - curl_y.min())
    '''
    total_curl = np.sqrt(curl_x ** 2 + curl_y ** 2)

    return total_curl

# Calculate the curl of the vector field at each point in the grid
#total_curl = calc_curl(flow_x, flow_y)

#print(total_curl.shape)
#print(total_curl.min())
#print(total_curl.max())

'''
plot_vector_field(flow_x, flow_y)

div = calc_divergence(flow_x, flow_y)
print(div.shape)

img = im.fromarray(div, 'RGB')
img.show()

plt.imshow(div, cmap='hot', interpolation='nearest')
plt.show()

c = calc_curl(flow_x, flow_y)

print(c)
'''










import numpy as np
from pytreegrav import Accel, Potential
import time


N = 50  # number of particles
x = np.random.rand(N, 3)  # positions randomly sampled in the unit cube
m = np.repeat(1.0 / N, N)  # masses - let the system have unit mass
h = np.repeat(
    0.01, N
)  # softening radii - these are optional, assumed 0 if not provided to the frontend functions

stime = time.perf_counter_ns()
accel_bruteforce = Potential(x, m, h, method="bruteforce")
print(f"Time to build tree: {(time.perf_counter_ns() - stime) / 1e6} mseconds")

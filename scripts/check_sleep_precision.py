# %%
import asyncio
import time
import numpy as np
from matplotlib import pyplot as plt

K = 10000
T = np.zeros(K)


async def custom_sleep(duration):
    start = time.perf_counter_ns()
    while True:
        await asyncio.sleep(0)  # Yield control to allow other tasks to run
        if time.perf_counter_ns() - start >= duration:
            break


async def async_loop():
    for i in range(K):
        start_time = time.perf_counter_ns()
        await asyncio.sleep(0.001)
        elapsed = time.perf_counter_ns() - start_time
        await custom_sleep(max(0.001 - elapsed, 0))
        T[i] = time.perf_counter_ns() - start_time


async def main():
    global T
    # Run the async loop
    await async_loop()

    T = T / 1e6  # Convert to milliseconds
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(T)
    ax1.set(xlabel="Sample", ylabel="Measured Time (ms)")
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.hist(T, bins=100, range=(1, 1.5))
    ax2.set(xlabel="Measured Time (ms)", ylabel="Hist")
    plt.suptitle("Measured Time (ms) for asyncio.sleep(0.001)")
    print(
        f"mean = {np.mean(T)}, std = {np.std(T)} and absolute deviation = {np.mean(np.abs(T - 1.0))} ms"
    )
    plt.show()


if __name__ == "__main__":
    # Running the main coroutine
    asyncio.run(main())

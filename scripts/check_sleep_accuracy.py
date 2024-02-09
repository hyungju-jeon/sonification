# %%
import asyncio
import time
import numpy as np
from matplotlib import pyplot as plt

K = 10000
T = np.zeros(K)
x = [0]


async def my_async_function():
    while True:
        await asyncio.sleep(1.5)
        x[0] += 1


async def custom_sleep(duration):
    while True:
        start = time.perf_counter_ns()
        while True:
            await asyncio.sleep(0)  # Yield control to allow other tasks to run
            if time.perf_counter_ns() - start >= duration:
                break
        print(f"Current x: {x[0]}")


async def main():
    await asyncio.gather(my_async_function(), custom_sleep(499e6))
    # Run the async loop


#     print("init_main")
#     await async_loop()

#     fig = plt.figure(figsize=(12, 8))
#     ax1 = fig.add_subplot(2, 1, 1)
#     ax1.plot(T / (1e6))
#     ax1.set(xlabel="Sample", ylabel="Measured Time (ms)")
#     ax2 = fig.add_subplot(2, 1, 2)
#     ax2.hist(T / (1e6), bins=100, range=(1, 1.5))
#     ax2.set(xlabel="Measured Time (ms)", ylabel="Hist")
#     plt.suptitle("Measured Time (ms) for asyncio.sleep(0.001)")
#     print(
#         f"mean = {np.mean(T)}, std = {np.std(T)} and absolute deviation = {np.mean(np.abs(T - 1.0))} ms"
#     )
#     plt.show()
#     print("init_main done")


if __name__ == "__main__":
    # Running the main coroutine
    asyncio.run(main())

import numpy as np

import matplotlib.pyplot as plt

'''
x = np.linspace(0, 0.5, 100)
y = 0.5 * (np.exp((x) ** 1))

plt.figure()
plt.plot(x, y)
plt.xlabel("$x$")
plt.ylabel(r"$\exp(x)$")
plt.title("Exponential function")

plt.show()
'''

'''
w = 1
b = 0.5

x = np.linspace(-1, 1, 100)
y = np.log((w * (x + b)) / (1 - (w * (x + b))))

plt.figure()
plt.plot(x, y)

plt.show()
'''


'''
x = np.linspace(0, 5, 100)
y = 0.5 * (2.71 ** x)

plt.figure()
plt.plot(x, y)
plt.xlabel("$x$")
plt.ylabel(r"$\exp(x)$")
plt.title("Exponential function")

plt.show()
'''

w = 1
b = 0.5

x = np.linspace(-1, 1, 100)
y = np.exp((w * (x + b)))

plt.figure()
plt.plot(x, y)

plt.show()

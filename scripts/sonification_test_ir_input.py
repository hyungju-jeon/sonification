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


w = 1
b = 0.5

x = np.linspace(-1, 1, 100)
y = np.exp((w * (x + b)))

plt.figure()
plt.plot(x, y)

plt.show()

'''

a = - 50
b = 50

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

x[0] = a + ((x[0] - min(x)) * ((b - a)) / (max(x) - min(x)))
x[9] = a + ((x[9] - min(x)) * ((b - a)) / (max(x) - min(x)))

print(min(x))
print(max(x))


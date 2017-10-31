# This program is written by Python3
# important function used in Deep Learning

import numpy as np
import matplotlib.pylab as plt

def sigmoid_function(x):
	return 1 / (1 + np.exp(-x))

# np.arange(min, max, interval)
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

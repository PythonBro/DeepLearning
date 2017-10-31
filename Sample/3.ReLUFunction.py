# This program is written by Python3
# important function used in MODERN Deep Learning

import numpy as np
import matplotlib.pylab as plt

def ReLU_function(x):
	return np.maximum(0, x)

# np.arange(min, max, interval)
x = np.arange(-5.0, 5.0, 0.1)
y = ReLU_function(x)
plt.plot(x, y)
plt.ylim(-1, 6)
plt.show()

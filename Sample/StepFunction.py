# This program is written by Python3

import numpy as np
import matplotlib.pylab as plt

def step_function(x):
	return np.array(x>0, dtype=np.int)

# np.arange(min, max, interval)
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

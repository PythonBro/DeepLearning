# This program is written by Python3

import numpy as np

def softmax_function(a):
	c = np.max(a)
	# overflow measure
	exp_a = np.exp(a - c)
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a

	return y

x = np.array([0.3, 2.9, 4.0])
y = softmax_function(x)
print(y)
# sum of softmax
print(np.sum(y))

# This program in written in Python3


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from numpy.random import normal

# Number of study data
N=0
# Degree of polynomial
M=[0, 0, 0, 0]

# Original polynomial
def h(x):
	y = 10 * (x - 0.5) ** 2
	return y

# create a dataset
def create_data(num):
	dataset = DataFrame(columns=['x', 'y'])
	for i in range(num):
		x = float(i) / float(num-1)
		# h(x) + noise
		y = h(x) + normal(scale=0.2)
		dataset = dataset.append(Series([x, y], index=['x', 'y']), ignore_index=True)
	return dataset

# calculate parameters by likelihood
def M_l_estimation(dataset, m):
	t = dataset.y
	phi = DataFrame()
	# create phi matrix
	for i in range(0, m+1):
		p = dataset.x ** i
		p.name = "x ** %d" % i
		phi = pd.concat([phi, p], axis=1)
	ws = np.dot(np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T), t)

	def f(x):
		y = 0.0
		for i, w in enumerate(ws):
			y += w * (x ** i)
		return y

	# s = standard deviation
	s = 0.0
	for index, line in dataset.iterrows():
		s += (f(line.x) - line.y) ** 2
	s /= len(dataset)

	return(f, ws, np.sqrt(s))

def main():
	N = int(input("学習用データの個数N:"))
	M = [int(i) for i in (input("多項式の字数M（スペース区切りで4つ入力）:").split(" "))]
	training_set = create_data(N)
	df_ws = DataFrame()
	# make a graph
	fig = plt.figure()
	for c, m in enumerate(M):
		f, ws, s = M_l_estimation(training_set, m)
		df_ws = df_ws.append(Series(ws, name="M=%d" % m))

		subplot = fig.add_subplot(2, 2, c+1)
		subplot.set_xlim(-0.1, 1.1)
		subplot.set_ylim(-2, 4)
		subplot.set_title("M:%d" % m)

		line_x = np.linspace(0, 1, 101)
		line_y = h(line_x)
		subplot.plot(line_x, line_y, color="green", linestyle="--")

		subplot.scatter(training_set.x, training_set.y, marker="x", color="blue")

		line_x = np.linspace(0, 1, 101)
		line_y = f(line_x)
		label = "sigma=%.2f" % s
		subplot.plot(line_x, line_y, color="red", label=label)

		subplot.plot(line_x, line_y + s, color="red", linestyle="--")
		subplot.plot(line_x, line_y - s, color="red", linestyle="--")
		subplot.legend(loc='best')

	fig.show()
	input("enter.")

if __name__ == '__main__':
	main()

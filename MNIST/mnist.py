# This program is written in Python2
# https://qiita.com/phyblas/items/375ab130e53b0d04f784

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

mnist = datasets.fetch_mldata('MNIST original')
X, y = mnist.data, mnist.target
X = X/255.

print(X.shape)
print(y.shape)
print(y)


#for i in range(1, 10):
#	plt.subplot(330+i)
#	plt.imshow(X[30+i*6500].reshape(28, 28), cmap='gray_r')
#	plt.show()

n = len(X)
s = np.random.permutation(n)
nn = int(n/5)
X_kunren, X_kensho = X[s[nn:]], X[s[:nn]]
y_kunren, y_kensho = y[s[nn:]], y[s[:nn]]

class kaiki:
	def __init__(self, gakushuritsu):
		self.gakushuritsu = gakushuritsu

	def gakushu(self, X, y, kurikaeshi, n_batch=0, X_kensho=0, y_kensho=0, patience=0):
		n = len(y)

		if(type(X_kensho) != np.ndarray):
			X_kensho, y_kensho = X, y

		if(n_batch==0 or n<n_batch):
			n_batch = n

		self.n_group = int(y.max()+1)
		y_1h = y[:, None] == range(self.n_group)
		self.w = np.zeros([X.shape[1]+1, self.n_group])

		self.sonshitsu = []
		self.kunren_seikaku = []
		self.kensho_seikaku = []
		saikou = 0
		agaranai = 0

		for j in range(kurikaeshi):
			s = np.random.permutation(n)
			for i in range(0, n, n_batch):
				Xn = X[s[i:i+n_batch]]
				yn = y_1h[s[i:i+n_batch]]
				phi = self.softmax(Xn)
				eee = (yn - phi)/len(yn)*self.gakushuritsu
				self.w[1:] += np.dot(eee.T, Xn).T
				self.w[0] += eee.sum(0)

			seigo = self.yosoku(X) == y
			kunren_seikaku = seigo.mean()*100
			seigo = self.yosoku(X_kensho) == y_kensho
			kensho_seikaku = seigo.mean()*100

			if(kensho_seikaku > saikou):
				saikou = kensho_seikaku
				agaranai = 0
				w = self.w.copy()
			else:
				agaranai += 1

			self.kunren_seikaku += [kunren_seikaku]
			self.kensho_seikaku += [kensho_seikaku]
			self.sonshitsu += [self.entropy(X, y_1h)]

			if(patience != 0 and agaranai >= patience):
				break

		self.w = w

	def yosoku(self, X):
		return (np.dot(X, self.w[1:]) + self.w[0]).argmax(1)

	def softmax(self, X):
		h = np.dot(X, self.w[1:]) + self.w[0]
		exp_h = np.exp(h.T - h.max(1))
		return (exp_h / exp_h.sum(0)).T

	def entropy(self, X, y_1h):
		return -(y_1h*np.log(self.softmax(X)+1e-7)).mean()

gakushuritsu = 0.24
kurikaeshi = 1000
n_batch = 100
patience = 10
mmk = kaiki(gakushuritsu)
mmk.gakushu(X_kunren, y_kunren, kurikaeshi, n_batch, X_kensho, y_kensho, patience)

plt.figure(figsize=[8, 8])

ax = plt.subplot(211)
plt.plot(mmk.sonshitsu, '#000077')
plt.legend([u'lost'], prop={'family':'AppleGothic', 'size':17})
plt.tick_params(labelbottom = 'off')

ax = plt.subplot(212)
ax.set_ylabel(u'accuracy(%)', fontname='AppleGothic', size=18)
plt.plot(mmk.kunren_seikaku, '#dd0000')
plt.plot(mmk.kensho_seikaku, '#00aa00')
plt.legend([u'training', u'test'], prop={'family':'AppleGothic', 'size':17})

plt.show()

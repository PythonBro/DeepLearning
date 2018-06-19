# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from numpy.random import normal

N=0 #学習用のデータ数
M=[0,0,0,0] #多項式の次数

#元の多項式
def h(x):
    y = 10 * (x - 0.5) ** 2
    return y

#データセットを作成
def create_data(num):
    dataset = DataFrame(columns=['x', 'y'])
    for i in range(num):
        x = float(i) / float(num-1) #0~1の間をデータ数で区切る
        y = h(x) + normal(scale=0.7) #正規分布の標準偏差0.7の乱数に従った雑音を付加
        dataset = dataset.append(Series([x,y], index=['x', 'y']), ignore_index=True)
    return dataset

#最尤推定でパラメータを出す
def M_l_estimation(dataset, m):
    t = dataset.y
    phi = DataFrame()
    #Φ行列を作成
    for i in range(0, m+1):
        p = dataset.x ** i
        p.name = "x ** %d" % i
        phi = pd.concat([phi, p], axis = 1)
    ws = np.dot(np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T), t) #w=(Φ^T*Φ)^(−1)*Φ^T*tを計算

    #仮に置く多項式
    def f(x):
        y = 0.0
        for i, w in enumerate(ws):
            y += w * (x ** i)
        return y

    #s=平方平均二乗誤差(すなわち標準偏差σ)
    s = 0.0
    for index, line in dataset.iterrows():
        s += (f(line.x) - line.y) ** 2
    s /= len(dataset)

    return(f, ws, np.sqrt(s))

def main():
    N = int(input("学習用データの個数N:"))
    M = [int(i) for i in (input("多項式の次数M(スペース区切りで４つ入力):")).split(" ")]
    training_set = create_data(N)
    df_ws = DataFrame()
    #グラフの作成
    fig = plt.figure()
    for c, m in enumerate(M):
        f, ws, s = M_l_estimation(training_set, m)
        df_ws = df_ws.append(Series(ws, name="M=%d" % m))


        #subplotで複数のグラフを表示
        subplot = fig.add_subplot(2, 2, c+1)
        subplot.set_xlim(-0.1, 1.1)
        subplot.set_ylim(-7, 7)
        subplot.set_title("M:%d" % m)

        #元のグラフy=(x-0.5)^2を緑で表示の破線で表示
        line_x = np.linspace(0, 1, 101)
        line_y = h(line_x)
        subplot.plot(line_x, line_y, color = "green", linestyle = "--")

        #データの点を青いバツ印で表示
        subplot.scatter(training_set.x, training_set.y, marker = "x", color = "blue")

        #最尤推定で得られた多項式のグラフを表示
        line_x = np.linspace(0, 1, 101)
        line_y = f(line_x)
        label = "simga=%.2f" % s
        subplot.plot(line_x, line_y, color = "red", label = label)
        #誤差範囲の表示
        subplot.plot(line_x, line_y + s, color = "red", linestyle = "--")
        subplot.plot(line_x, line_y - s, color = "red", linestyle = "--")
        subplot.legend(loc = 1)

    fig.show()
    input("enter.")

if __name__ == '__main__':
    main()

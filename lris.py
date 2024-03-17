import os
import numpy as np
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd
from perceptron import Perceptron, plot_decision_regions
from ADALINE import AdalineGD

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import seaborn as sns





s = os.path.join(".", "Iris.csv")

df = pd.read_csv(s, header=None, encoding="utf-8")

y = df.iloc[1:101, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[1:101, [0, 2]].values
X = X.astype(float) 


#データの確認
# plt.scatter(X[:50,0],X[:50,1], color="red", marker="o", label="setosa")
# plt.scatter(X[51:,0],X[51:,1], color="blue", marker="x", label="versicolor")

# plt.xlabel("sepal length[cm]")
# plt.ylabel("petal length[cm]")

# plt.legend(loc="upper left")
# plt.show()




#パーセプトロン
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)


#エラーの数可視化
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")

# plt.xlabel("Epochs")
# plt.ylabel("Number of update")

# plt.show()


#決定線可視化
# plot_decision_regions(X, y, ppn)




#Adaline
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
 
# ada1 = AdalineGD(n_iter=10,eta=0.01).fit(X,y)

# ax[0].plot(range(1,len(ada1.cost_)+1), np.log10(ada1.cost_), marker="o")
# ax[0].set_xlabel("Epochs")
# ax[0].set_ylabel("log(Sum-squared-error)")
# ax[0].set_title("Adaline 0.01")


# ada2 = AdalineGD(n_iter=10,eta=0.0001).fit(X,y)

# ax[1].plot(range(1,len(ada2.cost_)+1), np.log10(ada2.cost_), marker="o")
# ax[1].set_xlabel("Epochs")
# ax[1].set_ylabel("log(Sum-squared-error)")
# ax[1].set_title("Adaline 0.0001")

# plt.show()


X_std = np.copy(X)
X_std[:,0] = (X[:,0]-X[:,0].mean())/X[:,0].std()
X_std[:,1] = (X[:,1]-X[:,1].mean())/X[:,1].std()

ada_gb = AdalineGD(n_iter=15,eta=0.01)
ada_gb.fit(X_std,y)


plot_decision_regions(X_std,y,ada_gb)
plt.xlabel("sepal length[cm]")
plt.ylabel("petal length[cm]")
plt.title("Adaline_std")
plt.show()


plt.plot(range(1,len(ada_gb.cost_)+1), np.log10(ada_gb.cost_), marker="o")
plt.xlabel("Epochs")
plt.ylabel("log(Sum-squared-error)")
plt.title("Adaline_std")
plt.show()


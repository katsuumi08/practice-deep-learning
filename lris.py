import os
import numpy as np
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd
from perceptron import Perceptron, plot_decision_regions

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

# plt.scatter(X[:50,0],X[:50,1], color="red", marker="o", label="setosa")
# plt.scatter(X[51:,0],X[51:,1], color="blue", marker="x", label="versicolor")

# plt.xlabel("sepal length[cm]")
# plt.ylabel("petal length[cm]")

# plt.legend(loc="upper left")
# plt.show()


ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")

plt.xlabel("Epochs")
plt.ylabel("Number of update")

plt.show()

plot_decision_regions(X, y, ppn)

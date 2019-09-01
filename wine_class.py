import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SBS import SBS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('https://archive.ics.uci.edu/ml/machinelearning-databases/wine/wine.data', header=None)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

std = StandardScaler()
X = std.fit_transform(X)
y = std.transform(y)
knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(X, y)
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.0])
plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.grid()
plt.show()
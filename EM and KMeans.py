import matplotlib.pyplot
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
iris = datasets.load_iris()
x = pd.DataFrame(iris.data)
x.columns = ['Sepal_length','Sepal_width','Petal_length','Petal_width']
y = pd.DataFrame(iris.target)
y.columns = ['Targets']

model = KMeans(n_clusters = 3)
model.fit(x)

plt.Figure(figsize=(14,14))
colormap = np.array(['red','blue','green'])
plt.subplot(2,2,1)
plt.scatter(x.Petal_length,x.Petal_width,c = colormap[y.Targets], s = 40)
plt.title("Real Clusters")
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

plt.subplot(2,2,2)
plt.scatter(x.Petal_length,x.Petal_width, c = colormap[model.labels_], s = 40)
plt.title("K-Means Clustering")
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(x)
xsa = scaler.transform(x)
xs = pd.DataFrame(xsa, columns = x.columns)

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components= 3)
gmm.fit(xs)
gmm_y = gmm.predict(xs)

plt.subplot(2,2,3)
plt.scatter(x.Petal_length,x.Petal_width,c = colormap[gmm_y], s = 40)
plt.title("GMM Clustering")
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')


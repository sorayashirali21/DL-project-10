#First, import the KMeans class and initialize a model with three clusters, as
#follows:

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, 2:4]
y = iris.target

from matplotlib import pyplot as plt

k = 3
from sklearn.cluster import KMeans

#We then fit the model on the data:
kmeans_sk = KMeans(n_clusters=3, random_state=42)
kmeans_sk.fit(X)

#After that, we can obtain the clustering results, including the clusters for data
#samples and centroids of individual clusters:
clusters_sk = kmeans_sk.labels_
centroids_sk = kmeans_sk.cluster_centers_

# Similarly, we plot the clusters along with the centroids:
plt.scatter(X[:, 0], X[:, 1], c=clusters_sk)
plt.scatter(centroids_sk[:, 0], centroids_sk[:, 1], marker='*', s=200, c='#050505')
plt.show()

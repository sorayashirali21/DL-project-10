#choosing the value of k

from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt

#the Elbow method to the example we covered in the previous section
iris = datasets.load_iris()
X = iris.data
y = iris.target

# resulting SSE
k_list = list(range(1, 7))
sse_list = [0] * len(k_list)

for k_ind, k in enumerate(k_list):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    clusters = kmeans.labels_
    centroids = kmeans.cluster_centers_

    sse = 0
    for i in range(k):
        cluster_i = np.where(clusters == i)

        sse += np.linalg.norm(X[cluster_i] - centroids[i])

    print('k={}, SSE={}'.format(k, sse))
    sse_list[k_ind] = sse

#t the SSE versus the various k ranges
plt.plot(k_list, sse_list)
plt.show()

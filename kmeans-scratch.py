#Implementing k-means from scratch
#We herein only use two features out of the original four for
#simplicity:

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, 2:4]
y = iris.target

#Since the dataset contains three iris classes, we plot it in three different colors, as
#follows:
#This will give us the following output for the original data plot:

import numpy as np
from matplotlib import pyplot as plt
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

#by randomly selecting three samples as initial centroids:
k = 3
random_index = np.random.choice(range(len(X)), k)
centroids = X[random_index]

#We visualize the data (without labels any more) along with the initial random
#centroids:
def visualize_centroids(X, centroids):
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
    plt.show()



visualize_centroids(X, centroids)

#Refer to the following screenshot for the data, along with the initial random
#centroids:
#we need to define a function calculating distance that is measured
#by the Euclidean distance, as demonstrated herein:

def dist(a, b):
    return np.linalg.norm(a - b, axis=1)

#Then, we develop a function that assigns a sample to the cluster of the nearest
#centroid:
def assign_cluster(x, centroids):
    distances = dist(x, centroids)
    cluster = np.argmin(distances)
    return cluster

def update_centroids(X, centroids, clusters):
    for i in range(k):
        cluster_i = np.where(clusters == i)
        centroids[i] = np.mean(X[cluster_i], axis=0)

clusters = np.zeros(len(X))

#We set the tolerance of the first condition and the maximum number of iterations
#as follows:
tol = 0.0001
max_iter = 100

#Initialize the clusters' starting values, along with the starting clusters for all samples
as follows:
iter = 0
centroids_diff = 100000

#then visualizes the
#latest centroids:
from copy import deepcopy
while iter < max_iter and centroids_diff > tol:
    for i in range(len(X)):
        clusters[i] = assign_cluster(X[i], centroids)
    centroids_prev = deepcopy(centroids)
    update_centroids(X, centroids, clusters)
    iter += 1
    centroids_diff = np.linalg.norm(centroids - centroids_prev)
    print('Iteration:', str(iter))
    print('Centroids:\n', centroids)
    print('Centroids move: {:5.4f}'.format(centroids_diff))
    visualize_centroids(X, centroids)

plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
plt.show()




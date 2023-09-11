import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:
    '''
    KMeans is an _unsupervised learning_ method that clusters data sets into
    k different clusters. Each sample is assigned to the cluster with the nearest
    mean, then the means (centroids) and clusters are updated during an iterative
    optimization process

    1. Initialize cluster centers
    2. Repeat until converged
        2a. Update cluster labels: Assign pointes to the nearest cluster center (centroid)
        2b. Update the cluster centers (centraois)
    '''

    def __init__(self, K=5, max_iters=100, plot_steps=False) -> None:
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]

        # store the centers (mean vector for each cluster)
        self.centroids = []

    def predict(self, X): # no y value since this is unsupervised
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize centroids
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # optimize clusters
        for _ in range(self.max_iters):
            # assign samples to the closest centroids (create initial clusters)
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            #calculate new centroids from clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids)
                break

            if self.plot_steps:
                self.plot()

        # classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)
    
    def _get_cluster_labels(self, clusters):
        # each sample will get the label of thecluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(cluster):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels

    def _create_clusters(self, centroids):
        # assign the samples to the closest centroids
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        pass

    def _is_converged(self, centroids_old, centroids):
        pass

    def plot(self):
        pass

    def _get_cluster_labels():
        pass
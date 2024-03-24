from collections import Counter
import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(x1-x2)**2)

# create a function that calculates the Minkowski distance
def minkowski_distance(x1, x2, p):
    return np.sum(np.abs(x1-x2)**p)**(1/p)

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1-x2))

def cosine_distance(x1, x2):
    return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

class KNN:

    def __init__ (self, k=3, distance='euclidean'):
        self.k = k
        self.distance = distance

    def fit (self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X) -> list:
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute the distance
        if self.distance == 'euclidean':
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.distance == 'minkowski':
            distances = [minkowski_distance(x, x_train, p=2) for x_train in self.X_train]
        elif self.distance == 'manhattan':
            distances = [manhattan_distance(x, x_train) for x_train in self.X_train]
        elif self.distance == 'cosine':
            distances = [cosine_distance(x, x_train) for x_train in self.X_train]
        else:
            raise ValueError('Invalid distance metric')
        
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        #get the closest K
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
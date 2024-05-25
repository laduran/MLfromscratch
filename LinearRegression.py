# From here: https://www.youtube.com/watch?v=ltXSoduiVwY
import numpy as np

class LinearRegression:


    def __init__(self, alpha=0.001, n_iters=1000) -> None:
        self.α = alpha
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    # used for training on a dataset
    def fit(self, X: np.ndarray, y):
        n_samples, n_features = X.shape
        #self.weights = np.zeros(n_features) # Could try this with random values
        self.weights = np.random.rand(n_features)
        self.bias = 0

        for _ in range (self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            # calculate the gradients (use transpose of X vector)
            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            # update the weights and bias based using gradient direction
            self.weights = self.weights - self.α * dw
            self.bias = self.bias - self.α * db


    # used to predict next value
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

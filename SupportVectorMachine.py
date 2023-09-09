# From here: https://www.youtube.com/watch?v=T9UcK-TxQGw

import numpy as np

class SupportVectorMachine:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000) -> None:
        self.lr = learning_rate
        self.lambra_param = lambda_param
        self.n_iters = n_iters
        self.w = None   #weights
        self.b = None   #biases

    def fit(self, X: np.ndarray, y:np.ndarray):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        # init the weights
        self.w = np.zeros(n_features)  # Again it would better to randomly initialize the weights
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambra_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambra_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X: np.ndarray):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

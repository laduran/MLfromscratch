import numpy as np

def unit_step_func(x):
    '''
    Using this as the activation function for output
    '''
    return np.where(x > 0 , 1, 0)

class Perceptron:
    '''
    A single layer neural network a.k.a. a Perceptron.
    Can only learn linearly separable problem types
    
    '''
    def __init__(self, learning_rate=0.1, n_iters=1000) -> None:
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None
        pass

    def fit(self, X: np.ndarray, y):
        n_samples, n_features = X.shape
        #init parameters

        self.weights = np.zeros(n_features) # better way would be to randomly initialize
        self.bias = 0
        y_ = np.where(y>0, 1, 0)

        # learn weights
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # perceptron update rule
                update = self.lr * (y_[idx] - y_predicted)    
                self.weights += update * x_i
                self.bias += update
        

    def predict(self, X: np.ndarray):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted
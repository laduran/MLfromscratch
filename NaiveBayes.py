import numpy as np

class NaiveBayes:

    def fit(self, X: np.ndarray, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Calculate mean, variance & prior for each class
        self._mean = np.zeros(shape=(n_classes, n_features), dtype=np.float64)
        self._variance = np.zeros(shape=(n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(shape=(n_classes), dtype=np.float64) # why only n_classes here for the shape?

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._variance[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_predict = [self._predict(x) for x in X]
        return np.array(y_predict)

    def _predict(self, x):
        posteriors = []

        #calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)

        # return class with the highest posterior
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        # Probability Density Function
        mean = self._mean[class_idx]
        var = self._variance[class_idx]
        numerator = np.exp(-(x-mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
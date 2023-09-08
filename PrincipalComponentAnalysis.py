import numpy as np


class PCA:
    '''
    Dimensionality Reduction - efficiency?
    An unsupervised learning method that is often used to reduce the dimensionality
    of the dateset by transforming a large set into a lower dimensional set that still contains
    most of the information of the large set.

    Find a transoformation such that:
        1) The transformed features are linearly independent
        2) Dimensionality can be reduced by only taking dimensions (features?) with
        the highest importance
        3) New dimensions should maximize the projection error (?)
        4) The projected points should have maximum spread (max variance)

    This boils down to an Eigenvector/Eigenvalue problem...

    Steps:
        Subtract mean from X
        Calculate Covariance(X, X)
        Calculate eigenvectors and eigenvalues of the Covariance matrix
        Sort the eigenvectors according to the eigenvalues in descending order
        Choose first k eigenvectors that will be the new k dimensions
        Transform the original n-dimensional data points into k dimensions
            (Projects with dot product)
    '''
    def __init__(self, n_components) -> None:
        self.n_components = n_components
        self.components = None
        self.mean = None


    def fit(self, X: np.ndarray):
        # mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Covariance
        cov = np.cov(X.T) # Covariance(X, X)

        #Calc Eigenvectors and Eigenvalues
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        # eigenvectors v = [:, i] columne vector, transpose for easier calculations
        eigenvectors = eigenvectors.T

        #sort eigenvectors according to their eigenvalues
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)
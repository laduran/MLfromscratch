import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from LogisticRegression import LogisticRegression

def accuracy(y_pred, y_test):
    '''Calculate a Mean-squared error'''
    return np.sum(y_pred==y_test)/len(y_test)

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = LogisticRegression() # classifier
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy(y_pred, y_test)
print(accuracy)
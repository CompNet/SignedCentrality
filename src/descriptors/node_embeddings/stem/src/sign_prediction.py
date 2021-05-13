import models
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

def train_classifier(model, X, y, operation='hadamard'):
    clf = LogisticRegression()
    X_train = []
    for u, v in X:
        feature = model.get_edge_features(u, v, operation)
        X_train.append(feature)
    X_train = np.array(X_train)
    y_train = np.array(y)
    clf.fit(X_train, y_train)

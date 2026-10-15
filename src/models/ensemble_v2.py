"""Importable estimators keep CLI-produced checkpoints portable across processes."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

class EqualEnsemble(ClassifierMixin, BaseEstimator):
    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        self.weights = weights

    def fit(self, x, y):
        self.classes_ = np.unique(y)
        self.fitted_ = [clone(model).fit(x, y) for model in self.estimators]
        return self

    def predict_proba(self, x):
        probabilities = []
        for model in self.fitted_:
            p = model.predict_proba(x)
            probabilities.append(p[:, [list(model.classes_).index(label) for label in self.classes_]])
        return np.average(probabilities, axis=0, weights=self.weights)

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(axis=1)]


class DenseArray(BaseEstimator):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.toarray() if hasattr(x, 'toarray') else x

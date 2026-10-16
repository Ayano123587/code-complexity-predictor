"""Training-fold-only family and class-mass weighting for learned ensembles."""
from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
from src.models.mechanics_v3 import candidate
from src.models.ensemble_v2 import EqualEnsemble


def sample_weights(labels,groups,balance_classes=False):
    labels=np.asarray(labels);groups=np.asarray(groups)
    if labels.ndim!=1 or groups.ndim!=1 or len(labels)!=len(groups) or not len(labels):raise ValueError('Aligned nonempty training labels/groups required')
    sizes=Counter(groups.tolist());weights=np.asarray([1./sizes[g] for g in groups])
    if balance_classes:
        mass={label:weights[labels==label].sum() for label in np.unique(labels)}
        weights*=np.asarray([weights.sum()/(len(mass)*mass[label]) for label in labels])
    return weights/weights.mean()


class FamilyEnsemble(ClassifierMixin,BaseEstimator):
    def __init__(self,mode='family',mechanics=False):
        self.mode=mode;self.mechanics=mechanics

    def fit(self,x,y,groups):
        if self.mode not in ['family','family_class']:raise ValueError('Unknown family weighting mode')
        weights=sample_weights(y,groups,self.mode=='family_class')
        template=candidate('mechanics_leaf3' if self.mechanics else 'v2_baseline')
        fitted=[]
        for model in template.estimators:
            if self.mode=='family_class':model.set_params(classifier__class_weight=None)
            fitted.append(model.fit(x,y,classifier__sample_weight=weights))
        self.classes_=np.unique(y);self.ensemble_=EqualEnsemble([],weights=[.25,.75])
        self.ensemble_.classes_=self.classes_;self.ensemble_.fitted_=fitted
        return self

    def predict_proba(self,x):return self.ensemble_.predict_proba(x)
    def predict(self,x):return self.ensemble_.predict(x)

"""Interpretable lexical structure features, not formal static analysis."""
from collections import Counter
import math
from sklearn.base import BaseEstimator,TransformerMixin
class StructureFeatures(BaseEstimator,TransformerMixin):
    def fit(self,x,y=None):return self
    def transform(self,texts):
        out=[]
        for text in texts:
            ts=text.split();counts=Counter(ts);depth=0;maxdepth=0
            for token in ts:
                if token=='{':depth+=1;maxdepth=max(maxdepth,depth)
                elif token=='}':depth=max(0,depth-1)
            out.append(dict(tokens=math.log1p(len(ts)),brace_depth=maxdepth,**{f'count_{t}':math.log1p(counts[t]) for t in ['for','while','if','sort','lower_bound','upper_bound','vector','map','set','queue','stack','return','[','<<','>>','*','/','%']}))
        return out

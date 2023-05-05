import itertools as it 

import mmh3
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer
from skpartial.pipeline import make_partial_union



class BloomVectorizer(BaseEstimator, TransformerMixin):
    """
    BloomVectorizer. Will hash each word in text multiple times using mmh3. 

    The benefit of this method is that it can accept generators in the `.transform()` method. 
    The downside is that it is a fair bit slower. There is also a larger chance of a hash 
    collision but the output embedding should also be somewhat more efficiently used.
    """
    def __init__(self, n_buckets=2000, n_hash=3, lowercase=True) -> None:
        self.n_buckets = n_buckets
        self.n_hash = n_hash
        self.lowercase = lowercase
    
    def fit(self, X, y=None):
        return self 
    
    def partial_fit(self, X, y=None, classes=None):
        return self 

    def transform(self, X, y=None):
        x_orig, x_new = it.tee(X)
        size = sum(1 for x in x_orig)
        res = lil_matrix((size, self.n_buckets), dtype=np.int8)
        for i, x in enumerate(x_new):
            for w in x.lower().split(" ") if self.lowercase else x.split(" "):
                for _ in range(self.n_hash):
                    col = mmh3.hash(f"{_}-{w}", signed=False) % self.n_buckets
                    res[i, col] = 1
        return res


class BloomishVectorizer(BaseEstimator, TransformerMixin):
    """
    BloomishVectorizer. Will hash each word in text multiple times by re-using the HashingVectorizer from sklearn. 

    The downside of this method is that it cannot accept generators in the `.transform()` method. 
    It is however a fair bit faster, typically 5x. The output is going to be more sparse than the
    BloomVectorizer because we simple concatenate the HashingVectorizers.
    """
    def __init__(self, n_buckets=6000, n_hash=3, lowercase=True, ngram_range=(1, 1), analyzer="word") -> None:
        self.n_buckets = n_buckets
        self.n_hash = n_hash
        self.lowercase = lowercase
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.pipes = make_partial_union(*[
            HashingVectorizer(n_features=n_buckets//n_hash + i, ngram_range=ngram_range, analyzer=analyzer, binary=True) for i in range(n_hash)
        ])
    
    def fit(self, X, y=None):
        return self 
    
    def partial_fit(self, X, y=None, classes=None):
        return self 

    def transform(self, X, y=None):
        return self.pipes.transform(X)

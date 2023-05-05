import itertools as it 

import mmh3
import numpy as np
from scipy.sparse import lil_matrix, coo_matrix, csr_array
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer
from skpartial.pipeline import make_partial_union
from sklearn.utils import murmurhash3_32


class BloomVectorizer(BaseEstimator, TransformerMixin):
    """
    BloomVectorizer. Will hash each word in text multiple times using mmh3. 

    The benefit of this method is that it can accept generators in the `.transform()` method. 
    The downside is that it is a fair bit slower. There is also a larger chance of a hash 
    collision but the output embedding should also be somewhat more efficiently used.
    """
    def __init__(self, n_buckets=6000, n_hash=3) -> None:
        self.n_buckets = n_buckets
        self.n_hash = n_hash
        self.vectorizer = HashingVectorizer()
    
    def fit(self, X, y=None):
        return self 
    
    def partial_fit(self, X, y=None, classes=None):
        return self 

    def transform(self, X, y=None):
        X_tfm = self.vectorizer.transform(X)
        coo_mat = coo_matrix(X_tfm)
        row_ind = np.concatenate([coo_mat.row for i in range(self.n_hash)])
        col_ind = np.concatenate([coo_mat.col % (self.n_buckets - 1) for i in range(self.n_hash)])
        ones = np.ones(row_ind.shape)
        return csr_array((ones, (row_ind, col_ind)), shape=(len(X), self.n_buckets))


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

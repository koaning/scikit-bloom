import hashlib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from skpartial.pipeline import make_partial_union
from sklearn.feature_extraction.text import HashingVectorizer
from skbloom.skbloom import hash_to_cols

class BloomVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=5000, n_hash=3, lowercase=True) -> None:
        self.n_features = n_features
        self.n_hash = n_hash
        self.lowercase = lowercase
    
    def fit(self, X, y=None):
        return self 
    
    def partial_fit(self, X, y=None, classes=None):
        return self 

    def transform(self, X, y=None):
        row, col = [], []
        for i, x in enumerate(X):
            col_ids = hash_to_cols(x, self.n_hash, self.n_features)
            row_ids = [i for _ in col_ids]
            row.extend(row_ids)
            col.extend(col_ids)
        return csr_matrix((np.ones(len(row)), (row, col)), dtype=np.int8, shape=(len(X), self.n_features))


class SlowBloomVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=5000, n_hash=3, lowercase=True) -> None:
        self.n_features = n_features
        self.n_hash = n_hash
        self.lowercase = lowercase
    
    def fit(self, X, y=None):
        return self 
    
    def partial_fit(self, X, y=None, classes=None):
        return self 

    def transform(self, X, y=None):
        row, col = [], []
        for i, x in enumerate(X):
            for w in x.lower().split(" ") if self.lowercase else x.split(" "):
                for n_hash in range(self.n_hash):
                    row.append(i)
                    h = hashlib.md5(f"{w}{n_hash}".encode())
                    idx = int(h.hexdigest(), 16)
                    col.append(idx % self.n_features)
        return csr_matrix((np.ones(len(row)), (row, col)), dtype=np.int8, shape=(len(X), self.n_features))



class BloomishVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=2000, n_hash=3, lowercase=True, ngram_range=(1, 1), analyzer="word") -> None:
        self.n_features = n_features
        self.n_hash = n_hash
        self.lowercase = lowercase
        self.pipe = make_partial_union(*[
            HashingVectorizer(n_features=n_features + i, ngram_range=ngram_range, analyzer=analyzer) for i in range(n_hash)
        ])
    
    def fit(self, X, y=None):
        return self 
    
    def partial_fit(self, X, y=None, classes=None):
        return self 

    def transform(self, X, y=None):
        return self.pipe.transform(X)

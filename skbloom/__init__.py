import numpy as np
from scipy.sparse import csr_matrix, dok_array
from sklearn.base import BaseEstimator, TransformerMixin


class BloomVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_buckets=5000, n_hash=3, lowercase=True) -> None:
        self.n_buckets = n_buckets
        self.n_hash = n_hash
        self.lowercase = lowercase
    
    def fit(self, X, y=None):
        return self 

    def transform(self, X, y=None):
        row, col = [], []
        for i, x in enumerate(X):
            for w in x.lower().split(" ") if self.lowercase else x.split(" "):
                for _ in range(self.n_hash):
                    row.append(i)
                    col.append(hash(f"{_}-{w}") % self.n_buckets)
        return csr_matrix((np.ones(len(row)), (row, col)), dtype=np.int8)

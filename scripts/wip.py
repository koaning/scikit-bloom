import numpy as np
from scipy.sparse import csr_matrix 
import time
from sklearn.base import BaseEstimator, TransformerMixin
from datasets import load_dataset
import hashlib
import mmh3

dataset = load_dataset("clinc_oos", "plus")
texts = dataset['train']['text'] * 10


class MyBloomVectorizer(BaseEstimator, TransformerMixin):
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
            for w in x.lower().split(" "):
                h = mmh3.hash_bytes(w)
                for n_hash in range(self.n_hash):
                    row.append(i)
                    col.append(int.from_bytes(h[n_hash:n_hash + 10]) % self.n_features)
        return csr_matrix((np.ones(len(row)), (row, col)), dtype=np.int8, shape=(len(X), self.n_features))


for trial in range(3):
    tic = time.time()
    MyBloomVectorizer().fit_transform(texts)
    toc = time.time()
    print(f"Trial {trial}: {toc - tic}")

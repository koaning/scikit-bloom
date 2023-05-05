import time

import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline

from skbloom import BloomVectorizer, BloomishVectorizer

from scipy.sparse import csr_array, coo_matrix

from datasets import load_dataset

dataset = load_dataset("ag_news")
data = []
for n in [1_000, 5_000, 10_000, 30_000]:
    for tfm in [CountVectorizer(), BloomVectorizer(), BloomishVectorizer(n_hash=2), HashingVectorizer()]:
        pipe = make_pipeline(tfm, RidgeClassifier(class_weight="balanced"))
        X_train, y_train = dataset['train']['text'][:n], dataset['train']['label'][:n]
        X_test, y_test = dataset['test']['text'][:n], dataset['test']['label'][:n]
        
        tic_train = time.time()
        pipe.fit(X_train, y_train)
        toc_train = time.time()
        tic_infer = time.time()
        width = tfm.transform(["hi"]).shape[1]
        acc = np.mean(pipe.predict(X_test) == y_test)
        toc_infer = time.time()
        data.append({
            "n": n,
            "tfm": str(tfm), 
            "train_time": toc_train - tic_train, 
            "infer_time": toc_infer - tic_infer,
            "acc": acc,
            "width": width
        })
        print(data[-1])

import pandas as pd 

print(pd.DataFrame(data))
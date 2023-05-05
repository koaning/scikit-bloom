import time

import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline

from skbloom import BloomVectorizer, BloomishVectorizer

# for k in [1_000, 5_000, 10_000, 50_000]:
#     texts = [f"text-{i} {i} foo-{i} bar buzz{i}" for i in range(k)]

#     tic = time.time()
#     X = BloomVectorizer().transform(texts)
#     print(f"BloomVectorizer took {time.time() - tic}s for {k=} examples")

#     tic = time.time()
#     X = BloomishVectorizer().transform(texts)
#     print(f"BloomishVectorizer took {time.time() - tic}s {k=} examples")


from datasets import load_dataset

dataset = load_dataset("ag_news")
print(len(dataset['test']))
data = []
n = 10000
for n in [1_000, 5_000, 10_000, 30_000, 60_000]:
    for tfm in [CountVectorizer(), BloomishVectorizer(n_hash=1), BloomishVectorizer(n_hash=2), BloomishVectorizer(n_hash=3)]:
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
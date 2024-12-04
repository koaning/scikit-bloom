from uvtrick import Env 
from sklearn.datasets import make_regression
from datasets import load_dataset


dataset = load_dataset("clinc_oos", "plus")
texts = dataset['train']['text'] * 10

def bench(texts):
    from sklearn.feature_extraction.text import HashingVectorizer
    from time import time
    tic = time()
    HashingVectorizer().fit_transform(texts)
    toc = time()
    return toc - tic

for version in ["1.4", "1.5"]:
    for i in range(4):
        timed = Env(f"scikit-learn=={version}").run(bench, texts)
        print(version, timed)
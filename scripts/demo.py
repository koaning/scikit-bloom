import time
from datasets import load_dataset
from skbloom import BloomVectorizer, BloomishVectorizer, SlowBloomVectorizer, OhMyBloomVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from skbloom.skbloom import hash_to_cols

print(hash_to_cols("haha this is great", n_hashes=3, n_buckets=40))



dataset = load_dataset("clinc_oos", "plus")
texts = dataset['train']['text'] * 10

print("Original benchmark")
trials = [BloomVectorizer(n_features=10_000), 
          BloomishVectorizer(n_features=10_000), 
          SlowBloomVectorizer(n_features=10_000), 
          HashingVectorizer(n_features=10_000), 
          OhMyBloomVectorizer(n_features=10_000)]
for trial in trials:
    tic = time.time()
    trial.fit_transform(texts)
    toc = time.time()
    print(f"{trial.__class__.__name__}: {toc - tic}")


# print("Comparing to the standard sklearn implementation")
# for feats in [3000, 5000, 10000, 20000, 100_000, 200_000, 1_000_000]:
#     trials = [BloomVectorizer(n_hash=1, n_features=feats), HashingVectorizer(n_features=feats), OhMyBloomVectorizer(n_hash=1, n_features=feats)]
#     for trial in trials:
#         tic = time.time()
#         trial.fit_transform(texts)
#         toc = time.time()
#         print(f"{feats}: {trial.__class__.__name__}: {toc - tic}")

import time
from skbloom import BloomVectorizer, BloomishVectorizer

for k in [1_000, 5_000, 10_000, 50_000]:
    texts = [f"text-{i} {i} foo-{i} bar buzz{i}" for i in range(k)]

    tic = time.time()
    X = BloomVectorizer().transform(texts)
    print(f"BloomVectorizer took {time.time() - tic}s for {k=} examples")

    tic = time.time()
    X = BloomishVectorizer().transform(texts)
    print(f"BloomishVectorizer took {time.time() - tic}s {k=} examples")

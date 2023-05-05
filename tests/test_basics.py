import pytest 
from skbloom import BloomVectorizer, BloomishVectorizer


@pytest.mark.parametrize("vectorizer", [BloomVectorizer, BloomishVectorizer])
@pytest.mark.parametrize("n_hash", [1, 2, 3])
@pytest.mark.parametrize("n_buckets", [1000, 2000, 6000])
def test_n_hash(vectorizer, n_hash, n_buckets):
    texts = ["hello", "word", "thing"]
    X = vectorizer(n_hash=n_hash, n_buckets=n_buckets).fit_transform(texts)
    for x in X:
        assert x.shape[1] >= n_buckets
        assert x.sum() == n_hash


@pytest.mark.parametrize("vectorizer", [BloomVectorizer, BloomishVectorizer])
def test_can_handle_generator(vectorizer):
    texts = (f"text-{i}" for i in range(10))
    X = vectorizer().transform(texts)
    assert X.shape[0] == 10
    assert X.sum() == 30

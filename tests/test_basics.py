import pytest 
from skbloom import BloomVectorizer, BloomishVectorizer


@pytest.mark.parametrize("vectorizer", [BloomVectorizer, BloomishVectorizer])
def test_n_hash(vectorizer):
    texts = ["hello", "word", "thing"]
    X = vectorizer().fit_transform(texts)
    for x in X:
        print(x)
        assert x.sum() == 3

import pytest 
from skbloom import BloomVectorizer, SlowBloomVectorizer, BloomishVectorizer


@pytest.mark.parametrize("n_feats", [1_000, 2_000, 10_000])
@pytest.mark.parametrize("method", [BloomVectorizer, SlowBloomVectorizer])
@pytest.mark.parametrize("n_hash", [1, 2, 3])
def test_size_output(n_feats, method, n_hash):
    tfm = method(n_features=n_feats, n_hash=n_hash)
    X = tfm.fit_transform(["this is some data yeah"])
    assert X.shape[1] == n_feats
    assert X.sum() == (5 * n_hash)

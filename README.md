# scikit-bloom

Bloom tricks for text pipelines in scikit-learn. To learn more about this trick, check out [this blogpost](https://explosion.ai/blog/bloom-embeddings).

You can install it via:

```
python -m pip install scikit-bloom
```

And you can import the components via: 

```python
from skbloom import BloomVectorizer, BloomishVectorizer
```

In fairness, while this trick is interesting ... you _might_ be fine just using the `HashingVectorizer` that just comes with sklearn.

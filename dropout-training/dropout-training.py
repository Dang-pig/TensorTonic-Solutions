import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x=np.asarray(x)
    if rng is None:
        rng = np.random.default_rng()
    rand = rng.random(x.shape)
    dropout_pattern = np.zeros_like(x, dtype=np.float32)
    dropout_pattern[rand < (1 - p)] = 1 / (1 - p)
    return (x * dropout_pattern, dropout_pattern)
    pass
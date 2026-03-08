import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y, dtype=np.float32)
    _, y = np.unique(y, return_counts=True)
    y = y / y.sum()
    return -(y * np.where(y == 0, 0, np.log2(y))).sum().item()
    pass
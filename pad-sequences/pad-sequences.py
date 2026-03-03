import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    for seq in seqs:
        l = len(seq)
        if l < max_len:
            seq.extend([pad_value] * (max_len - l))
        else:
            seq[:] = seq[:max_len]
    # print(seqs)
    return seqs
    pass
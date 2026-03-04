import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pos = np.arange(seq_len)[:,None]
    i = base ** (2*np.arange(0, d_model, 2)/2/d_model)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(pos/i)
    pe[:, 1::2] = np.cos(pos/i[:pe[:, 1::2].shape[1]])
    return pe
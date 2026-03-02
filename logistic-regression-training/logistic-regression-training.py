import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    w, b = np.zeros_like(X[0]), 0.0
    print(w.shape)

    lr = np.float32(lr)
    n = X.shape[0]
    
    for st in range(steps):
        out = _sigmoid(X @ w + b)
        grad_w = X.T @ (out - y) / n
        grad_b = np.sum(out - y) / n
        
        print(grad_w, grad_b)
        w = w - lr * grad_w
        b = b - lr * grad_b

    return w, b
    pass
import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n = X.shape[1]
    W = np.zeros((1,n))
    y = y.reshape(-1,1)
    b= 0.0
    m = X.shape[0]

    for _ in range(steps):
        z= np.matmul(X,W.T)+b
        # print("h")
        y_hat = _sigmoid(z)
        delta = y_hat-y
        grad_W = np.matmul((y_hat-y).T,X)
        grad_b = np.sum((y_hat-y)).item()
        W = W - (lr/m)*grad_W
        b = b - (lr/m)*grad_b
    print(W)
    print(b)
    print("here")
    print(np.matmul(X,W.T)+b)
    return W.flatten(), b
        
        
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
    x=np.array(X)
    y=np.array(y).reshape(-1,1)
    w=np.zeros((x.shape[-1],1))
    b=0.0
    # print(w,b)

    for _ in range(steps):
        y_hat = _sigmoid(np.matmul(x,w)+b)
        del_l = y_hat-y
        w = w-lr*np.matmul(x.T,del_l)
        b = b-lr*np.sum(del_l)
    print(w)
    print(b)
    return w.reshape(-1),b
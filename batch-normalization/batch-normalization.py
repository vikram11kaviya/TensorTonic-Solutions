import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    x= np.array(x)
    xd = x.ndim
    xs = tuple([i for i in range(xd) if i!=1])
    mean_ = np.mean(x,axis=xs, keepdims=True)
    # print(x.shape)
    # print(mean.shape)
    view_shape = [1] * xd
    view_shape[1] = -1 # -1 tells NumPy to put the 'C' dimension here
    gamma = np.array(gamma).reshape(view_shape)
    beta = np.array(beta).reshape(view_shape)
    var_ = np.var(x,axis=xs, keepdims=True)
    # var = np.mean(np.square(x-mean),axis=0)
    x_ = (x-mean_)/np.sqrt(var_+eps)
    y = gamma*x_+beta
    return y
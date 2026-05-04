import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Write code here
    X = np.array(X)
    y =np.array(y).reshape(-1,1)
    # print(X.shape,y.shape)
    a = np.linalg.inv(np.matmul(X.T,X))
    b = np.matmul(X.T,y)
    # print(a.shape,b.shape)
    t=np.matmul(a,b)
    # print(t.shape)
    t= t.reshape(-1)
    # print(t.shape)
    return t
    # return np.matmul(a,b)
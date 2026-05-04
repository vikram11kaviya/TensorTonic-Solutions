import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    a = np.array(A)
    try:
        return np.linalg.inv(a)
    except:
        return None

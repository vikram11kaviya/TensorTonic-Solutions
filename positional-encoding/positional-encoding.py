import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    positions = np.arange(seq_len)
    inds = 2*np.arange((d_model+1)//2)

    denoms = np.power(base,inds/d_model)

    positions = positions.reshape(-1,1)
    pe_temp = positions/denoms
    # print(pe_temp.shape)

    sin_e = np.sin(pe_temp)
    cos_e = np.cos(pe_temp)

    pe = np.empty((seq_len,d_model), dtype=float)

    pe[:,::2]=sin_e
    pe[:,1::2]=cos_e[:,:(d_model)//2]
    return pe
    # print(inds)
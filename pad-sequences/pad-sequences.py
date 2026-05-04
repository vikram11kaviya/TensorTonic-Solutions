import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if max_len is None:
        if seqs:
            L = max(len(seq) for seq in seqs)
        else:
            L=0
    else:
        L=max_len
    N = len(seqs)
    padded = np.full(shape=(N,L), fill_value=pad_value)

    for i in range(N):
       padded[i,:min(len(seqs[i]),L)]=seqs[i][:L]
    return padded
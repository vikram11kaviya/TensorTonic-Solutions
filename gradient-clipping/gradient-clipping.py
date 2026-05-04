import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write codnoe here
    # print(g)
    g = np.array(g)
    if max_norm<=0:
        return g
    g_norm = np.sqrt(np.sum(g**2))
    print(g_norm)
    if g_norm<=max_norm:
        return g
    return g*(max_norm/g_norm)
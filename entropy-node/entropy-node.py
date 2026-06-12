import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    values, counts = np.unique(y, return_counts = True)

    prop = counts / len(y)

    return - np.sum(prop*np.log2(prop))
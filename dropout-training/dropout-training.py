import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    X_array = np.array(x)

    if rng is not None:
        probs = rng.random(X_array.shape)
    else:
        probs = np.random.random(X_array.shape)
        
    mask = probs > p

    scale_factor =  1 / (1 - p)
    
    numeric_mask = mask * scale_factor
    
    X_array_dropout =  X_array * numeric_mask

    return (X_array_dropout, numeric_mask)
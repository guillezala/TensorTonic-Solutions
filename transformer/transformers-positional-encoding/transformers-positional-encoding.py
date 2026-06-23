import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    seq = np.arange(seq_length).reshape(-1, 1)
    pos_encoding = np.zeros((seq_length, d_model))

    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
    pos_encoding[:,0::2] = np.sin(seq * div_term)
    pos_encoding[:,1::2] = np.cos(seq * div_term)

    return pos_encoding
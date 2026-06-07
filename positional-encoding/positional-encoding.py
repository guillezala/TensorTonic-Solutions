import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    seq_len_arr = np.array([i for i in range(seq_len)])

    result = []

    for i in range(d_model):
        if i % 2 == 0:
            PE = np.sin(seq_len_arr/(base**(i/d_model)))
            print(i, "sin")
        else:
            PE = np.cos(seq_len_arr/(base**((i-1)/d_model)))
            print(i, "cos")

        result.append(PE)

    return np.array(result).T
import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    new_seqs = []
    if max_len is None:
        max_len = max([len(seq) for seq in seqs])
    
    for seq in seqs:
        seq_len=len(seq)

        if seq_len >= max_len:
            new_seq = seq[:max_len]
        else:
            diff = max_len - seq_len
            new_seq = seq + [pad_value for i in range(diff)]

        new_seqs.append(new_seq)

    return new_seqs
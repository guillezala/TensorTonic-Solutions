import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    print(num_heads)
    batch_size, seq_length, d_model = Q.shape

    print(batch_size, seq_length, d_model)
    
    d_k = d_model // num_heads

    Q_proj = np.dot(Q, W_q)
    K_proj = np.dot(K, W_k)
    V_proj = np.dot(V, W_v)

    Q_heads = Q_proj.reshape(batch_size, seq_length, num_heads, d_k).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(batch_size, seq_length, num_heads, d_k).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(batch_size, seq_length, num_heads, d_k).transpose(0, 2, 1, 3)

    scores = np.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

    soft = softmax(scores)

    att = np.matmul(soft, V_heads).transpose(0, 2, 1, 3)

    concat_heads = att.reshape(batch_size, seq_length, d_model)

    output = np.matmul(concat_heads, W_o)

    return output
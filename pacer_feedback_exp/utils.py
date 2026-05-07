import numpy as np
import pandas as pd

def softmax(x, axis=0):
    """
    softmax turns arbitrary real numbers into a probability distribution. 

    # we would need this to convert aspect scores phi into proper preference vector probabilities.

    Example:
        x = [2, 1, 0]
        softmax(x) -> [0.66, 0.24, 0.09] (roughly)

    Why subtract max?
        Prevents exp(large number) overflow.
    """
        
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / (ex.sum(axis=axis, keepdims=True) + 1e-12)


def sigmoid(x):
    """
    Sigmoid function maps real numbers to (0,1). Useful for converting utilities into probabilities of positive feedback.
    """
    return 1.0 / (1.0 + np.exp(-x))


def cosine_similarity(a, b, eps=1e-12):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + eps))


def kl_divergence(p, q, eps=1e-12):
    """
    KL divergence between two distributions p and q. Both should be non-negative and sum to 1.
    KL(p || q) = sum_i p_i * (log(p_i) - log(q_i)))
    """
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))

def normalize_simplex(w, eps=1e-12):
    """
    Normalize a vector w to lie in the probability simplex: non-negative and sums to 1.
    """
    w = np.asarray(w, dtype=np.float32)
    w = np.maximum(w, 0.0)
    s = w.sum()

    if s <= eps:
        return np.ones_like(w, dtype=np.float32) / len(w)

    return w / s


# def normalize_simplex(w, eps=1e-12):
#     """
#     Normalize a vector w to lie in the probability simplex (non-negative, sums to 1).
#     """
#     w = np.asarray(w, dtype=np.float32)
#     return w / (w.sum() + eps)

def ensure_global_index(df_sent: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df_sent has a 'global_idx' column that uniquely identifies each sentence across the dataset.
    If not present, create it as a simple range index.
    """
    df_sent = df_sent.copy()
    if "global_idx" not in df_sent.columns:
        df_sent["global_idx"] = np.arange(len(df_sent))
    return df_sent


def ensure_length_col(df_sent: pd.DataFrame, text_col="review_text", length_col="len") -> pd.DataFrame:
    """
    Ensure df_sent has a 'len' column that contains the length of each sentence.
    If not present, create it based on the 'review_text' column.
    """
    df_sent = df_sent.copy()
    if length_col not in df_sent.columns:
        df_sent[length_col] = df_sent[text_col].fillna("").astype(str).map(len)
    return df_sent


def get_phi_matrix(df_sent: pd.DataFrame, phi_cols):
    """
    Extract the aspect matrix Phi for the candidate sentences.
    """
    return df_sent[list(phi_cols)].to_numpy(dtype=np.float32)

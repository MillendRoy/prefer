import numpy as np
import pandas as pd

from pacer_feedback_exp.utils import softmax, normalize_simplex, ensure_global_index, ensure_length_col


def compute_sentence_utilities(cand: pd.DataFrame, phi_cols, w_u: np.ndarray, lam=0.0, len_col="len"):
    """
    For a product p, we have candidate sentences.

    Compute for each sentence i:
      base_i = w_u^T phi_i   (how much it matches the user's aspects) (Section 3.1 Rel_i(u,p))

    Optionally penalize long sentences:
      base_i -= lam * len_i / mean_len

    This is the relevance score U_i for each candidate sentence.
    
    Returns:
      U   : (n,) utility/relevance scores
      Phi : (n,K) aspect matrix for candidates
    """
    Phi = cand[list(phi_cols)].to_numpy(dtype=np.float32) # shape : (n = number of candidates, K = number of aspects)
    w_u = np.asarray(w_u, dtype=np.float32) # shape : (K,) user preference vector

    rel = Phi @ w_u
    if lam != 0.0:
        lengths = cand[len_col].to_numpy(dtype=np.float32)
        U = rel - lam * lengths
    else:
        U = rel.copy()

    return U.astype(np.float32), Phi


def compute_alpha(
    selected_scores: np.ndarray,
    mode: str = "utility_softmax",
    tau_alpha: float = 1.0,
    rank_decay: float = 0.3,
):
    """
    selected_scores should already be aligned with selected_df order.
    mode: how to compute alpha weights over selected sentences for aspect aggregation
      - "uniform": equal weights
      - "utility_softmax": softmax over the utility scores of the selected sentences (higher utility -> higher weight)
      - "rank_softmax": softmax over the rank of the sentences in the selection order (earlier selected -> higher weight)
      - "blended": combine utility and rank for alpha computation
    tau_alpha: temperature for utility-based softmax (if applicable)
    rank_decay: decay factor for rank-based weighting (if applicable)
    Returns:
      alpha: (n_selected,) weights for the selected sentences, aligned with selected_df order
    """
    selected_scores = np.asarray(selected_scores, dtype=np.float32)
    n = len(selected_scores)

    if n == 0:
        return np.zeros(0, dtype=np.float32)

    if mode == "uniform":
        return np.ones(n, dtype=np.float32) / n

    if mode == "utility_softmax":
        return softmax(tau_alpha * selected_scores, axis=0)

    if mode == "rank_softmax":
        ranks = np.arange(n, dtype=np.float32)
        return softmax(-rank_decay * ranks, axis=0)

    if mode == "blended":
        ranks = np.arange(n, dtype=np.float32)
        raw = tau_alpha * selected_scores - rank_decay * ranks
        return softmax(raw, axis=0)

    raise ValueError(f"Unknown alpha mode: {mode}")


def prepare_candidates(
    df_sent: pd.DataFrame,
    product_id: str,
    phi_cols,
    text_col="review_text",
    len_col="len",
):
    """
    Filter dataframe for the given product_id and ensure necessary columns are present.
    Necessary columns as in :
        - global index (for tracking sentences across rounds)
        - length column (for length-based penalties if needed)
        Returns the candidate dataframe for the product.
    """
    df_sent = ensure_global_index(df_sent)
    df_sent = ensure_length_col(df_sent, text_col=text_col, length_col=len_col)

    cand = df_sent[df_sent["parent_asin"] == product_id].copy()
    if len(cand) == 0:
        return None

    return cand
from typing import Optional, Sequence
import numpy as np
from .utils import softmax, normalize_simplex, get_phi_matrix

# Section 3.3 : importance weighted aspect profile with feedback weighting
def aspect_profile(phi_rows: np.ndarray, idx: np.ndarray, alpha: Optional[np.ndarray] = None):
    """
    Compute aspect profile z_t for a selected set S_t (importance weighted).

    Inputs:
      phi_rows: (N, K) full aspect matrix for all rows in your dataset
      idx: list/array of indices that represent the selected set S_t
      alpha: optional weights per selected sentence (importance weighting)

    Output:
      z_t: (K,) vector representing aspect distribution of that selection
    """
    idx = np.asarray(idx, dtype=int)
    if len(idx) == 0:
        raise ValueError("Empty S_t.")

    if alpha is None:
        return phi_rows[idx].mean(axis=0) # uniform averaging section 3.3

    alpha = np.asarray(alpha, dtype=np.float32)
    alpha = alpha / (alpha.sum() + 1e-12) # ensures summation of alpha_i is 1, so that z_t remains in a comparable scale to the uniform case
    return (phi_rows[idx] * alpha[:, None]).sum(axis=0) # weighted average of aspect profiles for selected sentences (importance weighting section 3.3)


def estimate_user_pref_from_history(history, phi_rows, beta=5.0, eps=1e-8, prior=None):
    """
    Estimate the user preference vector w_u from interaction history using an empirical Boltzmann estimation.

    history: list of events, each event dict contains:
      {
        "S_idx": np.array([...])   # indices of selected sentences (S_t)
        "f": float                 # feedback in [0,1]
        "alpha": optional weights over S_t
      }

    phi_rows: (N, K) full aspect matrix
    beta: preference sharpness for softmax
    prior: optional prior distribution over aspects (K,)

    Output:
      w_u: (K,) user preference vector in simplex
    """
        
    K = phi_rows.shape[1]

    # if no history: return prior or uniform
    if history is None or len(history) == 0:
        if prior is None:
            return np.ones(K, dtype=np.float32) / K
        return normalize_simplex(prior)

    # numerator and denominator for ztilde_u
    num = np.zeros(K, dtype=np.float32)
    den = 0.0
    
    # build: ztilde_u = sum_t f_t z_t / sum_t f_t
    # for each event in history, compute aspect profile z_t and accumulate weighted by feedback f_t
    for ev in history:
        f = float(ev.f if hasattr(ev, "f") else ev["f"])
        S_idx = np.asarray(ev.S_idx if hasattr(ev, "S_idx") else ev["S_idx"], dtype=int)
        alpha = ev.alpha if hasattr(ev, "alpha") else ev.get("alpha", None)

        z_t = aspect_profile(phi_rows, S_idx, alpha=alpha)
        num += f * z_t
        den += f

    z_tilde = num / (den + eps)
    w = softmax(beta * z_tilde, axis=0)

    if prior is not None:
        prior = normalize_simplex(prior)
        w = 0.9 * w + 0.1 * prior
        w = normalize_simplex(w)

    return w


def estimate_user_pref_from_own_text(df_sent, user_id, phi_cols: Sequence[str], beta=5.0):
    """
    Bootstraps w_u directly from the user's own sentences/reviews  if present 
    """
    sub = df_sent[df_sent["user_id"] == user_id]
    K = len(phi_cols)

    if len(sub) == 0:
        return np.ones(K, dtype=np.float32) / K

    Phi = get_phi_matrix(sub, phi_cols)
    ztilde = Phi.mean(axis=0)
    return softmax(beta * ztilde, axis=0)


def make_synthetic_true_preference(df_sent, user_id, phi_cols, beta=8.0, noise_scale=0.03, seed=42):
    """
    Hidden oracle w* as noisy version of w0.
    For feedback generation : w^{*} as true user preference vector but hidden (oracle) to generate synthetic feedback. 
    """
    rng = np.random.default_rng(seed)

    w0 = estimate_user_pref_from_own_text(
        df_sent=df_sent,
        user_id=user_id,
        phi_cols=phi_cols,
        beta=beta,
    )

    noisy_logits = np.log(w0 + 1e-12) + noise_scale * rng.standard_normal(len(w0))
    w_true = softmax(noisy_logits, axis=0)
    return normalize_simplex(w_true)
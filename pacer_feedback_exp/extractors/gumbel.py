import numpy as np
from .base import BaseExtractor


def sample_gumbel(shape, rng=None):
    """
    Generate Gumbel(0,1) noise. Section 3.1.2  describes g_{t',j}

    If U ~ Uniform(0,1), then:
      G = -log(-log(U)) is ensures Gumbel distribution
    """
    rng = np.random.default_rng() if rng is None else rng
    U = rng.uniform(low=1e-12, high=1.0 - 1e-12, size=shape)
    return -np.log(-np.log(U))


def gumbel_priority_order(U, tau_ext=1.0, rng=None):
    """
    Create a random ordering of sentences where higher-utility sentences
    tend to appear earlier.

    # section 3.1.2 describes this process of creating a random priority ordering using gumbel-perturbed utilities.
    Steps:
      1) sample g_i ~ Gumbel
      2) xi_i = tau_ext * U_i + g_i
      3) sort by xi in descending order

    Output:
      perm: indices sorted high-to-low
      xi: perturbed scores
    """
    if rng is None:
        rng = np.random.default_rng()
    # rng = np.random.default_rng(seed)
    g = sample_gumbel(len(U),rng=rng)

    # what is the effect of tau_ext here?
    xi = tau_ext * np.asarray(U, dtype=np.float32) + g
    perm = np.argsort(-xi)
    return perm, xi


def gumbel_priority_greedy_select(
    df_candidates,
    U,
    k=8,
    L=800,
    len_col="len",
    tau_ext=1.0,
    rng=None,
):
    """
    Implements algorithm 2 : section 3.1.2

    - create random priority ordering using gumbel-perturbed utilities
    - scan in that order
    - add sentence if it doesn't violate constraints:
        (1) |S| <= k
        (2) total length <= L

    Output:
      selected_local_idx: indices (within df_candidates) that were selected
      perm: the full ordering
      xi: reminder scores
      total_len: length used
    """
    perm, xi = gumbel_priority_order(U, tau_ext=tau_ext, rng=rng)

    selected = []
    total_len = 0
    lens = df_candidates[len_col].to_numpy(dtype=int)

    for j in perm:
        if len(selected) >= k:
            break
        if total_len + int(lens[j]) <= L:
            selected.append(j)
            total_len += int(lens[j])

    return np.array(selected, dtype=int), perm, xi, total_len


class GumbelExtractor(BaseExtractor):
    def __init__(
        self,
        *,
        k=8,
        L=800,
        seed=0,
        text_col="review_text",
        len_col="len",
        alpha_mode="utility_softmax",
        tau_alpha=1.0,
        rank_decay=0.3,
    ):
        """
        Gumbel-Top-k extractor that implements the gumbel-based selection strategy 
        described in Section 3.1.2 of the paper.
        k: number of sentences to select
        L: total length constraint for selected sentences
        seed: random seed for reproducibility of gumbel noise
        text_col: column name in df_sent that contains the sentence text
        len_col: column name in df_sent that contains the sentence length (for length-constrained extraction)
        alpha_mode: how to compute alpha weights over selected sentences for aspect aggregation ()
        tau_alpha: temperature for computing alpha weights
        rank_decay: decay factor for utility-based alpha weighting (if applicable)
        """
        super().__init__(
            text_col=text_col,
            len_col=len_col,
            alpha_mode=alpha_mode,
            tau_alpha=tau_alpha,
            rank_decay=rank_decay,
        )
        self.k = k
        self.L = L
        self.seed = seed
        self.rng = np.random.default_rng(seed)


    def _select_local_indices(self, cand, utility_scores, phi, w_u, tau_ext, tau_alpha):
        sel_local_idx, perm, xi, total_len = gumbel_priority_greedy_select(
            cand,
            utility_scores,
            k=self.k,
            L=self.L,
            len_col=self.len_col,
            tau_ext=tau_ext,
            rng=self.rng,
        )
        return {
            "selected_local_idx": sel_local_idx,
            "selection_order_scores": xi,
            "total_len": total_len,
            "meta": {
                "perm": perm,
                "xi": xi,
            }
        }

    def _finalize_selected_df(self, selected_df, cand, selection_output):
        """
        Finalize the selected dataframe by adding gumbel scores and sorting by them.
        """
        xi = selection_output["meta"]["xi"]
        selected_local_idx = selection_output["selected_local_idx"]
        selected_df = selected_df.copy()
        selected_df["gumbel_score"] = xi[selected_local_idx]
        selected_df = selected_df.sort_values("gumbel_score", ascending=False).copy()
        return selected_df

    def _alpha_scores_for_selected_df(self, selected_df):
        # choice: use utility rather than gumbel_score for stable alpha
        return selected_df["utility"].to_numpy(dtype=np.float32)

    def _collect_scores(self, selected_df, selection_output):
        return {
            "utility": selected_df["utility"].to_numpy(dtype=np.float32),
            "gumbel_score": selected_df["gumbel_score"].to_numpy(dtype=np.float32),
        }
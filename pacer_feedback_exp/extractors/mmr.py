import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseExtractor
from pacer_feedback_exp.utils import normalize_simplex, softmax


def build_sentence_vectors_tfidf(texts, max_features=5000):
    """
    Lightweight baseline for sentence vectors s_i using TF-IDF.
    Produces S: (n, d).

    # help me understand this function:
        - We use sklearn's TfidfVectorizer to convert raw text sentences into TF-IDF vectors.
        - max_features limits the vocabulary size to the top max_features terms.
        - stop_words="english" removes common English words that are not informative.
        - We handle None values by converting them to empty strings.
        - Finally, we convert the sparse matrix output to a dense numpy array of type float32.
        - The resulting S matrix can be used for cosine similarity or other downstream tasks.
        - This is a simple baseline; more sophisticated methods could use pretrained embeddings or fine-tuning.
    """
    texts = ["" if t is None else str(t) for t in texts]
    vec = TfidfVectorizer(max_features=max_features, stop_words="english")
    X = vec.fit_transform(texts)
    return X.toarray().astype(np.float32)


def cosine_sim_matrix(S):
    """
    Compute cosine similarity matrix for sentence vectors S.

    S: (n, d)
    returns Sim: (n, n) with Sim[i,j] = cosine(s_i, s_j)
    """
    return cosine_similarity(S)

def kl_target_to_profile(
    target: np.ndarray,
    profile: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Computes D_KL(target || profile).

    This is useful when target is the user preference vector w_u
    and profile is the selected summary aspect profile z(S).

    Both vectors are smoothed and normalized for numerical stability.
    """
    target = normalize_simplex(target, eps=eps)
    profile = normalize_simplex(profile, eps=eps)

    return float(np.sum(target * np.log(target / profile)))


def mmr_select_indices(
    Rel: np.ndarray,
    Sim: np.ndarray,
    lengths: np.ndarray,
    k: int = 8,
    L: int = 800,
    lam_mmr: float = 0.7,
    Phi: np.ndarray | None = None,
    w_u: np.ndarray | None = None,
    beta_kl: float = 0.0,
    eps_kl: float = 1e-12,
    tau_alpha: float = 1.0,
):
    """
    Implements Algorithm MMR-Select from your writeup:

      Δ(j | S) = λ Rel[j] - (1-λ) max_{i in S} Sim[i,j]

    Optional KL-aware gain:
        Δ(j | S)
        =
        λ Rel[j]
        - (1-λ) max_{i in S} Sim[i,j]
        + beta_kl * [ KL(w_u || z(S)) - KL(w_u || z(S ∪ {j})) ]

    with feasibility constraints:
      |S| <= k,  sum(lengths[i]) <= L

    Returns selected indices (local to candidate set).
    """
    n = len(Rel) # number of candidate sentences for this product
    Rel = np.asarray(Rel, dtype=np.float32)
    Sim = np.asarray(Sim, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.int32)

    use_kl = beta_kl > 0.0 and Phi is not None and w_u is not None
    if use_kl:
        Phi = np.asarray(Phi, dtype=np.float64)
        w_u = normalize_simplex(np.asarray(w_u, dtype=np.float64), eps=eps_kl)

        if Phi.ndim != 2:
            raise ValueError("Phi must be a 2D array with shape (n_candidates, n_aspects).")
        if Phi.shape[0] != n:
            raise ValueError("Phi must have the same number of rows as Rel.")
        if Phi.shape[1] != len(w_u):
            raise ValueError("Phi.shape[1] must match len(w_u).")
        
        # Normalize each sentence aspect vector.
        Phi = np.vstack([normalize_simplex(row, eps=eps_kl) for row in Phi])
        


    selected = []
    used_len = 0
    max_sim_to_S = np.zeros(n, dtype=np.float32)

    # Running sum of selected aspect vectors.
    phi_sum = None
    if use_kl:
        phi_sum = np.zeros(Phi.shape[1], dtype=np.float64)


    # Useful diagnostics.
    greedy_scores = []
    kl_values_after = []
    kl_improvements = []


    for _ in range(k):
        best_j = None
        best_gain = -1e18
        best_kl_after = None
        best_kl_improvement = 0.0

        if use_kl and len(selected) > 0:
            weights_current = softmax(tau_alpha*np.asarray(Rel[selected], dtype=np.float64))
            z_current = weights_current @ Phi[selected]
            
            # z_current = phi_sum / len(selected)
            kl_current = kl_target_to_profile(w_u, z_current, eps=eps_kl)
        else:
            kl_current = None

        for j in range(n):
            if j in selected:
                continue
            if used_len + int(lengths[j]) > L:
                continue

            redundancy = 0.0 if len(selected) == 0 else float(max_sim_to_S[j])
            gain = lam_mmr * float(Rel[j]) - (1.0 - lam_mmr) * redundancy

            kl_after = None
            kl_improvement = 0.0

            if use_kl:
                candidate_indices = selected + [j]
                weights = softmax(tau_alpha*np.asarray(Rel[candidate_indices], dtype=np.float64))
                z_after = weights @ Phi[candidate_indices]
                # z_after = (phi_sum + Phi[j]) / (len(selected) + 1)
                kl_after = kl_target_to_profile(w_u, z_after, eps=eps_kl)

                # For the first selected sentence, there is no existing z(S).
                # So we use -KL(w || z_after) directly.
                if kl_current is None:
                    kl_improvement = -kl_after
                else:
                    kl_improvement = kl_current - kl_after

                gain += beta_kl * kl_improvement            

            if gain > best_gain:
                best_gain = gain
                best_j = j
                best_kl_after = kl_after
                best_kl_improvement = kl_improvement            

        if best_j is None:
            break

        selected.append(best_j)
        used_len += int(lengths[best_j])
        max_sim_to_S = np.maximum(max_sim_to_S, Sim[best_j])

        if use_kl:
            phi_sum += Phi[best_j]

        greedy_scores.append(best_gain)
        kl_values_after.append(best_kl_after)
        kl_improvements.append(best_kl_improvement)

    meta = {
        "greedy_scores": np.array(greedy_scores, dtype=np.float32),
        "kl_values_after": np.array(
            [np.nan if v is None else v for v in kl_values_after],
            dtype=np.float32,
        ),
        "kl_improvements": np.array(kl_improvements, dtype=np.float32),
    }        

    return np.array(selected, dtype=int), used_len,meta


class MMRExtractor(BaseExtractor):
    def __init__(
        self,
        *,
        k=8,
        L=800,
        lam_mmr=0.7,
        beta_kl=0.0,
        eps_kl=1e-12,
        sim_method="tfidf",
        tfidf_max_features=5000,
        text_col="review_text",
        len_col="len",
        alpha_mode="utility_softmax",
        tau_alpha=1.0,
        rank_decay=0.3,
    ):
        super().__init__(
            text_col=text_col,
            len_col=len_col,
            alpha_mode=alpha_mode,
            tau_alpha=tau_alpha,
            rank_decay=rank_decay,
        )
        self.k = k # max number of sentences to select
        self.L = L # max total length of selected sentences
        self.lam_mmr = lam_mmr
        self.beta_kl = beta_kl
        self.eps_kl = eps_kl
        self.sim_method = sim_method
        self.tfidf_max_features = tfidf_max_features

    def _select_local_indices(self, cand, utility_scores, phi, w_u, tau_ext, tau_alpha):
        if self.sim_method != "tfidf":
            raise ValueError("Only sim_method='tfidf' is currently supported.")

        texts = cand[self.text_col].fillna("").astype(str).tolist()
        S = build_sentence_vectors_tfidf(texts, max_features=self.tfidf_max_features)
        Sim = cosine_sim_matrix(S)
        lengths = cand[self.len_col].to_numpy(dtype=np.int32)

        sel_idx, used_len, mmr_meta = mmr_select_indices(
            Rel=utility_scores,
            Sim=Sim,
            lengths=lengths,
            k=self.k,
            L=self.L,
            lam_mmr=self.lam_mmr,
            Phi=phi,
            w_u=w_u,
            beta_kl=self.beta_kl,
            eps_kl=self.eps_kl,
            tau_alpha=tau_alpha,
        )

        return {
            "selected_local_idx": sel_idx,
            "selection_order_scores": utility_scores,
            "total_len": used_len,
            "meta": {
                "Sim": Sim,
                "mmr_meta":mmr_meta,
            }
        }

    def _finalize_selected_df(self, selected_df, cand, selection_output):
        """
        Finalize the selected dataframe by adding MMR scores and sorting by them.
        """
        Sim = selection_output["meta"]["Sim"]
        mmr_meta = selection_output["meta"].get("mmr_meta", {})
        selected_local_idx = selection_output["selected_local_idx"]
        selected_df = selected_df.copy()

        # selected order is greedy order already
        selected_df["Rel"] = selected_df["utility"].to_numpy(dtype=np.float32)

        red = []
        for t, j in enumerate(selected_local_idx):
            if t == 0:
                red.append(0.0)
            else:
                prev = selected_local_idx[:t]
                red.append(float(np.max(Sim[prev, j])))
        selected_df["max_sim_to_prev"] = red
        if "greedy_scores" in mmr_meta:
            selected_df["mmr_greedy_score"] = mmr_meta["greedy_scores"]

        if "kl_values_after" in mmr_meta:
            selected_df["kl_after_selection"] = mmr_meta["kl_values_after"]

        if "kl_improvements" in mmr_meta:
            selected_df["kl_improvement"] = mmr_meta["kl_improvements"]

        return selected_df

    def _alpha_scores_for_selected_df(self, selected_df):
        return selected_df["utility"].to_numpy(dtype=np.float32)

    def _collect_scores(self, selected_df, selection_output):
        out = {
            "utility": selected_df["utility"].to_numpy(dtype=np.float32),
        }
        if "Rel" in selected_df.columns:
            out["Rel"] = selected_df["Rel"].to_numpy(dtype=np.float32)
        return out
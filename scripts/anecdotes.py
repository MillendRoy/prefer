import numpy as np
import pandas as pd

from pacer_feedback_exp.extractors.gumbel import GumbelExtractor
from pacer_feedback_exp.extractors.mmr import MMRExtractor,kl_target_to_profile
from pacer_feedback_exp.utils import get_phi_matrix, ensure_global_index, normalize_simplex, cosine_similarity
from pacer_feedback_exp.extractors.common import prepare_candidates

from pacer_feedback_exp.rewriter.pipeline import rewrite_selected_df
from pacer_feedback_exp.rewriter.models import fallback_summarizer

# -------------------------------------------------
# Config
# -------------------------------------------------
DATA_PATH = "data/train_with_aspect_scores_sentences.csv"
PRODUCT_ID = "B085BB7B1M"
USER_ID = "AG73BVBKUOH22USSFJA5ZWL7AKXA"   # only for metadata if needed

K_SELECT = 10 # number of sentences to select for summary (before rewriting)
L_BUDGET = 1000 # total length budget for selected sentences (before rewriting)
SEED = 42   # keep fixed for controlled anecdote

TEXT_COL = "review_text"
LEN_COL = "len"
LAMBDA_KL_GOODNESS = 0.2


# -------------------------------------------------
# Load data
# -------------------------------------------------
data = pd.read_csv(DATA_PATH)
data = ensure_global_index(data)
phi_cols = [c for c in data.columns if c.startswith("asp_")]
K = len(phi_cols) # number of aspects

phi_rows = get_phi_matrix(data, phi_cols)

cand = prepare_candidates(
    df_sent=data,
    product_id=PRODUCT_ID,
    phi_cols=phi_cols,
    text_col=TEXT_COL,
    len_col=LEN_COL,
)

if cand is None or len(cand) == 0:
    raise ValueError(f"No candidates found for product_id={PRODUCT_ID}")

print(f"Number of candidate sentences for product {PRODUCT_ID}: {len(cand)}")


# -------------------------------------------------
# Helper: construct synthetic user preference vectors
# -------------------------------------------------
def make_pref_vector(K, main_idx, second_idx=None, main_weight=0.85, second_weight=0.10):
    w = np.zeros(K, dtype=np.float32)
    if main_idx is None:
        w += 1.0 / K
    else:
        w[main_idx] = main_weight
        if second_idx is not None:
            w[second_idx] = second_weight

        remaining = 1.0 - w.sum()
        if remaining > 0:
            # spread tiny leftover mass uniformly over remaining aspects
            other_idx = [i for i in range(K) if i not in [main_idx, second_idx]]
            if len(other_idx) > 0:
                for i in other_idx:
                    w[i] = remaining / len(other_idx)

        w = w / w.sum()
    return w


# -------------------------------------------------
# Choose which two aspects to contrast
# Example: aspect 0 and aspect 1
# -------------------------------------------------
def goodness_score_with_kl(w_hat, z, lambda_kl=0.5, eps=1e-12):
    """
    Computes cosine alignment and optional KL-regularized goodness.

        G_cos(w_hat, z)
        =
        <w_hat, z> / (||w_hat||_2 ||z||_2)

        G_KL(w_hat, z)
        =
        G_cos(w_hat, z) - lambda_kl KL(w_hat || z)

    Returns:
        G_total: KL-regularized cosine goodness score
        G_cos: cosine alignment score
        KL: KL divergence penalty
    """
    w_hat = normalize_simplex(w_hat, eps=eps)
    z = normalize_simplex(z, eps=eps)

    G_cos = cosine_similarity(w_hat, z, eps=eps)
    KL = float(kl_target_to_profile(w_hat, z, eps=eps))
    G_total = float(G_cos - lambda_kl * KL)

    return G_total, G_cos, KL

# def goodness_score_with_kl(w_hat, z, lambda_kl=0.5, eps=1e-12):
#     """
#     Computes the KL-regularized goodness score:

#         G(w_hat, p)
#         =
#         <w_hat, z(p)> - lambda_kl KL(w_hat || z(p)).

#     Returns:
#         G_total: KL-regularized goodness score
#         G_align: plain alignment score
#         KL: KL divergence penalty
#     """
#     w_hat = normalize_simplex(w_hat, eps=eps)
#     z = normalize_simplex(z, eps=eps)

#     G_align = float(np.dot(w_hat, z))
#     KL = kl_target_to_profile(w_hat, z, eps=eps)
#     G_total = G_align - lambda_kl * KL 

#     G_total = float(G_total)/float(np.dot(w_hat, w_hat))  # normalize by self-alignment of w_hat for interpretability
#     G_align = float(G_align)/float(np.dot(w_hat, w_hat))  # normalize by self-alignment of w_hat for interpretability
#     KL = float(KL)/float(np.dot(w_hat, w_hat))  # normalize by self-alignment of w_hat for interpretability
#     return G_total, G_align, KL

ASPECT_1_IDX = 2
ASPECT_2_IDX = 0

user_profiles = {
    "user_pref_aspect_1": make_pref_vector(K, main_idx=ASPECT_1_IDX, main_weight=0.90),
    "user_pref_aspect_2": make_pref_vector(K, main_idx=ASPECT_2_IDX, main_weight=0.90),
    "user_pref_mixture": make_pref_vector(K, main_idx=ASPECT_1_IDX, second_idx=ASPECT_2_IDX, main_weight=0.49, second_weight=0.49),
    "generic_user": make_pref_vector(K, main_idx=None, main_weight=0.1), # generic user with no strong aspect preference
}


# -------------------------------------------------
# Fixed extractor for controlled comparison
# If you want fully deterministic comparison, you can later replace Gumbel with MMR
# or set tau_ext small/large depending on your intended behavior.
# -------------------------------------------------
# extractor = GumbelExtractor(
#     k=K_SELECT,
#     L=L_BUDGET,
#     seed=SEED,
#     alpha_mode="utility_softmax",
#     tau_alpha=1.5,
# )

extractor = MMRExtractor(
    k=K_SELECT,
    L=L_BUDGET,
    # seed=SEED,
    lam_mmr=0.9,
    beta_kl=LAMBDA_KL_GOODNESS,
    alpha_mode="utility_softmax",
    tau_alpha=20,
)


# -------------------------------------------------
# Run extraction for each user profile
# -------------------------------------------------
results = {}

for profile_name, w_u in user_profiles.items():
    out = extractor.extract(
        df_sent=data,
        user_id=USER_ID,
        product_id=PRODUCT_ID,
        phi_cols=phi_cols,
        w_u=w_u,
        phi_rows=phi_rows,
        utility_lam=0.0, # to penalize long sentences more, increase this lambda parameter to make the extractor prefer shorter sentences (which may lead to more concise summaries but potentially lower utility if important content is in longer sentences)
        tau_ext=10.0,   # moderate/high: more utility-driven, less random- doesnot matter in MMR
    )

    if out is None:
        print(f"{profile_name}: no output")
        continue

    selected_df = out.selected_df.copy()

    rewrite_out = rewrite_selected_df(
        selected_df,
        text_col=TEXT_COL,
        score_col="utility",
        bin_summarizer=fallback_summarizer,
        stitch_rewriter=fallback_summarizer,
        final_rewriter=fallback_summarizer,
        high_q=0.67,
        low_q=0.33,
        use_low=True,
    )

    results[profile_name] = {
        "w_u": w_u,
        "z_t": out.z_t,
        "alpha": out.alpha,
        "selected_df": selected_df,
        "rewrite": rewrite_out,
        "summary_text": rewrite_out["final_summary"],   # final rewritten summary
        "stitched_summary": rewrite_out["stitched_summary"],
        "high_bin_summary": rewrite_out["stage1"]["S_H"],
        "mid_bin_summary": rewrite_out["stage1"]["S_M"],
        "low_bin_summary": rewrite_out["stage1"]["S_L"],
    }


# -------------------------------------------------
# Print compact results
# -------------------------------------------------
for profile_name, res in results.items():
    print("\n" + "=" * 80)
    print(f"PROFILE: {profile_name}")
    print("-" * 80)

    print("Top user preference aspects:")
    top_w = np.argsort(-res["w_u"])[:5]
    for idx in top_w:
        print(f"  {phi_cols[idx]}: {res['w_u'][idx]:.4f}")

    print("\nTop resulting summary aspect profile z_t:")
    top_z = np.argsort(-res["z_t"])[:5]
    for idx in top_z:
        print(f"  {phi_cols[idx]}: {res['z_t'][idx]:.4f}")

    # print("\nSelected sentences:")
    # for i, row in res["selected_df"].reset_index(drop=True).iterrows():
    #     print(f"[{i+1}] utility={row['utility']:.4f}  alpha={row['alpha']:.4f}")
    #     print(f"    {row[TEXT_COL]}")
    #     print()

    G_total, G_cos, KL_w_z = goodness_score_with_kl(
        w_hat=res["w_u"],
        z=res["z_t"],
        lambda_kl=LAMBDA_KL_GOODNESS,
    )

    print(f"Selected-evidence cosine alignment G_cos(w_hat, z): {G_cos:.4f}")
    print(
        f"KL-regularized goodness G_KL(w_hat, z) "
        f"= cosine alignment - {LAMBDA_KL_GOODNESS} * KL: {G_total:.4f}"
    )

    # Selected-evidence alignment
    # G_total, G_align, KL_w_z = goodness_score_with_kl(
    #     w_hat=res["w_u"],
    #     z=res["z_t"],
    #     lambda_kl=1.0,
    # )

    
    # G_align = float(np.dot(res["w_u"], res["z_t"]))/float(np.dot(res["w_u"], res["w_u"]))  # cosine similarity between w_u and z_t

    # print(f"Selected-evidence alignment G(w_hat, p) = w_hat^T z(p): {G_align:.4f}")
    # print(
    #     f"KL-regularized goodness G_KL(w_hat, p) "
    #     f"= alignment - {LAMBDA_KL_GOODNESS} * KL: {G_total:.4f}"
    # )

    print("\n--- Bin summaries ---")
    print("HIGH:", res["high_bin_summary"])
    print("MID :", res["mid_bin_summary"])
    print("LOW :", res["low_bin_summary"])

    # print("\n--- Stitched summary ---")
    # print(res["stitched_summary"])

    # print("\n--- Final rewritten summary ---")
    # print(res["summary_text"])
import numpy as np
import pandas as pd

from pacer_feedback_exp.preference import make_synthetic_true_preference
from pacer_feedback_exp.feedback import SyntheticFeedbackProvider
from pacer_feedback_exp.extractors.gumbel import GumbelExtractor
from pacer_feedback_exp.experiment import run_online_experiment
from pacer_feedback_exp.history import save_history_jsonl

# Load your sentence-level dataframe here
# Must contain:
#   user_id, parent_asin (or product_id column), phi_cols, review text, etc.
data = pd.read_csv("data/train_with_aspect_scores_sentences.csv")

phi_cols = [c for c in data.columns if c.startswith("asp_")]
user_id = "AG73BVBKUOH22USSFJA5ZWL7AKXA" # choose a user_id that exists in your data and has enough sentences for the experiment
product_ids = ["B085BB7B1M"]*100
SEED = 0

# -------How to generate synthetic oracle user preference and feedback provider for testing-------
# Create synthetic oracle user preference for testing.
w_true = make_synthetic_true_preference(
    df_sent=data,
    user_id=user_id,
    phi_cols=phi_cols,
    beta=8.0, # parameter 1
    noise_scale=0.03, # parameter 2
    seed=SEED,
)
# current w_true has higher Aspect 03 compared to Aspect 00 to 19.

# Synthetic feedback provider that simulates user feedback based on the true preference w_true.
feedback_provider = SyntheticFeedbackProvider(
    w_true=w_true,
    gamma=12.0, # parameter 3
    noise_std=0.0, # parameter 4
    seed=SEED,
)

#-------Select Relevant Sentences with Gumbel-Top-k Extractor-------
extractor = GumbelExtractor(
    k=10, # number of sentences to select
    L=1000, # total length constraint for selected sentences
    # tau_ext=1.0, # parameter 5 : temperature for Gumbel noise in extraction
    seed=SEED,
    alpha_mode="utility_softmax",
    tau_alpha=1.5, # parameter 6 : temperature for alpha weights based on utility scores
)

#------------- Run the online learning experiment with the Gumbel extractor and synthetic feedback provider.
result = run_online_experiment(
    df_sent=data,
    user_id=user_id,
    product_ids=product_ids,
    phi_cols=phi_cols,
    extractor = extractor,
    # extractor_fn=lambda **kwargs: run_gumbel_extractor(
    #     extract_for_user_product,
    #     **kwargs,
    # ),
    # extractor_kwargs={
    #     "k": 10,
    #     "L": 900,
    #     "lam": 0.2,
    #     "tau_ext": 2.0,
    #     "seed": 42,
    # },
    feedback_provider=feedback_provider,
    method_name="gumbel",
    beta_init=8.0, # parameter 7 : inverse temperature for estimating initial preference from user's own text
    beta_boltz=8.0, # parameter 8 : inverse temperature for Boltzmann updates
    eta_omd=2.59, # parameter 9 : learning rate for OMD updates
    # eta_omd=4.2,
    baseline_mode="ema",  # "ema" or "last" or None - for OMD baseline
    ema_alpha=0.9,  # for EMA baseline if baseline_mode == "ema
    clip_centered_feedback=0.15, # parameter 10 : clipping value for centered feedback in OMD updates
    use_policy="omd",
    synthetic_oracle=True,  # already created externally
    seed=SEED,
    utility_lam=0.0, # parameter 10 : lambda for combining utility scores with aspect relevance in extraction (if applicable to the extractor)
)

print("\nw_init:\n", result.w_init)
print("\nw_true:\n", result.w_true)
# print("\nw_boltz_final:\n", result.w_boltz_final)
print("\nw_omd_final:\n", result.w_omd_final)
print("\nnum_events:", len(result.history))

save_history_jsonl(result.history, "outputs/history_gumbel.jsonl")

# # I want to create a plot on how cosine OMD evolves from result.logs at every time t get logs[t].get("cos_omd","N/A")
# # cos_omd = [log.get("cos_omd", np.nan)[0][0] for log in result.logs]
import matplotlib.pyplot as plt
# # plt.plot(cos_omd)
# # plt.xlabel("Round")
# # plt.ylabel("Cosine Similarity (OMD)")
# # plt.title("Cosine Similarity between w_omd and w_true over Rounds")
# # plt.show()

# cos_omd_avg = [log.get("cos_omd_avg", np.nan)[0][0] for log in result.logs]
# import matplotlib.pyplot as plt
# plt.plot(cos_omd_avg)
# plt.xlabel("Round")
# plt.ylabel("Cosine Similarity (OMD Avg)")
# plt.title("Cosine Similarity between w_omd_avg and w_true over Rounds")
# plt.show()



# # plot the cumulativeregret over time
# regret = [log.get("cum_regret", np.nan) for log in result.logs]
# plt.plot(regret)
# plt.xlabel("Round")
# plt.ylabel("Cumulative Regret")
# plt.title("Cumulative Regret over Rounds")
# plt.show()

# # plot avg regret over time
# avg_regret = [log.get("avg_regret", np.nan) for log in result.logs]
# plt.plot(avg_regret)
# plt.xlabel("Round")
# plt.ylabel("Average Regret")
# plt.title("Average Regret over Rounds")
# plt.show()

def scalarize(x, default=np.nan):
    """
    Converts log entries like float, np.array([[x]]), list [[x]], etc. into scalar.
    """
    if x is None:
        return default
    try:
        return float(np.asarray(x).ravel()[0])
    except Exception:
        return default


def get_series(logs, key):
    return np.array([scalarize(log.get(key, np.nan)) for log in logs], dtype=float)


logs = result.logs
t_vals = np.arange(1, len(logs) + 1)

# # -----------------------------
# # 1. Cosine similarity
# # -----------------------------
# cos_omd = get_series(logs, "cos_omd")
# cos_omd_avg = get_series(logs, "cos_omd_avg")

# plt.figure(figsize=(6, 4))
# plt.plot(t_vals, cos_omd, label="OMD")
# plt.plot(t_vals, cos_omd_avg, label="OMD averaged")
# plt.xlabel("Round")
# plt.ylabel("Cosine similarity")
# plt.title("Cosine Similarity with True Preference")
# plt.legend(frameon=False)
# plt.tight_layout()
# # plt.savefig(os.path.join(FIG_DIR, "cosine_omd.png"), dpi=300, bbox_inches="tight")
# plt.show()


# # -----------------------------
# # 2. Preference and evidence alignment
# # -----------------------------
# A_pref = get_series(logs, "A_pref")
# A_pref_avg = get_series(logs, "A_pref_avg")
# A_evid = get_series(logs, "A_evid")

# plt.figure(figsize=(6, 4))

# if not np.all(np.isnan(A_pref)):
#     plt.plot(t_vals, A_pref, label=r"$A_t^{\mathrm{pref}}$")
# if not np.all(np.isnan(A_pref_avg)):
#     plt.plot(t_vals, A_pref_avg, label=r"$A_t^{\mathrm{pref}}$ avg")
# if not np.all(np.isnan(A_evid)):
#     plt.plot(t_vals, A_evid, label=r"$A_t^{\mathrm{evid}}$")

# plt.xlabel("Round")
# plt.ylabel("Alignment")
# plt.title("Preference and Evidence Alignment")
# plt.legend(frameon=False)
# plt.tight_layout()
# # plt.savefig(os.path.join(FIG_DIR, "alignment_curves.png"), dpi=300, bbox_inches="tight")
# plt.show()


# # -----------------------------
# # 3. Add exact-recovery and simplex-upper reference lines
# # -----------------------------
# if result.w_true is not None:
#     w_true = np.asarray(result.w_true, dtype=float)

#     pref_self = float(np.dot(w_true, w_true))      # exact recovery benchmark
#     simplex_upper = float(np.max(w_true))          # true simplex upper bound

#     plt.figure(figsize=(6, 4))

#     if not np.all(np.isnan(A_pref_avg)):
#         plt.plot(t_vals, A_pref_avg, label=r"$A_t^{\mathrm{pref}}$ avg")
#     elif not np.all(np.isnan(A_pref)):
#         plt.plot(t_vals, A_pref, label=r"$A_t^{\mathrm{pref}}$")

#     plt.axhline(
#         pref_self,
#         linestyle=":",
#         linewidth=2,
#         label=rf"Exact recovery $\|w^\star\|_2^2={pref_self:.3f}$",
#     )

#     plt.axhline(
#         simplex_upper,
#         linestyle="--",
#         linewidth=1.5,
#         label=rf"Simplex upper $\max_k w_k^\star={simplex_upper:.3f}$",
#     )

#     plt.xlabel("Round")
#     plt.ylabel(r"Preference alignment $A_t^{\mathrm{pref}}$")
#     plt.title("Preference Alignment with Reference Lines")
#     plt.legend(frameon=False)
#     plt.tight_layout()
#     # plt.savefig(os.path.join(FIG_DIR, "preference_alignment_with_refs.png"), dpi=300, bbox_inches="tight")
#     plt.show()

#     plt.figure(figsize=(6, 4))

#     if not np.all(np.isnan(A_evid)):
#         plt.plot(t_vals, A_evid, label=r"$A_t^{\mathrm{evid}}$")

#     plt.axhline(
#         simplex_upper,
#         linestyle=":",
#         linewidth=2,
#         label=rf"Simplex upper $\max_k w_k^\star={simplex_upper:.3f}$",
#     )

#     plt.xlabel("Round")
#     plt.ylabel(r"Evidence alignment $A_t^{\mathrm{evid}}$")
#     plt.title("Selected-Evidence Alignment with Upper Bound")
#     plt.legend(frameon=False)
#     plt.tight_layout()
#     # plt.savefig(os.path.join(FIG_DIR, "evidence_alignment_with_ref.png"), dpi=300, bbox_inches="tight")
#     plt.show()


# -----------------------------
# 4. Surrogate regret diagnostics
# -----------------------------
cum_surrogate_regret = get_series(logs, "cum_surrogate_regret")
avg_surrogate_regret = get_series(logs, "avg_surrogate_regret")
theory_regret_bound = get_series(logs, "theory_regret_bound")
theory_avg_regret_bound = get_series(logs, "theory_avg_regret_bound")

# Cumulative surrogate regret
plt.figure(figsize=(6, 4))

if not np.all(np.isnan(cum_surrogate_regret)):
    plt.plot(t_vals, cum_surrogate_regret, label="Empirical surrogate regret")

# if not np.all(np.isnan(theory_regret_bound)):
#     plt.plot(t_vals, theory_regret_bound, linestyle="--", label="Theoretical upper bound")

plt.xlabel("Round")
plt.ylabel("Cumulative surrogate regret")
plt.title("OMD Surrogate Regret Bound Check")
plt.legend(frameon=False)
plt.tight_layout()
# plt.savefig(os.path.join(FIG_DIR, "cumulative_surrogate_regret.png"), dpi=300, bbox_inches="tight")
plt.show()


# Average surrogate regret
plt.figure(figsize=(6, 4))

if not np.all(np.isnan(avg_surrogate_regret)):
    plt.plot(t_vals, avg_surrogate_regret, label="Empirical average regret")

# if not np.all(np.isnan(theory_avg_regret_bound)):
#     plt.plot(t_vals, theory_avg_regret_bound, linestyle="--", label="Theoretical average bound")

plt.axhline(0.0, linewidth=1, linestyle=":")
plt.xlabel("Round")
plt.ylabel("Average surrogate regret")
plt.title("Average Surrogate Regret")
plt.legend(frameon=False)
plt.tight_layout()
# plt.savefig(os.path.join(FIG_DIR, "average_surrogate_regret.png"), dpi=300, bbox_inches="tight")
plt.show()


# -----------------------------
# 5. Truncated-simplex diagnostic
# -----------------------------
min_w_before = get_series(logs, "min_w_before_update")
min_w_after = get_series(logs, "min_w_after_update")

plt.figure(figsize=(6, 4))

if not np.all(np.isnan(min_w_before)):
    plt.plot(t_vals, min_w_before, label=r"Before update: $\min_k \widehat w_{t,k}$")

if not np.all(np.isnan(min_w_after)):
    plt.plot(t_vals, min_w_after, linestyle="--", label=r"After update: $\min_k \widehat w_{t+1,k}$")

# Add this line only if you used delta_omd in run_online_experiment.
delta_omd = 1e-4
plt.axhline(
    delta_omd,
    linestyle=":",
    linewidth=2,
    label=rf"Allowed minimum $\delta={delta_omd:g}$",
)

plt.xlabel("Round")
plt.ylabel("Minimum coordinate")
plt.title("Truncated-Simplex Check")
plt.legend(frameon=False)
plt.tight_layout()
# plt.savefig(os.path.join(FIG_DIR, "truncated_simplex_check.png"), dpi=300, bbox_inches="tight")
plt.show()


# -----------------------------
# 6. Print final diagnostics
# -----------------------------
# print("\n--- Final diagnostics ---")
# print("Final cosine OMD:", cos_omd[-1] if len(cos_omd) else np.nan)
# print("Final cosine OMD avg:", cos_omd_avg[-1] if len(cos_omd_avg) else np.nan)
# print("Final A_pref:", A_pref[-1] if len(A_pref) else np.nan)
# print("Final A_pref_avg:", A_pref_avg[-1] if len(A_pref_avg) else np.nan)
# print("Final A_evid:", A_evid[-1] if len(A_evid) else np.nan)

if not np.all(np.isnan(avg_surrogate_regret)):
    print("Final avg surrogate regret:", avg_surrogate_regret[-1])

if not np.all(np.isnan(theory_avg_regret_bound)):
    print("Final theoretical avg bound:", theory_avg_regret_bound[-1])

if not np.all(np.isnan(min_w_after)):
    print("Minimum coordinate after update:", np.nanmin(min_w_after))

# if result.w_true is not None:
#     print("Exact recovery benchmark ||w_true||^2:", pref_self)
#     print("Simplex upper bound max_k w_true:", simplex_upper)
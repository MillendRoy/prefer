import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pacer_feedback_exp.preference import make_synthetic_true_preference
from pacer_feedback_exp.feedback import SyntheticFeedbackProvider
from pacer_feedback_exp.extractors.gumbel import GumbelExtractor
from pacer_feedback_exp.extractors.mmr import MMRExtractor  # adjust name if your file/class differs
from pacer_feedback_exp.experiment import run_online_experiment


# ----------------------------
# Configuration
# ----------------------------

DATA_PATH = "data/train_with_aspect_scores_sentences.csv"
OUT_DIR = "outputs/convergence_seeds_finalrun"
FIG_DIR = "figures_finalrun"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

data = pd.read_csv(DATA_PATH)
phi_cols = [c for c in data.columns if c.startswith("asp_")]

user_id = "AG73BVBKUOH22USSFJA5ZWL7AKXA"
product_ids = ["B085BB7B1M"] * 1000

seeds = list(range(10))  # use 20 seeds for the paper plot

COMMON_PARAMS = dict(
    beta_init=8.0,
    beta_boltz=8.0,
    eta_omd=2.59,
    delta_omd=1e-4,
    baseline_mode="running_mean",
    ema_alpha=0.1,
    clip_centered_feedback=0.15,
    synthetic_oracle=True,
    utility_lam=0.0,
)


# ----------------------------
# Extractor factory
# ----------------------------

def make_extractor(extractor_name: str, seed: int):
    if extractor_name == "gumbel":
        return GumbelExtractor(
            k=10,
            L=1000,
            seed=seed,
            alpha_mode="utility_softmax",
            tau_alpha=20,
        )

    if extractor_name == "mmr":
        return MMRExtractor(
            k=10,
            L=1000,
            lam_mmr=0.95,
            alpha_mode="utility_softmax",
            tau_alpha=20,
            # lam=0.2,  # adjust to your MMR convention
        )

    raise ValueError(f"Unknown extractor_name={extractor_name}")


# ----------------------------
# Convert logs to dataframe
# ----------------------------

def logs_to_frame(result, seed: int, variant: str, extractor_name: str, policy_name: str):
    rows = []
    for log in result.logs:
        rows.append({
            "seed": seed,
            "variant": variant,
            "extractor": extractor_name,
            "policy": policy_name,
            "t": int(log["t"]),
            "feedback": float(log["feedback"]),
            "A_pref": float(log.get("A_pref", np.nan)),
            "A_pref_avg": float(log.get("A_pref_avg", np.nan)),
            "cos_omd": float(np.asarray(log.get("cos_omd", [[np.nan]])).ravel()[0]),
            "cos_omd_avg": float(np.asarray(log.get("cos_omd_avg", [[np.nan]])).ravel()[0]),
            "A_evid": float(log.get("A_evid", np.nan)),
            "A_evid_avg": float(log.get("A_evid_avg", np.nan)),
            "cos_evid": float(log.get("cos_evid", np.nan)),
            "cos_evid_avg": float(log.get("cos_evid_avg", np.nan)),            
            "kl_omd_to_true": float(log.get("kl_omd_to_true", np.nan)),
        })
    return pd.DataFrame(rows)


# ----------------------------
# Run one variant
# ----------------------------

def run_one(seed: int, extractor_name: str, policy_name: str):
    np.random.seed(seed)

    w_true = make_synthetic_true_preference(
        df_sent=data,
        user_id=user_id,
        phi_cols=phi_cols,
        beta=8.0,
        noise_scale=0.03,
        seed=seed,
    )

    feedback_provider = SyntheticFeedbackProvider(
        w_true=w_true,
        gamma=12.0,
        noise_std=0.0,
        seed=seed,
    )

    extractor = make_extractor(extractor_name, seed)

    if policy_name == "online":
        use_policy = "omd"
        variant = f"PREFER-{extractor_name.upper()}"
    elif policy_name == "static":
        use_policy = "static"
        variant = f"Static-{extractor_name.upper()}"
    else:
        raise ValueError(f"Unknown policy_name={policy_name}")

    result = run_online_experiment(
        df_sent=data,
        user_id=user_id,
        product_ids=product_ids,
        phi_cols=phi_cols,
        extractor=extractor,
        feedback_provider=feedback_provider,
        method_name=extractor_name,
        use_policy=use_policy,
        seed=seed,
        **COMMON_PARAMS,
    )

    df_run= logs_to_frame(
        result=result,
        seed=seed,
        variant=variant,
        extractor_name=extractor_name,
        policy_name=policy_name,
    )

    return df_run, result


# ----------------------------
# Run full experiment
# ----------------------------

all_frames = []
regret_frames = []


# Store one online OMD run for regret diagnostics.
regret_result = None
regret_label = None


for seed in seeds:
    for extractor_name in ["gumbel", "mmr"]:
        for policy_name in ["static","online"]:
            print(f"Running seed={seed}, extractor={extractor_name}, policy={policy_name}")
            df_run, result = run_one(seed, extractor_name, policy_name)
            all_frames.append(df_run)

            # Store regret logs for every online OMD run.
            if policy_name == "online":
                df_regret = pd.DataFrame([
                    {
                        "seed": seed,
                        "extractor": extractor_name,
                        "variant": f"PREFER-{extractor_name.upper()}",
                        "t": log["t"] + 1,
                        "cum_surrogate_regret": log.get("cum_surrogate_regret", np.nan),
                        "avg_surrogate_regret": log.get("avg_surrogate_regret", np.nan),
                        "theory_regret_bound": log.get("theory_regret_bound", np.nan),
                        "theory_avg_regret_bound": log.get("theory_avg_regret_bound", np.nan),
                        "min_w_before_update": log.get("min_w_before_update", np.nan),
                        "min_w_after_update": log.get("min_w_after_update", np.nan),
                    }
                    for log in result.logs
                ])
                regret_frames.append(df_regret)

df_all = pd.concat(all_frames, ignore_index=True)
df_all.to_csv(os.path.join(OUT_DIR, "convergence_all_runs.csv"), index=False)
print(df_all.head())

df_regret_all = pd.concat(regret_frames, ignore_index=True)
df_regret_all.to_csv(os.path.join(OUT_DIR, "regret_all_online_runs.csv"), index=False)


def summarize_for_plot(df, metric):
    summary = (
        df.groupby(["extractor", "variant", "t"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci95"] = 1.96 * summary["sem"]
    summary["lower"] = summary["mean"] - summary["ci95"]
    summary["upper"] = summary["mean"] + summary["ci95"]
    return summary


def plot_metric_by_extractor(df, extractor_name, metric, ylabel, out_path):
    summary = summarize_for_plot(df[df["extractor"] == extractor_name], metric)

    plt.figure(figsize=(5.2, 3.4))

    for variant in sorted(summary["variant"].unique()):
        sub = summary[summary["variant"] == variant].sort_values("t")

        x = sub["t"].to_numpy()
        y = sub["mean"].to_numpy()
        lo = sub["lower"].to_numpy()
        hi = sub["upper"].to_numpy()

        plt.plot(x, y, label=variant)
        plt.fill_between(x, lo, hi, alpha=0.2)

    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(f"{extractor_name.upper()} extractor")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

# Main paper figure: selected-evidence alignment
plot_metric_by_extractor(
    df=df_all,
    extractor_name="mmr",
    metric="A_evid",
    ylabel=r"Selected-evidence alignment $A_t^{\mathrm{evid}}$",
    out_path=os.path.join(FIG_DIR, "convergence_mmr.pdf"),
)

plot_metric_by_extractor(
    df=df_all,
    extractor_name="gumbel",
    metric="A_evid",
    ylabel=r"Selected-evidence alignment $A_t^{\mathrm{evid}}$",
    out_path=os.path.join(FIG_DIR, "convergence_gumbel.pdf"),
)

# Optional appendix figures: preference-space alignment
plot_metric_by_extractor(
    df=df_all,
    extractor_name="mmr",
    metric="A_pref_avg",
    ylabel=r"Preference alignment $A_t^{\mathrm{pref}}$",
    out_path=os.path.join(FIG_DIR, "preference_alignment_mmr.pdf"),
)

plot_metric_by_extractor(
    df=df_all,
    extractor_name="gumbel",
    metric="A_pref_avg",
    ylabel=r"Preference alignment $A_t^{\mathrm{pref}}$",
    out_path=os.path.join(FIG_DIR, "preference_alignment_gumbel.pdf"),
)


# ============================================================
# Regret verification plots across all online OMD runs
# ============================================================

def summarize_regret_metric(df_regret_all: pd.DataFrame, metric: str) -> pd.DataFrame:
    summary = (
        df_regret_all
        .groupby(["extractor", "variant", "t"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci95"] = 1.96 * summary["sem"]
    summary["lower"] = summary["mean"] - summary["ci95"]
    summary["upper"] = summary["mean"] + summary["ci95"]
    return summary


def plot_regret_bound_by_extractor(
    df_regret_all: pd.DataFrame,
    extractor_name: str,
    out_path: str,
):
    """
    Plot cumulative empirical surrogate regret against the theorem bound,
    averaged across seeds.
    """
    df_ex = df_regret_all[df_regret_all["extractor"] == extractor_name].copy()

    emp = summarize_regret_metric(df_ex, "cum_surrogate_regret")
    bd = summarize_regret_metric(df_ex, "theory_regret_bound")

    plt.figure(figsize=(6, 4))

    plt.plot(
        emp["t"],
        emp["mean"],
        label="Empirical surrogate regret",
    )
    plt.fill_between(
        emp["t"],
        emp["lower"],
        emp["upper"],
        alpha=0.2,
    )

    plt.plot(
        bd["t"],
        bd["mean"],
        linestyle="--",
        label="Theoretical upper bound",
    )

    plt.xlabel("Round")
    plt.ylabel("Cumulative surrogate regret")
    plt.title(f"PREFER-{extractor_name.upper()}: regret bound check")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_average_regret_bound_by_extractor(
    df_regret_all: pd.DataFrame,
    extractor_name: str,
    out_path: str,
):
    """
    Plot average empirical surrogate regret against the average theorem bound.
    """
    df_ex = df_regret_all[df_regret_all["extractor"] == extractor_name].copy()

    emp = summarize_regret_metric(df_ex, "avg_surrogate_regret")
    bd = summarize_regret_metric(df_ex, "theory_avg_regret_bound")

    plt.figure(figsize=(6, 4))

    plt.plot(
        emp["t"],
        emp["mean"],
        label="Empirical average regret",
    )
    plt.fill_between(
        emp["t"],
        emp["lower"],
        emp["upper"],
        alpha=0.2,
    )

    plt.plot(
        bd["t"],
        bd["mean"],
        linestyle="--",
        label="Theoretical average bound",
    )

    plt.xlabel("Round")
    plt.ylabel("Average surrogate regret")
    plt.title(f"PREFER-{extractor_name.upper()}: average regret")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_min_coordinate_check(
    df_regret_all: pd.DataFrame,
    extractor_name: str,
    out_path: str,
):
    """
    Check whether the OMD iterates remain inside the truncated simplex.
    """
    df_ex = df_regret_all[df_regret_all["extractor"] == extractor_name].copy()

    before = summarize_regret_metric(df_ex, "min_w_before_update")
    after = summarize_regret_metric(df_ex, "min_w_after_update")

    plt.figure(figsize=(6, 4))

    plt.plot(
        before["t"],
        before["mean"],
        label=r"Before update: $\min_k \widehat w_{t,k}$",
    )
    plt.fill_between(
        before["t"],
        before["lower"],
        before["upper"],
        alpha=0.2,
    )

    plt.plot(
        after["t"],
        after["mean"],
        linestyle="--",
        label=r"After update: $\min_k \widehat w_{t+1,k}$",
    )
    plt.fill_between(
        after["t"],
        after["lower"],
        after["upper"],
        alpha=0.2,
    )

    plt.xlabel("Round")
    plt.ylabel(r"Minimum coordinate")
    plt.title(f"PREFER-{extractor_name.upper()}: truncated-simplex check")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# Cumulative regret vs theorem bound
plot_regret_bound_by_extractor(
    df_regret_all=df_regret_all,
    extractor_name="mmr",
    out_path=os.path.join(FIG_DIR, "omd_surrogate_regret_mmr.pdf"),
)

plot_regret_bound_by_extractor(
    df_regret_all=df_regret_all,
    extractor_name="gumbel",
    out_path=os.path.join(FIG_DIR, "omd_surrogate_regret_gumbel.pdf"),
)

# Average regret vs theorem bound
plot_average_regret_bound_by_extractor(
    df_regret_all=df_regret_all,
    extractor_name="mmr",
    out_path=os.path.join(FIG_DIR, "omd_average_surrogate_regret_mmr.pdf"),
)

plot_average_regret_bound_by_extractor(
    df_regret_all=df_regret_all,
    extractor_name="gumbel",
    out_path=os.path.join(FIG_DIR, "omd_average_surrogate_regret_gumbel.pdf"),
)

# Truncated simplex diagnostic
plot_min_coordinate_check(
    df_regret_all=df_regret_all,
    extractor_name="mmr",
    out_path=os.path.join(FIG_DIR, "omd_min_coordinate_mmr.pdf"),
)

plot_min_coordinate_check(
    df_regret_all=df_regret_all,
    extractor_name="gumbel",
    out_path=os.path.join(FIG_DIR, "omd_min_coordinate_gumbel.pdf"),
)


# ============================================================
# Console diagnostics for paper/debugging
# ============================================================

print("\n===== OMD regret diagnostics across all online runs =====")

diagnostics = (
    df_regret_all
    .groupby(["extractor", "variant"])
    .agg(
        final_cum_regret=("cum_surrogate_regret", "last"),
        final_theory_bound=("theory_regret_bound", "last"),
        final_avg_regret=("avg_surrogate_regret", "last"),
        final_avg_bound=("theory_avg_regret_bound", "last"),
        min_coord_seen=("min_w_after_update", "min"),
    )
    .reset_index()
)

print(diagnostics)

diagnostics.to_csv(
    os.path.join(OUT_DIR, "omd_regret_diagnostics.csv"),
    index=False,
)
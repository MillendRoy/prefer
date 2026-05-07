import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pacer_feedback_exp.preference import make_synthetic_true_preference
from pacer_feedback_exp.feedback import TimeVaryingSyntheticFeedbackProvider
from pacer_feedback_exp.extractors.gumbel import GumbelExtractor
from pacer_feedback_exp.extractors.mmr import MMRExtractor
from pacer_feedback_exp.experiment import run_online_experiment


DATA_PATH = "data/train_with_aspect_scores_sentences.csv"
OUT_DIR = "outputs/preference_drift"
FIG_DIR = "figures/preference_drift"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

DRIFT_START = 20
DRIFT_END = 90


data = pd.read_csv(DATA_PATH)
phi_cols = [c for c in data.columns if c.startswith("asp_")]



PRODUCT_ID = "B085BB7B1M"
cand = data[data["parent_asin"] == PRODUCT_ID].copy()
aspect_means = cand[phi_cols].mean().sort_values(ascending=False)
user_id = "AG73BVBKUOH22USSFJA5ZWL7AKXA"
product_ids = [PRODUCT_ID] * 200

print("Top product-level aspects:")
print(aspect_means.head(10))

K = len(phi_cols)


def make_aspect_focused_preference(K, focus_aspects, mass=0.75):
    """
    Create a synthetic preference vector concentrated on selected aspects.
    K: total number of aspects
    focus_aspects: list of aspect indices to focus on (0-based)
    mass: total mass to assign to focused aspects (between 0 and 1)
    """
    w = np.ones(K, dtype=float) * ((1.0 - mass) / (K - len(focus_aspects)))

    for a in focus_aspects:
        w[a] = mass / len(focus_aspects)

    return w / w.sum()

start_aspect = int(aspect_means.index[0].replace("asp_", ""))
end_aspect = int(aspect_means.index[1].replace("asp_", ""))

# Example drift:
# initial user cares about Aspect 1;
# later user cares about Aspect 8.
w_start = make_aspect_focused_preference(K, focus_aspects=[start_aspect], mass=0.5)
w_end = make_aspect_focused_preference(K, focus_aspects=[end_aspect], mass=0.5)
# Alternative mixed drift:
# w_end = make_aspect_focused_preference(K, focus_aspects=[1, 8], mass=0.80)
print("Drift from aspect", start_aspect, "to aspect", end_aspect)
feedback_provider = TimeVaryingSyntheticFeedbackProvider(
    w_start=w_start,
    w_end=w_end,
    drift_start=DRIFT_START,
    drift_end=DRIFT_END,
    gamma=12.0,
    noise_std=0.0,
    seed=42,
    mode="linear",   # "linear" or "abrupt"
)


def make_extractor(name, seed):
    if name == "gumbel":
        return GumbelExtractor(
            k=10,
            L=1000,
            seed=seed,
            alpha_mode="utility_softmax",
            tau_alpha=1.5,
        )

    if name == "mmr":
        return MMRExtractor(
            k=10,
            L=1000,
            lam_mmr=0.95,
            # beta_kl=0.1,
            alpha_mode="uniform",
            tau_alpha=1.5,
        )

    raise ValueError(name)


def scalarize(x, default=np.nan):
    if x is None:
        return default
    try:
        return float(np.asarray(x).ravel()[0])
    except Exception:
        return default


def logs_to_df(result, extractor_name, seed):
    rows = []

    for log in result.logs:
        t = int(log["t"])

        w_true_t = np.asarray(log["w_true"], dtype=float)
        w_omd = np.asarray(log["w_omd"], dtype=float)
        w_avg = np.asarray(log["w_omd_avg"], dtype=float)
        z_t = np.asarray(log["z_t"], dtype=float)

        row = {
            "seed": seed,
            "extractor": extractor_name,
            "variant": f"PREFER-{extractor_name.upper()}",
            "policy": "online",
            "t": t,
            "feedback": float(log["feedback"]),

            # Alignment metrics: keep your existing notation.
            "cos_pref": scalarize(log.get("cos_omd")),
            "cos_pref_avg": scalarize(log.get("cos_omd_avg")),
            "cos_evid": scalarize(log.get("cos_evid")),
            "cos_evid_avg": scalarize(log.get("cos_evid_avg")),
            "A_pref": scalarize(log.get("A_pref")),
            "A_pref_avg": scalarize(log.get("A_pref_avg")),
            "A_evid": scalarize(log.get("A_evid")),
            "A_evid_avg": scalarize(log.get("A_evid_avg")),

            # Regret diagnostics.
            "cum_surrogate_regret": scalarize(log.get("cum_surrogate_regret")),
            "avg_surrogate_regret": scalarize(log.get("avg_surrogate_regret")),
            "theory_regret_bound": scalarize(log.get("theory_regret_bound")),
            "theory_avg_regret_bound": scalarize(log.get("theory_avg_regret_bound")),

            # Truncated-simplex diagnostics.
            "min_w_before_update": scalarize(log.get("min_w_before_update")),
            "min_w_after_update": scalarize(log.get("min_w_after_update")),

            # Optional update diagnostics.
            "f_eff": scalarize(log.get("f_eff")),
            "baseline_t": scalarize(log.get("baseline_t")),
        }

        # Save aspect coordinates for plotting drift.
        for k in range(len(w_true_t)):
            row[f"w_true_{k}"] = w_true_t[k]
            row[f"w_omd_{k}"] = w_omd[k]
            row[f"w_avg_{k}"] = w_avg[k]
            row[f"z_{k}"] = z_t[k]

        rows.append(row)

    return pd.DataFrame(rows)


all_runs = []

for seed in range(10):
    for extractor_name in ["gumbel", "mmr"]:
        print(f"Running seed={seed}, extractor={extractor_name}")

        extractor = make_extractor(extractor_name, seed)

        result = run_online_experiment(
            df_sent=data,
            user_id=user_id,
            product_ids=product_ids,
            phi_cols=phi_cols,
            extractor=extractor,
            feedback_provider=feedback_provider,
            method_name=extractor_name,
            beta_init=8.0,
            beta_boltz=8.0,
            eta_omd=1.5,
            # eta_omd=2.59,
            delta_omd=1e-4,
            c_eta=0.05,
            baseline_mode="ema",
            ema_alpha=0.1,
            clip_centered_feedback=0.15,
            # clip_centered_feedback=1.0,
            use_policy="omd",
            synthetic_oracle=True,
            seed=seed,
            utility_lam=0.0,
        )

        df_run = logs_to_df(result, extractor_name, seed)
        all_runs.append(df_run)


df_all = pd.concat(all_runs, ignore_index=True)
csv_path = os.path.join(OUT_DIR, "preference_drift_all_runs.csv")
df_all.to_csv(csv_path, index=False)
print("Saved:", csv_path)

def summarize(df, metric):
    out = (
        df.groupby(["extractor", "t"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    out["sem"] = out["std"] / np.sqrt(out["count"])
    out["ci95"] = 1.96 * out["sem"]
    out["lower"] = out["mean"] - out["ci95"]
    out["upper"] = out["mean"] + out["ci95"]
    return out


def plot_alignment_under_drift(df_all):
    for extractor_name in ["mmr", "gumbel"]:
        df_ex = df_all[df_all["extractor"] == extractor_name]

        s_pref = summarize(df_ex, "cos_pref")
        s_evid = summarize(df_ex, "cos_evid")

        plt.figure(figsize=(6, 4))

        for s, label, linestyle in [
            (s_pref, r"Preference alignment $A_t^{\mathrm{pref}}$", "-"),
            (s_evid, r"Evidence alignment $A_t^{\mathrm{evid}}$", "--"),
        ]:
            x = s["t"].to_numpy()
            y = s["mean"].to_numpy()
            lo = s["lower"].to_numpy()
            hi = s["upper"].to_numpy()

            plt.plot(x, y, linestyle=linestyle, label=label)
            plt.fill_between(x, lo, hi, alpha=0.2)

        plt.axvline(DRIFT_START, linestyle=":", linewidth=1.5, label="drift begins")
        plt.axvline(DRIFT_END, linestyle=":", linewidth=1.5, label="drift ends")

        plt.xlabel("Round")
        plt.ylabel("Alignment with current preference")
        plt.title(f"Preference drift tracking: {extractor_name.upper()}")
        plt.legend(frameon=False)
        plt.tight_layout()

        out = os.path.join(FIG_DIR, f"drift_alignment_{extractor_name}.pdf")
        plt.savefig(out, bbox_inches="tight")
        plt.show()


plot_alignment_under_drift(df_all)

def plot_aspect_tracking(df_all, extractor_name="gumbel", aspects=(1, 8)):
    df_ex = df_all[df_all["extractor"] == extractor_name].copy()

    plt.figure(figsize=(7, 4))

    for a in aspects:
        s_true = summarize(df_ex, f"w_true_{a}")
        s_hat = summarize(df_ex, f"w_omd_{a}")

        plt.plot(
            s_true["t"],
            s_true["mean"],
            linestyle="--",
            label=rf"True preference aspect {a}",
        )

        plt.plot(
            s_hat["t"],
            s_hat["mean"],
            linestyle="-",
            label=rf"Learned preference aspect {a}",
        )

    plt.axvline(DRIFT_START, linestyle=":", linewidth=1.5)
    plt.axvline(DRIFT_END, linestyle=":", linewidth=1.5)

    plt.xlabel("Round")
    plt.ylabel("Preference mass")
    plt.title(f"Aspect-level preference drift tracking: {extractor_name.upper()}")
    plt.legend(frameon=False)
    plt.tight_layout()

    out = os.path.join(FIG_DIR, f"aspect_tracking_{extractor_name}.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.show()


plot_aspect_tracking(df_all, extractor_name="gumbel", aspects=(start_aspect, end_aspect))
plot_aspect_tracking(df_all, extractor_name="mmr", aspects=(start_aspect, end_aspect))

# ============================================================
# Regret and OMD diagnostics under preference drift
# ============================================================

def summarize_drift_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    summary = (
        df
        .groupby(["extractor", "variant", "t"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci95"] = 1.96 * summary["sem"]
    summary["lower"] = summary["mean"] - summary["ci95"]
    summary["upper"] = summary["mean"] + summary["ci95"]
    return summary


def _add_drift_window():
    plt.axvline(DRIFT_START, linestyle=":", linewidth=1.5, label="drift begins")
    plt.axvline(DRIFT_END, linestyle=":", linewidth=1.5, label="drift ends")


def plot_drift_regret_bound_by_extractor(
    df_all: pd.DataFrame,
    extractor_name: str,
    out_path: str,
):
    """
    Plot cumulative empirical surrogate regret against the theorem-style bound,
    averaged across seeds, under preference drift.
    """
    df_ex = df_all[df_all["extractor"] == extractor_name].copy()

    emp = summarize_drift_metric(df_ex, "cum_surrogate_regret")
    bd = summarize_drift_metric(df_ex, "theory_regret_bound")

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

    _add_drift_window()

    plt.xlabel("Round")
    plt.ylabel("Cumulative surrogate regret")
    plt.title(f"PREFER-{extractor_name.upper()}: regret under drift")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_drift_average_regret_bound_by_extractor(
    df_all: pd.DataFrame,
    extractor_name: str,
    out_path: str,
):
    """
    Plot average empirical surrogate regret against the average theorem-style bound,
    averaged across seeds, under preference drift.
    """
    df_ex = df_all[df_all["extractor"] == extractor_name].copy()

    emp = summarize_drift_metric(df_ex, "avg_surrogate_regret")
    bd = summarize_drift_metric(df_ex, "theory_avg_regret_bound")

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

    plt.axhline(0.0, linestyle=":", linewidth=1.0)
    _add_drift_window()

    plt.xlabel("Round")
    plt.ylabel("Average surrogate regret")
    plt.title(f"PREFER-{extractor_name.upper()}: average regret under drift")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_drift_min_coordinate_check(
    df_all: pd.DataFrame,
    extractor_name: str,
    out_path: str,
    delta_omd: float = 1e-4,
):
    """
    Check whether the OMD iterates remain inside the truncated simplex
    during preference drift.
    """
    df_ex = df_all[df_all["extractor"] == extractor_name].copy()

    before = summarize_drift_metric(df_ex, "min_w_before_update")
    after = summarize_drift_metric(df_ex, "min_w_after_update")

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

    plt.axhline(
        delta_omd,
        linestyle=":",
        linewidth=1.5,
        label=rf"Threshold $\delta={delta_omd:g}$",
    )

    _add_drift_window()

    plt.xlabel("Round")
    plt.ylabel(r"Minimum coordinate")
    plt.title(f"PREFER-{extractor_name.upper()}: truncated-simplex check under drift")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_drift_feedback_diagnostics(
    df_all: pd.DataFrame,
    extractor_name: str,
    out_path: str,
):
    """
    Plot feedback and centered feedback signal under preference drift.
    """
    df_ex = df_all[df_all["extractor"] == extractor_name].copy()

    fb = summarize_drift_metric(df_ex, "feedback")
    feff = summarize_drift_metric(df_ex, "f_eff")

    plt.figure(figsize=(6, 4))

    plt.plot(
        fb["t"],
        fb["mean"],
        label=r"Feedback $f_t$",
    )
    plt.fill_between(
        fb["t"],
        fb["lower"],
        fb["upper"],
        alpha=0.2,
    )

    plt.plot(
        feff["t"],
        feff["mean"],
        linestyle="--",
        label=r"Centered feedback $\widetilde f_t$",
    )
    plt.fill_between(
        feff["t"],
        feff["lower"],
        feff["upper"],
        alpha=0.2,
    )

    plt.axhline(0.0, linestyle=":", linewidth=1.0)
    _add_drift_window()

    plt.xlabel("Round")
    plt.ylabel("Feedback signal")
    plt.title(f"PREFER-{extractor_name.upper()}: feedback under drift")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# Save regret/diagnostic figures for both extractors
# ------------------------------------------------------------

for extractor_name in ["mmr", "gumbel"]:
    plot_drift_regret_bound_by_extractor(
        df_all=df_all,
        extractor_name=extractor_name,
        out_path=os.path.join(FIG_DIR, f"drift_omd_surrogate_regret_{extractor_name}.pdf"),
    )

    plot_drift_average_regret_bound_by_extractor(
        df_all=df_all,
        extractor_name=extractor_name,
        out_path=os.path.join(FIG_DIR, f"drift_omd_average_surrogate_regret_{extractor_name}.pdf"),
    )

    plot_drift_min_coordinate_check(
        df_all=df_all,
        extractor_name=extractor_name,
        out_path=os.path.join(FIG_DIR, f"drift_omd_min_coordinate_{extractor_name}.pdf"),
        delta_omd=1e-4,
    )

    plot_drift_feedback_diagnostics(
        df_all=df_all,
        extractor_name=extractor_name,
        out_path=os.path.join(FIG_DIR, f"drift_feedback_{extractor_name}.pdf"),
    )

    diagnostics = (
    df_all
    .groupby(["extractor", "variant"])
    .agg(
        final_cum_regret=("cum_surrogate_regret", "last"),
        final_theory_bound=("theory_regret_bound", "last"),
        final_avg_regret=("avg_surrogate_regret", "last"),
        final_avg_bound=("theory_avg_regret_bound", "last"),
        min_coord_seen=("min_w_after_update", "min"),
        final_A_pref=("A_pref", "last"),
        final_A_evid=("A_evid", "last"),
        final_cos_pref=("cos_pref", "last"),
        final_cos_evid=("cos_evid", "last"),
    )
    .reset_index()
)

print("\n===== Preference drift diagnostics =====")
print(diagnostics)

diagnostics.to_csv(
    os.path.join(OUT_DIR, "preference_drift_diagnostics.csv"),
    index=False,
)
"""
Analyse the judge-panel study exported from the Google Sheet.

In the Sheet: File > Download > Comma-separated values, once for each tab
("participants", "intrusion", "personalization"), and save them as
participants.csv, intrusion.csv, personalization.csv next to this script.

    python analyze.py                # all participants
    python analyze.py --min-minutes 4 --exclude test,pilot

Reports:
  Part 1 (aspect validity)   accuracy vs chance (1/4) with an exact binomial test,
                             per-aspect accuracy, per-judge accuracy, Fleiss' kappa.
  Part 2 (personalization)   PREFER win rate vs 50% (binomial), per-aspect win rate,
                             Fleiss' kappa, Wilcoxon on Likert ratings, position-bias check.
  Part 1 bonus round        sentence-to-label matching accuracy vs chance, per-label recall and
                             "none of these" rate (label validity), Fleiss' kappa  (needs matching.csv).
  Simulator validation       Spearman between human Likert ratings and the simulated
                             aspect mass of each summary (swap in your f_t column if different).
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats


def fleiss_kappa(counts: np.ndarray) -> float:
    """counts: items x categories, each row sums to the number of raters n (same n for all rows)."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(1)
    if not np.allclose(n, n[0]) or n[0] < 2:
        return float("nan")
    n = n[0]
    p_j = counts.sum(0) / counts.sum()
    P_i = ((counts ** 2).sum(1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else float("nan")


def binom(k, n, p):
    r = stats.binomtest(int(k), int(n), p, alternative="greater")
    ci = stats.binomtest(int(k), int(n), p).proportion_ci(confidence_level=0.95)  # two-sided Clopper-Pearson
    return f"{k}/{n} = {k / n:.1%}  (chance {p:.0%}; one-sided p = {r.pvalue:.2g}; 95% CI {ci.low:.1%} to {ci.high:.1%})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--exclude", default="test", help="comma-separated participant-ID prefixes to drop")
    ap.add_argument("--min-minutes", type=float, default=0, help="drop sessions faster than this")
    a = ap.parse_args()

    part = pd.read_csv(f"{a.dir}/participants.csv")
    intr = pd.read_csv(f"{a.dir}/intrusion.csv")
    pers = pd.read_csv(f"{a.dir}/personalization.csv")

    drop = [p.strip().lower() for p in a.exclude.split(",") if p.strip()]
    keep = part[~part.participant_id.astype(str).str.lower().str.startswith(tuple(drop))] if drop else part
    keep = keep[keep.duration_s >= a.min_minutes * 60]
    sessions = set(keep.session_id)
    intr, pers = intr[intr.session_id.isin(sessions)], pers[pers.session_id.isin(sessions)]
    J = len(sessions)
    print(f"Judges kept: {J} (of {len(part)} sessions)")
    if part.repeat_participant_id.sum():
        print(f"  note: {int(part.repeat_participant_id.sum())} session(s) reused a participant ID")
    print(f"  median duration: {keep.duration_s.median() / 60:.1f} min\n")

    # ---------- Part 1: aspect validity (cluster intrusion) ----------
    n_opts = int(intr.display_order.astype(str).str.split().str.len().max())
    print("PART 1  Aspect validity (word/sentence intrusion)")
    print("  Overall accuracy:", binom(intr.is_correct.sum(), len(intr), 1 / n_opts))
    by_q = intr.groupby(["question", "target_aspect", "intruder_aspect"]).is_correct.agg(["sum", "count"])
    by_q["acc"] = by_q["sum"] / by_q["count"]
    print("\n  Per item (target aspect, intruder aspect):")
    print(by_q.to_string(float_format=lambda x: f"{x:.2f}"))
    per_judge = intr.groupby("participant_id").is_correct.mean()
    print(f"\n  Per-judge accuracy: mean {per_judge.mean():.2f}, min {per_judge.min():.2f}, max {per_judge.max():.2f}")
    counts = (intr.pivot_table(index="question", columns="chosen_option", values="session_id", aggfunc="count")
              .reindex(columns=range(1, n_opts + 1), fill_value=0).fillna(0))
    print(f"  Fleiss' kappa (agreement on which sentence is the intruder): {fleiss_kappa(counts.values):.3f}\n")

    # ---------- Part 1 bonus round: label validity ----------
    mpath = f"{a.dir}/matching.csv"
    if os.path.exists(mpath):
        m = pd.read_csv(mpath)
        m = m[m.session_id.isin(sessions)]
        n_choices = m.labels_shown.astype(str).str.split().str.len() + 1  # labels + "none of these"
        chance = float((1 / n_choices).mean())
        print("PART 1 BONUS  Sentence-to-label matching (label validity)")
        print("  Overall accuracy:", binom(m.is_correct.sum(), len(m), chance))
        by_lab = m.groupby("sentence_aspect").agg(n=("is_correct", "size"), recall=("is_correct", "mean"),
                                                  none_rate=("chosen_label", lambda s: (s == "none").mean()))
        print("\n  Per aspect label (recall = share of its sentences judges stamped with it):")
        print(by_lab.sort_values("recall").to_string(float_format=lambda x: f"{x:.2f}"))
        both = m.merge(intr[["session_id", "question", "is_correct"]].rename(columns={"is_correct": "intruder_found"}),
                       on=["session_id", "question"])
        acc_by = both.groupby("intruder_found").is_correct.mean()
        print("\n  Matching accuracy when the judge found / missed the intruder: "
              + ", ".join(f"{'found' if k else 'missed'} {v:.2f}" for k, v in acc_by.items()))
        cats = sorted(m.chosen_label.unique())
        counts_m = m.pivot_table(index=["question", "option"], columns="chosen_label", values="session_id",
                                 aggfunc="count").reindex(columns=cats, fill_value=0).fillna(0)
        print(f"  Fleiss' kappa (agreement on each sentence's label): {fleiss_kappa(counts_m.values):.3f}\n")

    # ---------- Part 2: personalization ----------
    print("PART 2  Summary personalization (PREFER vs generic)")
    print("  PREFER win rate:", binom(pers.chose_personalized.sum(), len(pers), 0.5))
    by_a = pers.groupby(["question", "aspect"]).chose_personalized.agg(["sum", "count"])
    by_a["win_rate"] = by_a["sum"] / by_a["count"]
    print("\n  Per item:")
    print(by_a.to_string(float_format=lambda x: f"{x:.2f}"))
    counts2 = pers.pivot_table(index="question", columns="choice", values="session_id", aggfunc="count") \
        .reindex(columns=["A", "B"], fill_value=0).fillna(0)
    print(f"\n  Fleiss' kappa (A/B choice): {fleiss_kappa(counts2.values):.3f}")
    rp, rg = pers.rating_personalized, pers.rating_generic
    w = stats.wilcoxon(rp, rg, alternative="greater", zero_method="wilcox")
    print(f"  Likert: PREFER mean {rp.mean():.2f} vs generic {rg.mean():.2f}; Wilcoxon one-sided p = {w.pvalue:.2g}")
    chose_left = ((pers.choice == pers.left_shown)).mean()
    print(f"  Position check: chose the left summary {chose_left:.1%} of the time (50% = no position bias)\n")

    # ---------- Simulator validation ----------
    print("SIMULATOR VALIDATION  human Likert vs simulated alignment")
    long = pd.concat([
        pd.DataFrame({"question": pers.question, "kind": "prefer", "rating": rp, "sim": pers.pers_aspect_mass}),
        pd.DataFrame({"question": pers.question, "kind": "generic", "rating": rg, "sim": pers.generic_aspect_mass}),
    ])
    rho, p = stats.spearmanr(long.rating, long.sim)
    print(f"  Per judgment (n = {len(long)}): Spearman rho = {rho:.3f}, p = {p:.2g}")
    agg = long.groupby(["question", "kind"]).agg(rating=("rating", "mean"), sim=("sim", "first"))
    rho2, p2 = stats.spearmanr(agg.rating, agg.sim)
    print(f"  Per summary, mean rating (n = {len(agg)}): Spearman rho = {rho2:.3f}, p = {p2:.2g}")


if __name__ == "__main__":
    main()

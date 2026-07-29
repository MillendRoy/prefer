"""Run PREFER against an independent cross-embedding judge.

The learner uses ``asp_*`` columns.  The judge uses ``judge_asp_*`` columns
created by ``scripts/build_cross_embedding_judge.py``.  The online learner
receives only scalar judge feedback and updates in the learner space.
"""

from __future__ import annotations

import argparse
from itertools import cycle, islice
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pacer_feedback_exp.cross_embedding import (
    CrossEmbeddingJudgeFeedbackProvider,
    aspect_columns,
    fit_preference_from_aspects,
    infer_text_column,
)
from pacer_feedback_exp.experiment import run_online_experiment
from pacer_feedback_exp.extractors.gumbel import GumbelExtractor
from pacer_feedback_exp.extractors.mmr import MMRExtractor
from pacer_feedback_exp.extractors.random import RandomExtractor
from pacer_feedback_exp.heldout_feedback import split_user_history
from pacer_feedback_exp.utils import ensure_global_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PREFER with an independent embedding-space judge."
    )
    parser.add_argument(
        "--data", default="data/cross_embedding/train_with_cross_judge.csv"
    )
    parser.add_argument("--user-ids", nargs="+", required=True)
    parser.add_argument("--user-col", default="user_id")
    parser.add_argument("--product-col", default="parent_asin")
    parser.add_argument("--text-col", default=None)
    parser.add_argument("--group-col", default=None)
    parser.add_argument("--allow-row-split", action="store_true")
    parser.add_argument("--holdout-fraction", type=float, default=0.5)
    parser.add_argument("--init-mode", choices=["fit", "uniform"], default="fit")
    parser.add_argument("--beta-judge", type=float, default=8.0)
    parser.add_argument("--beta-init", type=float, default=8.0)
    parser.add_argument("--learner-prefix", default="asp_")
    parser.add_argument("--judge-prefix", default="judge_asp_")

    parser.add_argument("--extractor", choices=["gumbel", "mmr"], default="gumbel")
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["online", "static", "random"],
        default=["online", "static", "random"],
    )
    parser.add_argument("--judge-aggregation", choices=["uniform", "learner_alpha"], default="uniform")
    parser.add_argument("--product-ids", nargs="*", default=None)
    parser.add_argument("--num-products", type=int, default=20)
    parser.add_argument("--min-candidates", type=int, default=30)
    parser.add_argument("--allow-history-products", action="store_true")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--run-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--length-budget", type=int, default=1000)
    parser.add_argument("--lam-mmr", type=float, default=0.95)
    parser.add_argument("--tau-alpha", type=float, default=20.0)
    parser.add_argument("--gamma", type=float, default=12.0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--eta-omd", type=float, default=2.59)
    parser.add_argument("--c-eta", type=float, default=0.05)
    parser.add_argument("--delta-omd", type=float, default=1e-4)
    parser.add_argument("--baseline-mode", choices=["ema", "running_mean"], default="running_mean")
    parser.add_argument("--ema-alpha", type=float, default=0.1)
    parser.add_argument("--clip-centered-feedback", type=float, default=0.15)
    parser.add_argument("--out-dir", default="outputs/cross_embedding_judge")
    parser.add_argument("--verbose-rounds", action="store_true")
    return parser.parse_args()


def reset_global_index(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop(columns=["global_idx"], errors="ignore").reset_index(drop=True)
    return ensure_global_index(data)


def select_evaluation_products(
    candidate_data: pd.DataFrame,
    user_history_products: set[str],
    args: argparse.Namespace,
) -> list[str]:
    values = candidate_data[args.product_col].astype(str)
    if args.product_ids:
        requested = [str(product) for product in args.product_ids]
        missing = [product for product in requested if not (values == product).any()]
        if missing:
            raise ValueError(f"Requested products are absent: {missing}")
        if not args.allow_history_products:
            overlap = sorted(set(requested) & user_history_products)
            if overlap:
                raise ValueError(
                    f"Evaluation products overlap user history: {overlap}. "
                    "Pass --allow-history-products to permit this."
                )
        return requested

    counts = candidate_data.groupby(values).size().sort_values(ascending=False)
    eligible = counts[counts >= args.min_candidates]
    if not args.allow_history_products:
        eligible = eligible[~eligible.index.astype(str).isin(user_history_products)]
    if eligible.empty:
        raise ValueError(
            "No eligible evaluation products remain. Lower --min-candidates or "
            "provide --product-ids."
        )
    return eligible.head(args.num_products).index.astype(str).tolist()


def make_product_sequence(products: Sequence[str], rounds: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    shuffled = list(products)
    rng.shuffle(shuffled)
    return list(islice(cycle(shuffled), rounds))


def make_extractor(
    args: argparse.Namespace,
    *,
    policy: str,
    seed: int,
    text_col: str,
):
    if policy == "random":
        return RandomExtractor(
            seed=seed,
            k=args.k,
            L=args.length_budget,
            text_col=text_col,
        )
    common = dict(
        k=args.k,
        L=args.length_budget,
        text_col=text_col,
        alpha_mode="utility_softmax",
        tau_alpha=args.tau_alpha,
    )
    if args.extractor == "gumbel":
        return GumbelExtractor(seed=seed, **common)
    return MMRExtractor(lam_mmr=args.lam_mmr, **common)


def mean_ci95(values: Iterable[float]) -> tuple[float, float, int]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, 0
    mean = float(arr.mean())
    if n == 1:
        return mean, np.nan, 1
    return mean, float(1.96 * arr.std(ddof=1) / np.sqrt(n)), n


def paired_statistics(values: Iterable[float], *, seed: int = 2026) -> dict:
    """Summarize paired policy differences without assuming independent runs."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return {
            "mean": np.nan,
            "ci95": np.nan,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
            "median": np.nan,
            "win_rate": np.nan,
            "p_ttest": np.nan,
            "p_wilcoxon": np.nan,
            "n": 0,
        }

    mean, ci95, _ = mean_ci95(arr)
    rng = np.random.default_rng(seed)
    bootstrap_means = np.mean(
        rng.choice(arr, size=(10000, n), replace=True), axis=1
    )
    boot_low, boot_high = np.quantile(bootstrap_means, [0.025, 0.975])

    if n >= 2 and np.std(arr, ddof=1) > 0:
        p_ttest = float(stats.ttest_1samp(arr, popmean=0.0).pvalue)
    else:
        p_ttest = 1.0 if np.allclose(arr, 0.0) else np.nan

    nonzero = arr[~np.isclose(arr, 0.0)]
    if len(nonzero) > 0:
        try:
            p_wilcoxon = float(stats.wilcoxon(nonzero, alternative="two-sided").pvalue)
        except ValueError:
            p_wilcoxon = np.nan
    else:
        p_wilcoxon = 1.0

    return {
        "mean": mean,
        "ci95": ci95,
        "bootstrap_ci_low": float(boot_low),
        "bootstrap_ci_high": float(boot_high),
        "median": float(np.median(arr)),
        "win_rate": float(np.mean(arr > 0.0)),
        "p_ttest": p_ttest,
        "p_wilcoxon": p_wilcoxon,
        "n": n,
    }


def aggregate_trajectory(runs: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for (policy, t), sub in runs.groupby(["policy", "t"], sort=True):
        mean, ci95, n = mean_ci95(sub[metric])
        rows.append(
            {"policy": policy, "t": int(t), "metric": metric, "mean": mean, "ci95": ci95, "n": n}
        )
    return pd.DataFrame(rows)


def plot_trajectory(summary: pd.DataFrame, *, ylabel: str, title: str, output: Path) -> None:
    plt.figure(figsize=(6.4, 4.0))
    for policy in summary["policy"].drop_duplicates():
        sub = summary[summary["policy"] == policy].sort_values("t")
        x = sub["t"].to_numpy(dtype=int)
        y = sub["mean"].to_numpy(dtype=float)
        ci = sub["ci95"].fillna(0.0).to_numpy(dtype=float)
        plt.plot(x, y, label=policy)
        plt.fill_between(x, y - ci, y + ci, alpha=0.2)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out_dir) / f"{args.extractor}_{args.init_mode}_{args.judge_aggregation}"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_data = pd.read_csv(args.data)
    required = {args.user_col, args.product_col}
    missing = sorted(required - set(raw_data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    text_col = infer_text_column(raw_data, args.text_col)
    learner_cols = aspect_columns(raw_data, args.learner_prefix)
    judge_cols = aspect_columns(raw_data, args.judge_prefix)
    if not learner_cols:
        raise ValueError(f"No learner columns with prefix {args.learner_prefix!r} were found.")
    if not judge_cols:
        raise ValueError(
            f"No judge columns with prefix {args.judge_prefix!r} were found. "
            "Run build_cross_embedding_judge.py first."
        )

    round_rows: list[dict] = []
    split_rows: list[dict] = []

    for user_id in args.user_ids:
        user_mask = raw_data[args.user_col].astype(str) == str(user_id)
        user_history = raw_data.loc[user_mask].copy()
        if user_history.empty:
            raise ValueError(f"User {user_id!r} is absent from the dataset.")
        history_products = set(user_history[args.product_col].astype(str))
        candidate_data = reset_global_index(raw_data.loc[~user_mask].copy())
        products = select_evaluation_products(candidate_data, history_products, args)

        print(f"\nUser {user_id}: {len(user_history)} history rows")
        print(f"Evaluation products ({len(products)}): {products[:5]}")

        for split_seed in args.split_seeds:
            split = split_user_history(
                raw_data,
                str(user_id),
                holdout_fraction=args.holdout_fraction,
                seed=split_seed,
                user_col=args.user_col,
                product_col=args.product_col,
                group_col=args.group_col,
                allow_row_split=args.allow_row_split,
            )
            balance_col = split.split_unit if split.split_unit in raw_data.columns else None
            w_fit_learner = fit_preference_from_aspects(
                split.fit_rows,
                learner_cols,
                beta=args.beta_init,
                balance_col=balance_col,
            )
            w_judge = fit_preference_from_aspects(
                split.holdout_rows,
                judge_cols,
                beta=args.beta_judge,
                balance_col=balance_col,
            )
            w_init = (
                w_fit_learner
                if args.init_mode == "fit"
                else np.ones(len(learner_cols), dtype=np.float32) / len(learner_cols)
            )

            split_rows.append(
                {
                    "user_id": str(user_id),
                    "split_seed": int(split_seed),
                    "split_unit": split.split_unit,
                    "fit_rows": len(split.fit_rows),
                    "holdout_rows": len(split.holdout_rows),
                    "fit_groups": len(split.fit_group_ids),
                    "holdout_groups": len(split.holdout_group_ids),
                    "learner_dimension": len(learner_cols),
                    "judge_dimension": len(judge_cols),
                    "w_fit_learner": json.dumps(w_fit_learner.tolist()),
                    "w_judge": json.dumps(w_judge.tolist()),
                    "note": "Vectors are in different spaces and are not compared.",
                }
            )

            for run_seed in args.run_seeds:
                combined_seed = int(
                    np.random.SeedSequence([split_seed, run_seed, 161803]).generate_state(1)[0]
                )
                sequence = make_product_sequence(products, args.rounds, combined_seed)

                for policy in args.policies:
                    print(
                        f"  split={split_seed}, run={run_seed}, policy={policy}, "
                        f"extractor={args.extractor}"
                    )
                    provider = CrossEmbeddingJudgeFeedbackProvider(
                        w_judge,
                        judge_cols,
                        gamma=args.gamma,
                        noise_std=args.noise_std,
                        seed=combined_seed,
                        threshold=args.threshold,
                        aggregation=args.judge_aggregation,
                    )
                    extractor = make_extractor(
                        args, policy=policy, seed=combined_seed, text_col=text_col
                    )
                    result = run_online_experiment(
                        df_sent=candidate_data,
                        user_id=str(user_id),
                        product_ids=sequence,
                        phi_cols=learner_cols,
                        extractor=extractor,
                        feedback_provider=provider,
                        method_name="random" if policy == "random" else args.extractor,
                        eta_omd=args.eta_omd,
                        delta_omd=args.delta_omd,
                        c_eta=args.c_eta,
                        baseline_mode=args.baseline_mode,
                        ema_alpha=args.ema_alpha,
                        clip_centered_feedback=args.clip_centered_feedback,
                        utility_lam=0.0,
                        use_policy="omd" if policy == "online" else "static",
                        synthetic_oracle=False,
                        seed=combined_seed,
                        w_init_override=w_init,
                        verbose=args.verbose_rounds,
                    )
                    if len(result.logs) != len(provider.records):
                        raise RuntimeError("Feedback diagnostics and experiment logs are misaligned.")

                    for log, judge_record in zip(result.logs, provider.records):
                        round_rows.append(
                            {
                                "user_id": str(user_id),
                                "split_seed": int(split_seed),
                                "run_seed": int(run_seed),
                                "policy": policy,
                                "extractor": "random" if policy == "random" else args.extractor,
                                "init_mode": args.init_mode,
                                "judge_aggregation": args.judge_aggregation,
                                "t": int(log["t"]) + 1,
                                "product_id": str(log["product_id"]),
                                "feedback": float(log["feedback"]),
                                "judge_utility": float(judge_record["judge_utility"]),
                                "judge_threshold": float(judge_record["judge_threshold"]),
                                "judge_noise": float(judge_record["judge_noise"]),
                                "centered_feedback": float(log["omd_f_eff"]),
                                "baseline": float(log["omd_baseline"]),
                                "learner_z": json.dumps(np.asarray(log["z_t"], dtype=float).tolist()),
                                "judge_z": json.dumps(
                                    np.asarray(judge_record["judge_z"], dtype=float).tolist()
                                ),
                                "learner_w": json.dumps(
                                    np.asarray(log["w_omd"], dtype=float).tolist()
                                ),
                                "selected_global_idx": json.dumps(
                                    np.asarray(log["selected_global_idx"], dtype=int).tolist()
                                ),
                            }
                        )

    runs = pd.DataFrame(round_rows)
    splits = pd.DataFrame(split_rows)
    if runs.empty:
        raise RuntimeError("No valid experiment rounds were produced.")
    runs.to_csv(output_dir / "cross_embedding_rounds.csv", index=False)
    splits.to_csv(output_dir / "cross_embedding_splits.csv", index=False)

    feedback_summary = aggregate_trajectory(runs, "feedback")
    utility_summary = aggregate_trajectory(runs, "judge_utility")
    feedback_summary.to_csv(output_dir / "feedback_trajectory_summary.csv", index=False)
    utility_summary.to_csv(output_dir / "judge_utility_trajectory_summary.csv", index=False)
    plot_trajectory(
        feedback_summary,
        ylabel="Independent-judge feedback",
        title=f"Cross-embedding judge: {args.extractor}, initialization={args.init_mode}",
        output=output_dir / "feedback_trajectory.pdf",
    )
    plot_trajectory(
        utility_summary,
        ylabel="Independent-judge utility",
        title=f"Cross-embedding utility: {args.extractor}, initialization={args.init_mode}",
        output=output_dir / "judge_utility_trajectory.pdf",
    )

    window = max(5, args.rounds // 5)
    replicate = (
        runs.groupby(["user_id", "split_seed", "run_seed", "policy"], as_index=False)
        .agg(
            mean_feedback=("feedback", "mean"),
            mean_utility=("judge_utility", "mean"),
            initial_feedback=("feedback", lambda x: x.iloc[:window].mean()),
            final_feedback=("feedback", lambda x: x.iloc[-window:].mean()),
            initial_utility=("judge_utility", lambda x: x.iloc[:window].mean()),
            final_utility=("judge_utility", lambda x: x.iloc[-window:].mean()),
        )
    )
    replicate["feedback_improvement"] = replicate["final_feedback"] - replicate["initial_feedback"]
    replicate["utility_improvement"] = replicate["final_utility"] - replicate["initial_utility"]
    replicate.to_csv(output_dir / "replicate_summary.csv", index=False)

    paired = replicate.pivot(
        index=["user_id", "split_seed", "run_seed"],
        columns="policy",
        values=[
            "mean_feedback",
            "mean_utility",
            "initial_feedback",
            "final_feedback",
            "initial_utility",
            "final_utility",
            "feedback_improvement",
            "utility_improvement",
        ],
    )
    paired.columns = [f"{metric}_{policy}" for metric, policy in paired.columns]
    paired = paired.reset_index()
    comparisons = []
    for baseline in ("static", "random"):
        if {"online", baseline}.issubset(set(args.policies)):
            for metric in (
                "mean_feedback",
                "mean_utility",
                "final_feedback",
                "final_utility",
                "feedback_improvement",
                "utility_improvement",
            ):
                column = f"online_minus_{baseline}_{metric}"
                paired[column] = paired[f"{metric}_online"] - paired[f"{metric}_{baseline}"]
                comparisons.append(column)
    paired.to_csv(output_dir / "paired_policy_comparisons.csv", index=False)

    aggregate_rows = []
    for index, column in enumerate(comparisons):
        summary = paired_statistics(paired[column], seed=2026 + index)
        aggregate_rows.append({"quantity": column, **summary})
    pd.DataFrame(aggregate_rows).to_csv(
        output_dir / "paired_policy_comparisons_aggregate.csv", index=False
    )

    metadata = vars(args).copy()
    metadata.update(
        {
            "learner_cols": learner_cols,
            "judge_cols": judge_cols,
            "text_col_resolved": text_col,
            "window": window,
            "valid_claim": (
                "Empirical representation-mismatch robustness: whether learner-space "
                "online updates improve feedback from an independent judge representation."
            ),
            "invalid_claims": [
                "The judge embedding is ground truth.",
                "Learner and judge preference vectors are directly comparable.",
                "The original shared-space regret theorem is verified by this experiment.",
            ],
        }
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nSaved outputs to:", output_dir)
    print("Primary result:", output_dir / "paired_policy_comparisons_aggregate.csv")


if __name__ == "__main__":
    main()

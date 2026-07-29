"""Run the PREFER held-out-judge experiment.

Run from the repository root, for example:

    python scripts/run_heldout_judge.py \
        --data data/train_with_aspect_scores_sentences.csv \
        --user-ids AG73BVBKUOH22USSFJA5ZWL7AKXA \
        --extractor gumbel \
        --init-mode fit \
        --rounds 100 \
        --split-seeds 0 1 2 3 4 \
        --run-seeds 0 1 2 3 4

``--init-mode fit`` implements the literal train/held-out design: the learner is
initialized from one half of the user's history and the hidden judge is fitted
from the other half.  ``--init-mode uniform`` matches the current paper's
uniform initialization while retaining the held-out evaluator.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pacer_feedback_exp.experiment import run_online_experiment
from pacer_feedback_exp.extractors.gumbel import GumbelExtractor
from pacer_feedback_exp.extractors.mmr import MMRExtractor
from pacer_feedback_exp.heldout_feedback import (
    HeldOutJudgeFeedbackProvider,
    cosine,
    fit_aspect_preference,
    jensen_shannon_divergence,
    split_user_history,
)
from pacer_feedback_exp.utils import ensure_global_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PREFER against a judge fitted on disjoint user history."
    )
    parser.add_argument(
        "--data", default="data/train_with_aspect_scores_sentences.csv"
    )
    parser.add_argument(
        "--user-ids",
        nargs="+",
        default=["AG73BVBKUOH22USSFJA5ZWL7AKXA"],
    )
    parser.add_argument("--user-col", default="user_id")
    parser.add_argument("--product-col", default="parent_asin")
    parser.add_argument("--text-col", default=None)
    parser.add_argument(
        "--group-col",
        default=None,
        help="Review identifier used for the split. Omit for automatic inference.",
    )
    parser.add_argument("--allow-row-split", action="store_true")
    parser.add_argument("--holdout-fraction", type=float, default=0.5)
    parser.add_argument("--init-mode", choices=["fit", "uniform"], default="fit")
    parser.add_argument("--beta-judge", type=float, default=8.0)
    parser.add_argument("--beta-init", type=float, default=8.0)

    parser.add_argument("--extractor", choices=["gumbel", "mmr"], default="gumbel")
    parser.add_argument("--policies", nargs="+", choices=["online", "static"], default=["online", "static"])
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
    parser.add_argument("--eta-omd", type=float, default=2.59)
    parser.add_argument("--c-eta", type=float, default=0.05)
    parser.add_argument("--delta-omd", type=float, default=1e-4)
    parser.add_argument("--baseline-mode", choices=["ema", "running_mean"], default="running_mean")
    parser.add_argument("--ema-alpha", type=float, default=0.1)
    parser.add_argument("--clip-centered-feedback", type=float, default=0.15)
    parser.add_argument("--out-dir", default="outputs/heldout_judge")
    parser.add_argument("--verbose-rounds", action="store_true")
    return parser.parse_args()


def infer_text_col(data: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in data.columns:
            raise ValueError(f"Requested text column {requested!r} is absent.")
        return requested
    for candidate in ("review_text", "sentence", "text"):
        if candidate in data.columns:
            return candidate
    raise ValueError("Could not infer a sentence-text column; pass --text-col.")


def reset_global_index(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop(columns=["global_idx"], errors="ignore").reset_index(drop=True)
    return ensure_global_index(data)


def select_evaluation_products(
    candidate_data: pd.DataFrame,
    user_history_products: set[str],
    args: argparse.Namespace,
) -> list[str]:
    product_values = candidate_data[args.product_col].astype(str)

    if args.product_ids:
        requested = [str(p) for p in args.product_ids]
        missing = [p for p in requested if not (product_values == p).any()]
        if missing:
            raise ValueError(f"Requested products are absent from candidate data: {missing}")
        if not args.allow_history_products:
            overlap = sorted(set(requested) & user_history_products)
            if overlap:
                raise ValueError(
                    "Requested evaluation products overlap the target user's history: "
                    f"{overlap}. Pass --allow-history-products to permit this."
                )
        return requested

    counts = candidate_data.groupby(product_values).size().sort_values(ascending=False)
    eligible = counts[counts >= args.min_candidates]
    if not args.allow_history_products:
        eligible = eligible[~eligible.index.astype(str).isin(user_history_products)]
    if eligible.empty:
        raise ValueError(
            "No eligible evaluation product remains. Lower --min-candidates, provide "
            "--product-ids, or pass --allow-history-products."
        )
    return eligible.head(args.num_products).index.astype(str).tolist()


def make_product_sequence(products: Sequence[str], rounds: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    shuffled = list(products)
    rng.shuffle(shuffled)
    return list(islice(cycle(shuffled), rounds))


def make_extractor(args: argparse.Namespace, *, seed: int, text_col: str):
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
    ci = float(1.96 * arr.std(ddof=1) / np.sqrt(n))
    return mean, ci, n


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

    # Parse command-line arguments and prepare output directory
    args = parse_args()
    output_dir = Path(args.out_dir) / f"{args.extractor}_{args.init_mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the dataset and validate required columns
    raw_data = pd.read_csv(args.data)
    required = {args.user_col, args.product_col}
    missing_required = sorted(required - set(raw_data.columns))
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    # Infer the text column and identify aspect columns
    text_col = infer_text_col(raw_data, args.text_col)
    phi_cols = sorted(c for c in raw_data.columns if c.startswith("asp_"))
    if not phi_cols:
        raise ValueError("No asp_* aspect columns were found.")

    all_round_rows: list[dict] = []
    split_rows: list[dict] = []

    # Iterate over each specified user ID and run the held-out judge experiment
    for user_id in args.user_ids:
        user_mask = raw_data[args.user_col].astype(str) == str(user_id)
        user_history = raw_data.loc[user_mask].copy()
        if user_history.empty:
            raise ValueError(f"User {user_id!r} is absent from the dataset.")

        user_history_products = set(user_history[args.product_col].astype(str).tolist())

        # The target user's own sentences are never candidate evidence.  This
        # prevents direct leakage from held-out judge text into selected sets.
        candidate_data = reset_global_index(raw_data.loc[~user_mask].copy())
        products = select_evaluation_products(candidate_data, user_history_products, args)

        print(f"\nUser {user_id}: {len(user_history)} history rows")
        print(f"Evaluation products ({len(products)}): {products[:5]}{' ...' if len(products) > 5 else ''}")

        for split_seed in args.split_seeds:

            # Split the user's history into fit and holdout sets,
            # fit aspect preferences, and prepare for online experiments
            split = split_user_history(
                raw_data,
                user_id,
                holdout_fraction=args.holdout_fraction,
                seed=split_seed,
                user_col=args.user_col,
                product_col=args.product_col,
                group_col=args.group_col,
                allow_row_split=args.allow_row_split,
            )

            # Fit the aspect preference weights for both the fit and holdout sets
            balance_col = split.split_unit if split.split_unit in raw_data.columns else None
            w_fit = fit_aspect_preference(
                split.fit_rows,
                phi_cols,
                beta=args.beta_init,
                balance_col=balance_col,
            )
            w_judge = fit_aspect_preference(
                split.holdout_rows,
                phi_cols,
                beta=args.beta_judge,
                balance_col=balance_col,
            )
            if args.init_mode == "fit":
                w_init = w_fit
            else:
                w_init = np.ones(len(phi_cols), dtype=np.float32) / len(phi_cols)

            split_rows.append(
                {
                    "user_id": str(user_id),
                    "split_seed": int(split_seed),
                    "split_unit": split.split_unit,
                    "fit_rows": len(split.fit_rows),
                    "holdout_rows": len(split.holdout_rows),
                    "fit_groups": len(split.fit_group_ids),
                    "holdout_groups": len(split.holdout_group_ids),
                    "fit_judge_cosine": cosine(w_fit, w_judge),
                    "fit_judge_js_divergence": jensen_shannon_divergence(w_fit, w_judge),
                    "w_fit": json.dumps(w_fit.tolist()),
                    "w_judge": json.dumps(w_judge.tolist()),
                }
            )

            for run_seed in args.run_seeds:
                combined_seed = int(
                    np.random.SeedSequence([split_seed, run_seed, 271828]).generate_state(1)[0]
                )
                product_sequence = make_product_sequence(products, args.rounds, combined_seed)

                for policy in args.policies:
                    print(
                        f"  split={split_seed}, run={run_seed}, policy={policy}, "
                        f"extractor={args.extractor}"
                    )
                    provider = HeldOutJudgeFeedbackProvider(
                        w_judge,
                        gamma=args.gamma,
                        noise_std=args.noise_std,
                        seed=combined_seed,
                    )
                    extractor = make_extractor(args, seed=combined_seed, text_col=text_col)

                    result = run_online_experiment(
                        df_sent=candidate_data,
                        user_id=str(user_id),
                        product_ids=product_sequence,
                        phi_cols=phi_cols,
                        extractor=extractor,
                        feedback_provider=provider,
                        method_name=args.extractor,
                        eta_omd=args.eta_omd,
                        delta_omd=args.delta_omd,
                        c_eta=args.c_eta,
                        baseline_mode=args.baseline_mode,
                        ema_alpha=args.ema_alpha,
                        clip_centered_feedback=args.clip_centered_feedback,
                        utility_lam=0.0,
                        use_policy="omd" if policy == "online" else "static",
                        synthetic_oracle=True,
                        seed=combined_seed,
                        w_init_override=w_init,
                        verbose=args.verbose_rounds,
                    )

                    for log in result.logs:
                        z_t = np.asarray(log["z_t"], dtype=float)
                        learned_w = np.asarray(log["w_omd"], dtype=float)
                        all_round_rows.append(
                            {
                                "user_id": str(user_id),
                                "split_seed": int(split_seed),
                                "run_seed": int(run_seed),
                                "policy": policy,
                                "extractor": args.extractor,
                                "init_mode": args.init_mode,
                                "t": int(log["t"]) + 1,
                                "product_id": str(log["product_id"]),
                                "feedback": float(log["feedback"]),
                                "judge_utility": float(np.dot(w_judge, z_t)),
                                "cos_evidence_judge": cosine(z_t, w_judge),
                                "cos_preference_judge": cosine(learned_w, w_judge),
                                "centered_feedback": float(log["omd_f_eff"]),
                                "baseline": float(log["omd_baseline"]),
                                "selected_global_idx": json.dumps(
                                    np.asarray(log["selected_global_idx"], dtype=int).tolist()
                                ),
                            }
                        )

    runs = pd.DataFrame(all_round_rows)
    splits = pd.DataFrame(split_rows)
    if runs.empty:
        raise RuntimeError("No valid experiment rounds were produced.")

    runs.to_csv(output_dir / "heldout_rounds.csv", index=False)
    splits.to_csv(output_dir / "heldout_splits.csv", index=False)

    feedback_summary = aggregate_trajectory(runs, "feedback")
    utility_summary = aggregate_trajectory(runs, "judge_utility")
    feedback_summary.to_csv(output_dir / "feedback_trajectory_summary.csv", index=False)
    utility_summary.to_csv(output_dir / "utility_trajectory_summary.csv", index=False)

    plot_trajectory(
        feedback_summary,
        ylabel="Held-out judge feedback",
        title=f"Held-out judge: {args.extractor}, initialization={args.init_mode}",
        output=output_dir / "feedback_trajectory.pdf",
    )
    plot_trajectory(
        utility_summary,
        ylabel=r"Held-out utility $w_{u}^{J\top} z_t$",
        title=f"Held-out utility: {args.extractor}, initialization={args.init_mode}",
        output=output_dir / "utility_trajectory.pdf",
    )

    window = max(5, args.rounds // 5)
    replicate = (
        runs.groupby(["user_id", "split_seed", "run_seed", "policy"], as_index=False)
        .agg(
            mean_feedback=("feedback", "mean"),
            mean_utility=("judge_utility", "mean"),
            final_feedback=("feedback", lambda x: x.iloc[-window:].mean()),
            initial_feedback=("feedback", lambda x: x.iloc[:window].mean()),
            final_utility=("judge_utility", lambda x: x.iloc[-window:].mean()),
            initial_utility=("judge_utility", lambda x: x.iloc[:window].mean()),
        )
    )
    replicate["feedback_improvement"] = replicate["final_feedback"] - replicate["initial_feedback"]
    replicate["utility_improvement"] = replicate["final_utility"] - replicate["initial_utility"]
    replicate.to_csv(output_dir / "replicate_summary.csv", index=False)

    paired = replicate.pivot(
        index=["user_id", "split_seed", "run_seed"],
        columns="policy",
        values=["mean_feedback", "mean_utility", "feedback_improvement", "utility_improvement"],
    )
    paired.columns = [f"{metric}_{policy}" for metric, policy in paired.columns]
    paired = paired.reset_index()

    if {"online", "static"}.issubset(set(args.policies)):
        paired["online_minus_static_feedback"] = (
            paired["mean_feedback_online"] - paired["mean_feedback_static"]
        )
        paired["online_minus_static_utility"] = (
            paired["mean_utility_online"] - paired["mean_utility_static"]
        )
        paired["online_minus_static_feedback_improvement"] = (
            paired["feedback_improvement_online"] - paired["feedback_improvement_static"]
        )
        paired["online_minus_static_utility_improvement"] = (
            paired["utility_improvement_online"] - paired["utility_improvement_static"]
        )
    paired.to_csv(output_dir / "paired_online_static.csv", index=False)

    aggregate_rows = []
    for column in [c for c in paired.columns if c.startswith("online_minus_static")]:
        mean, ci95, n = mean_ci95(paired[column])
        aggregate_rows.append({"quantity": column, "mean": mean, "ci95": ci95, "n": n})
    pd.DataFrame(aggregate_rows).to_csv(
        output_dir / "paired_online_static_aggregate.csv", index=False
    )

    metadata = vars(args).copy()
    metadata.update(
        {
            "phi_cols": phi_cols,
            "text_col_resolved": text_col,
            "window": window,
            "primary_interpretation": (
                "Data-level generalization within the same PREFER latent aspect space."
            ),
        }
    )
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("\nSaved outputs to:", output_dir)
    print("Primary paired result:", output_dir / "paired_online_static_aggregate.csv")


if __name__ == "__main__":
    main()

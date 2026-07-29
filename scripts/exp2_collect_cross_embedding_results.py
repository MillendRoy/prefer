"""Collect cross-embedding experiment folders into paper-ready CSV/LaTeX tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/cross_embedding_judge")
    parser.add_argument(
        "--out-prefix", default="outputs/cross_embedding_judge/cross_embedding_paper_table"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = []
    for aggregate_path in sorted(root.rglob("paired_policy_comparisons_aggregate.csv")):
        metadata_path = aggregate_path.parent / "run_metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        table = pd.read_csv(aggregate_path)
        for row in table.to_dict(orient="records"):
            rows.append(
                {
                    "extractor": metadata["extractor"],
                    "init_mode": metadata["init_mode"],
                    "judge_aggregation": metadata["judge_aggregation"],
                    "rounds": metadata["rounds"],
                    "split_seeds": len(metadata["split_seeds"]),
                    "run_seeds": len(metadata["run_seeds"]),
                    **row,
                }
            )

    if not rows:
        raise FileNotFoundError(f"No aggregate result files found below {root}.")

    combined = pd.DataFrame(rows)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_prefix.with_suffix(".csv"), index=False)

    primary_quantities = {
        "online_minus_static_mean_feedback",
        "online_minus_static_mean_utility",
        "online_minus_random_mean_feedback",
        "online_minus_random_mean_utility",
    }
    primary = combined[
        (combined["init_mode"] == "fit")
        & (combined["judge_aggregation"] == "uniform")
        & combined["quantity"].isin(primary_quantities)
    ].copy()
    primary["estimate_95ci"] = primary.apply(
        lambda row: (
            f"{row['mean']:.4f} "
            f"[{row['bootstrap_ci_low']:.4f}, {row['bootstrap_ci_high']:.4f}]"
        ),
        axis=1,
    )
    paper = primary.pivot(
        index="extractor", columns="quantity", values="estimate_95ci"
    ).reset_index()
    paper.to_csv(out_prefix.with_name(out_prefix.name + "_primary.csv"), index=False)
    paper.to_latex(
        out_prefix.with_name(out_prefix.name + "_primary.tex"),
        index=False,
        escape=False,
    )
    print("Saved:", out_prefix.with_suffix(".csv"))
    print("Saved:", out_prefix.with_name(out_prefix.name + "_primary.csv"))
    print("Saved:", out_prefix.with_name(out_prefix.name + "_primary.tex"))


if __name__ == "__main__":
    main()

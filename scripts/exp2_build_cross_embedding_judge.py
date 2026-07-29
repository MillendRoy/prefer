"""Build an independent embedding/PCA/clustering representation for the judge.

Example:
    python scripts/build_cross_embedding_judge.py \
      --data data/train_with_aspect_scores_sentences.csv \
      --out-data data/cross_embedding/train_with_cross_judge.csv \
      --model-dir artifacts/cross_embedding_judge \
      --exclude-user-ids AG73BVBKUOH22USSFJA5ZWL7AKXA
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pacer_feedback_exp.cross_embedding import (
    IndependentJudgeAspectModel,
    infer_text_column,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and apply an independent aspect representation for the judge."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-data", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text-col", default=None)
    parser.add_argument("--user-col", default="user_id")
    parser.add_argument("--exclude-user-ids", nargs="*", default=[])
    parser.add_argument(
        "--encoder",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Must differ from the learner encoder for the main stress test.",
    )
    parser.add_argument("--embeddings-npy", default=None)
    parser.add_argument("--n-aspects", type=int, default=10)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--fit-sample-size", type=int, default=100000)
    parser.add_argument("--transform-chunk-size", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--assignment-ratio", type=float, default=4.0)
    parser.add_argument("--prefix", default="judge_asp_")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.data)
    text_col = infer_text_column(data, args.text_col)

    eligible = np.ones(len(data), dtype=bool)
    if args.exclude_user_ids:
        if args.user_col not in data.columns:
            raise ValueError(f"User column {args.user_col!r} is absent.")
        excluded = {str(user_id) for user_id in args.exclude_user_ids}
        eligible &= ~data[args.user_col].astype(str).isin(excluded).to_numpy()

    eligible_indices = np.flatnonzero(eligible)
    if len(eligible_indices) < args.n_aspects:
        raise ValueError("Too few eligible rows to fit the requested judge aspects.")

    rng = np.random.default_rng(args.seed)
    fit_size = min(int(args.fit_sample_size), len(eligible_indices))
    fit_indices = rng.choice(eligible_indices, size=fit_size, replace=False)

    all_embeddings = None
    if args.embeddings_npy is not None:
        all_embeddings = np.load(args.embeddings_npy)
        if len(all_embeddings) != len(data):
            raise ValueError("--embeddings-npy must have one row per input CSV row.")
        model = IndependentJudgeAspectModel.fit_from_embeddings(
            all_embeddings[fit_indices],
            encoder_name=args.encoder,
            n_aspects=args.n_aspects,
            pca_components=args.pca_components,
            target_assignment_ratio=args.assignment_ratio,
            prefix=args.prefix,
            seed=args.seed,
        )
    else:
        fit_texts = data.iloc[fit_indices][text_col].fillna("").astype(str).tolist()
        model = IndependentJudgeAspectModel.fit_from_texts(
            fit_texts,
            encoder_name=args.encoder,
            n_aspects=args.n_aspects,
            pca_components=args.pca_components,
            target_assignment_ratio=args.assignment_ratio,
            prefix=args.prefix,
            seed=args.seed,
            batch_size=args.batch_size,
            device=args.device,
        )

    model_dir = Path(args.model_dir)
    model.save(model_dir)

    chunks: list[np.ndarray] = []
    chunk_size = max(1, int(args.transform_chunk_size))
    for start in range(0, len(data), chunk_size):
        end = min(start + chunk_size, len(data))
        print(f"Transforming rows {start}:{end} of {len(data)}")
        if all_embeddings is not None:
            phi = model.transform_embeddings(all_embeddings[start:end])
        else:
            texts = data.iloc[start:end][text_col].fillna("").astype(str).tolist()
            phi = model.transform_texts(
                texts,
                batch_size=args.batch_size,
                device=args.device,
            )
        chunks.append(phi)

    judge_phi = np.vstack(chunks)
    for index, column in enumerate(model.columns):
        data[column] = judge_phi[:, index]

    out_path = Path(args.out_data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_path, index=False)

    build_metadata = {
        "input_data": str(Path(args.data)),
        "output_data": str(out_path),
        "text_col": text_col,
        "encoder": args.encoder,
        "learner_embedding_expected": "sentence-transformers/all-MiniLM-L6-v2",
        "n_rows": len(data),
        "n_fit_rows": len(fit_indices),
        "excluded_user_ids": args.exclude_user_ids,
        "judge_columns": model.columns,
        "assignment_tau": model.assignment_tau,
        "seed": args.seed,
        "representation_role": "independent cross-embedding evaluator, not ground truth",
    }
    (model_dir / "build_metadata.json").write_text(
        json.dumps(build_metadata, indent=2), encoding="utf-8"
    )
    print("Saved enriched data to:", out_path)
    print("Saved judge model to:", model_dir)


if __name__ == "__main__":
    main()

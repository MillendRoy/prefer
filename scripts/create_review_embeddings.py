# create_review_embeddings.py
"""
Create review-level embedding matrices from previously saved train/valid/test CSV files.

Assumes the following files already exist:
    train_df.csv
    valid_df.csv
    test_df.csv

Outputs:
    review_embeddings_train.npy
    review_embeddings_valid.npy
    review_embeddings_test.npy
    compute_report.jsonl
"""

import argparse
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from compute_report import compute_tracker


def parse_args():
    parser = argparse.ArgumentParser(description="Create review-level embeddings.")
    parser.add_argument("--input_dir", type=str, default="../data", help="Directory containing train_df.csv, valid_df.csv, test_df.csv")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory where .npy files will be saved")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--text_col", type=str, default="review_text")
    parser.add_argument("--compute_report", type=str, default="output/compute_report_reviews.jsonl")
    return parser.parse_args()


def load_split(input_dir, split):
    path = os.path.join(input_dir, f"{split}_df.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}")
    return pd.read_csv(path)


def create_review_embeddings(df, split, embedder, args):
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found in {split}_df.csv")

    texts = df[args.text_col].fillna("").astype(str).tolist()

    with compute_tracker(
        f"{split}_review_embedding_creation",
        output_path=args.compute_report,
        extra={
            "dataset": "Amazon Reviews 2023 ALL_BEAUTY",
            "granularity": "review",
            "split": split,
            "num_reviews": len(texts),
            "embedding_model": args.model_name,
            "embedding_dim": 384,
            "batch_size": args.batch_size,
            "normalize_embeddings": True,
        },
    ):
        embeddings = embedder.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    print(f"{split} review embedding shape:", embeddings.shape)

    output_file = os.path.join(args.output_dir, f"review_embeddings_{split}.npy")
    with compute_tracker(
        f"{split}_review_embedding_npy_save",
        output_path=args.compute_report,
        extra={
            "output_file": output_file,
            "embedding_shape": embeddings.shape,
            "dtype": str(embeddings.dtype),
        },
    ):
        np.save(output_file, embeddings)

    return embeddings


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    embedder = SentenceTransformer(args.model_name)

    for split in ["train"]:
        df = load_split(args.input_dir, split)
        create_review_embeddings(df, split, embedder, args)


if __name__ == "__main__":
    main()

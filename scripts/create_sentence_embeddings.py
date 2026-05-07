# create_sentence_embeddings.py
"""
Create sentence-level dataset and sentence-level embedding matrix from saved CSV files.

By default, this script builds sentence-level data only from train_df.csv,
because aspect discovery is usually trained on the training split only.

Inputs:
    train_df.csv
    optionally valid_df.csv and test_df.csv if --splits train valid test is used

Outputs:
    sent_df_train.csv
    sentence_embeddings_train.npy
    compute_report.jsonl
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from compute_report import compute_tracker


def parse_args():
    parser = argparse.ArgumentParser(description="Create sentence-level dataset and embeddings.")
    parser.add_argument("--input_dir", type=str, default="../data", help="Directory containing train_df.csv, valid_df.csv, test_df.csv")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory where outputs will be saved")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--text_col", type=str, default="review_text")
    parser.add_argument("--max_sent_per_review", type=int, default=10)
    parser.add_argument("--min_len", type=int, default=20)
    parser.add_argument("--splits", nargs="+", default=["train"], choices=["train", "valid", "test"])
    parser.add_argument("--compute_report", type=str, default="output/compute_report_sentences.jsonl")
    return parser.parse_args()


def ensure_nltk_tokenizer():
    """Use NLTK sentence tokenizer, downloading punkt resources if needed."""
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except Exception:
            pass
    from nltk.tokenize import sent_tokenize
    return sent_tokenize


def clean_text(x):
    return re.sub(r"\s+", " ", str(x)).strip()


def load_split(input_dir, split):
    path = os.path.join(input_dir, f"{split}_df.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}")
    return pd.read_csv(path)


def build_sentence_table(reviews_df, split, text_col, sent_tokenize, max_sent_per_review=10, min_len=20):
    """
    Convert review-level dataframe into sentence-level dataframe.

    One review can produce multiple sentence rows. Each row keeps metadata
    needed to trace the sentence back to the original user/product/review.
    """
    if text_col not in reviews_df.columns:
        raise ValueError(f"Column '{text_col}' not found in dataframe")

    rows = []
    for review_index, r in tqdm(reviews_df.iterrows(), total=len(reviews_df), desc=f"Building sentence table: {split}"):
        text = clean_text(r.get(text_col, ""))
        if not text:
            continue

        sentences = [s.strip() for s in sent_tokenize(text)]
        sentences = [s for s in sentences if len(s) >= min_len]
        sentences = sentences[:max_sent_per_review]

        for sentence_id, sentence in enumerate(sentences):
            row = {
                "split": split,
                "review_index": review_index,
                "sentence_id": sentence_id,
                "sentence": sentence,
            }

            # Keep useful identifiers/metadata when available.
            for col in [
                "user_id",
                "parent_asin",
                "rating",
                "timestamp",
                "verified_purchase",
                "helpful_vote",
                "review_title",
                "item_title",
                "main_category",
            ]:
                if col in reviews_df.columns:
                    row[col] = r.get(col)

            rows.append(row)

    return pd.DataFrame(rows)


def create_sentence_embeddings(sent_df, split, embedder, args):
    sentences = sent_df["sentence"].fillna("").astype(str).tolist()

    with compute_tracker(
        f"{split}_sentence_embedding_creation",
        output_path=args.compute_report,
        extra={
            "dataset": "Amazon Reviews 2023 ALL_BEAUTY",
            "granularity": "sentence",
            "split": split,
            "num_sentences": len(sentences),
            "embedding_model": args.model_name,
            "embedding_dim": 384,
            "batch_size": args.batch_size,
            "normalize_embeddings": True,
        },
    ):
        embeddings = embedder.encode(
            sentences,
            batch_size=args.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    print(f"{split} sentence embedding shape:", embeddings.shape)

    output_file = os.path.join(args.output_dir, f"sentence_embeddings_{split}.npy")
    with compute_tracker(
        f"{split}_sentence_embedding_npy_save",
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

    sent_tokenize = ensure_nltk_tokenizer()
    embedder = SentenceTransformer(args.model_name)

    for split in args.splits:
        df = load_split(args.input_dir, split)

        with compute_tracker(
            f"{split}_sentence_table_creation",
            output_path=args.compute_report,
            extra={
                "dataset": "Amazon Reviews 2023 ALL_BEAUTY",
                "split": split,
                "num_reviews_input": len(df),
                "max_sent_per_review": args.max_sent_per_review,
                "min_sentence_length": args.min_len,
            },
        ):
            sent_df = build_sentence_table(
                reviews_df=df,
                split=split,
                text_col=args.text_col,
                sent_tokenize=sent_tokenize,
                max_sent_per_review=args.max_sent_per_review,
                min_len=args.min_len,
            )

        sent_csv = os.path.join(args.output_dir, f"sent_df_{split}.csv")
        with compute_tracker(
            f"{split}_sentence_table_csv_save",
            output_path=args.compute_report,
            extra={
                "output_file": sent_csv,
                "num_sentence_rows": len(sent_df),
            },
        ):
            sent_df.to_csv(sent_csv, index=False)

        create_sentence_embeddings(sent_df, split, embedder, args)


if __name__ == "__main__":
    main()

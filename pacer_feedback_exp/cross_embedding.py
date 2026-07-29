"""Independent-representation judge utilities for PREFER.

The learner selects evidence in the existing learner aspect space (``asp_*``).
The judge evaluates the selected raw sentences in a separately fitted aspect
space (for example, an MPNet encoder followed by an independent PCA/K-means
pipeline).  No learner/judge preference-vector comparison is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from .feedback import FeedbackProvider
from .utils import normalize_simplex, sigmoid, softmax


def infer_text_column(data: pd.DataFrame, requested: Optional[str] = None) -> str:
    """Resolve the sentence-text column used by both learner and judge."""
    if requested is not None:
        if requested not in data.columns:
            raise ValueError(f"Requested text column {requested!r} is absent.")
        return requested
    for candidate in ("sentence", "review_text", "text"):
        if candidate in data.columns:
            return candidate
    raise ValueError("Could not infer a text column; pass --text-col explicitly.")


def aspect_columns(data: pd.DataFrame, prefix: str) -> list[str]:
    """Return aspect columns in numeric suffix order when possible."""
    cols = [c for c in data.columns if c.startswith(prefix)]
    if not cols:
        return []

    def key(col: str):
        suffix = col[len(prefix) :]
        try:
            return (0, int(suffix))
        except ValueError:
            return (1, suffix)

    return sorted(cols, key=key)


def calibrate_assignment_temperature(
    squared_distances: np.ndarray,
    *,
    target_nearest_to_second_ratio: float = 4.0,
    eps: float = 1e-12,
) -> float:
    """Calibrate soft-cluster temperature from nearest-centroid gaps.

    With ``phi_k(x) proportional to exp(-tau * d_k(x))``, the ratio between
    the nearest and second-nearest assignments is

        exp(tau * (d_(2) - d_(1))).

    We choose tau so the median sentence has the requested ratio.
    """
    distances = np.asarray(squared_distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[1] < 2:
        raise ValueError("At least two centroid distances are required.")
    if target_nearest_to_second_ratio <= 1.0:
        raise ValueError("target_nearest_to_second_ratio must exceed 1.")

    nearest_two = np.partition(distances, kth=1, axis=1)[:, :2]
    nearest_two.sort(axis=1)
    gaps = nearest_two[:, 1] - nearest_two[:, 0]
    positive = gaps[np.isfinite(gaps) & (gaps > eps)]
    if len(positive) == 0:
        return 1.0
    median_gap = float(np.median(positive))
    return float(np.log(target_nearest_to_second_ratio) / max(median_gap, eps))


def soft_assign_distances(squared_distances: np.ndarray, tau: float) -> np.ndarray:
    """Convert centroid distances to simplex-valued aspect memberships."""
    distances = np.asarray(squared_distances, dtype=np.float64)
    if distances.ndim != 2:
        raise ValueError("squared_distances must be a two-dimensional array.")
    phi = softmax(-float(tau) * distances, axis=1)
    row_sums = phi.sum(axis=1, keepdims=True)
    return (phi / np.maximum(row_sums, 1e-12)).astype(np.float32)


@dataclass
class IndependentJudgeAspectModel:
    """Serializable independent embedding/PCA/clustering judge representation."""

    encoder_name: str
    pca: PCA
    clusterer: MiniBatchKMeans
    assignment_tau: float
    normalize_embeddings: bool = True
    prefix: str = "judge_asp_"
    fit_seed: int = 2026

    @property
    def n_aspects(self) -> int:
        return int(self.clusterer.n_clusters)

    @property
    def columns(self) -> list[str]:
        return [f"{self.prefix}{k}" for k in range(self.n_aspects)]

    @classmethod
    def fit_from_embeddings(
        cls,
        embeddings: np.ndarray,
        *,
        encoder_name: str,
        n_aspects: int = 12,
        pca_components: int = 64,
        target_assignment_ratio: float = 4.0,
        prefix: str = "judge_asp_",
        seed: int = 2026,
        kmeans_batch_size: int = 4096,
    ) -> "IndependentJudgeAspectModel":
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2 or len(x) < 3:
            raise ValueError("embeddings must have shape (n >= 3, d).")
        if not np.all(np.isfinite(x)):
            raise ValueError("embeddings contain NaN or infinity.")
        if not 2 <= n_aspects <= len(x):
            raise ValueError("n_aspects must be between 2 and the sample size.")

        n_components = min(int(pca_components), x.shape[1], len(x) - 1)
        if n_components < 1:
            raise ValueError("pca_components resolves to fewer than one component.")

        pca = PCA(n_components=n_components, random_state=seed)
        reduced = pca.fit_transform(x)

        clusterer = MiniBatchKMeans(
            n_clusters=int(n_aspects),
            batch_size=min(int(kmeans_batch_size), max(256, len(reduced))),
            n_init=10,
            random_state=seed,
            reassignment_ratio=0.01,
        )
        clusterer.fit(reduced)
        distances = clusterer.transform(reduced) ** 2
        tau = calibrate_assignment_temperature(
            distances,
            target_nearest_to_second_ratio=target_assignment_ratio,
        )
        return cls(
            encoder_name=encoder_name,
            pca=pca,
            clusterer=clusterer,
            assignment_tau=tau,
            prefix=prefix,
            fit_seed=seed,
        )

    @classmethod
    def fit_from_texts(
        cls,
        texts: Sequence[str],
        *,
        encoder_name: str,
        n_aspects: int = 12,
        pca_components: int = 64,
        target_assignment_ratio: float = 4.0,
        prefix: str = "judge_asp_",
        seed: int = 2026,
        batch_size: int = 128,
        device: Optional[str] = None,
    ) -> "IndependentJudgeAspectModel":
        embeddings = encode_texts(
            texts,
            encoder_name=encoder_name,
            batch_size=batch_size,
            normalize_embeddings=True,
            device=device,
        )
        return cls.fit_from_embeddings(
            embeddings,
            encoder_name=encoder_name,
            n_aspects=n_aspects,
            pca_components=pca_components,
            target_assignment_ratio=target_assignment_ratio,
            prefix=prefix,
            seed=seed,
        )

    def transform_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("embeddings must be two-dimensional.")
        reduced = self.pca.transform(x)
        squared_distances = self.clusterer.transform(reduced) ** 2
        return soft_assign_distances(squared_distances, self.assignment_tau)

    def transform_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 128,
        device: Optional[str] = None,
    ) -> np.ndarray:
        embeddings = encode_texts(
            texts,
            encoder_name=self.encoder_name,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            device=device,
        )
        return self.transform_embeddings(embeddings)

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pca, path / "judge_pca.joblib")
        joblib.dump(self.clusterer, path / "judge_kmeans.joblib")
        metadata = {
            "encoder_name": self.encoder_name,
            "assignment_tau": float(self.assignment_tau),
            "normalize_embeddings": bool(self.normalize_embeddings),
            "prefix": self.prefix,
            "fit_seed": int(self.fit_seed),
            "n_aspects": self.n_aspects,
            "pca_components": int(self.pca.n_components_),
            "pca_explained_variance_sum": float(
                np.sum(self.pca.explained_variance_ratio_)
            ),
        }
        (path / "judge_model_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "IndependentJudgeAspectModel":
        path = Path(directory)
        metadata = json.loads(
            (path / "judge_model_metadata.json").read_text(encoding="utf-8")
        )
        return cls(
            encoder_name=metadata["encoder_name"],
            pca=joblib.load(path / "judge_pca.joblib"),
            clusterer=joblib.load(path / "judge_kmeans.joblib"),
            assignment_tau=float(metadata["assignment_tau"]),
            normalize_embeddings=bool(metadata.get("normalize_embeddings", True)),
            prefix=metadata.get("prefix", "judge_asp_"),
            fit_seed=int(metadata.get("fit_seed", 2026)),
        )


def encode_texts(
    texts: Sequence[str],
    *,
    encoder_name: str,
    batch_size: int = 128,
    normalize_embeddings: bool = True,
    device: Optional[str] = None,
) -> np.ndarray:
    """Encode text lazily so experiment-only runs do not load the encoder."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required to build the independent judge. "
            "Install requirements_cross_embedding.txt."
        ) from exc

    model = SentenceTransformer(encoder_name, device=device)
    cleaned = ["" if text is None else str(text) for text in texts]
    return np.asarray(
        model.encode(
            cleaned,
            batch_size=int(batch_size),
            show_progress_bar=True,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )


def fit_preference_from_aspects(
    rows: pd.DataFrame,
    phi_cols: Sequence[str],
    *,
    beta: float = 8.0,
    balance_col: Optional[str] = None,
) -> np.ndarray:
    """Fit a simplex preference profile in whichever aspect space is supplied."""
    cols = list(phi_cols)
    missing = [c for c in cols if c not in rows.columns]
    if missing:
        raise ValueError(f"Missing aspect columns: {missing[:5]}")
    if rows.empty:
        raise ValueError("Cannot fit a preference from zero rows.")

    if balance_col is not None and balance_col in rows.columns:
        empirical = (
            rows.groupby(balance_col, dropna=False)[cols]
            .mean()
            .to_numpy(dtype=np.float64)
            .mean(axis=0)
        )
    else:
        empirical = rows[cols].to_numpy(dtype=np.float64).mean(axis=0)
    if not np.all(np.isfinite(empirical)):
        raise ValueError("The empirical aspect profile contains NaN or infinity.")
    return normalize_simplex(softmax(float(beta) * empirical, axis=0))


class CrossEmbeddingJudgeFeedbackProvider(FeedbackProvider):
    """Evaluate selected sentences in the independent judge aspect space.

    The provider deliberately exposes ``w_true_judge`` rather than ``w_true`` so
    the shared-space diagnostics and regret calculations in the original runner
    are not accidentally invoked.
    """

    def __init__(
        self,
        w_true_judge: Sequence[float],
        judge_phi_cols: Sequence[str],
        *,
        gamma: float = 12.0,
        noise_std: float = 0.0,
        seed: int = 42,
        threshold: Optional[float] = None,
        aggregation: str = "uniform",
    ) -> None:
        self.w_true_judge = normalize_simplex(
            np.asarray(w_true_judge, dtype=np.float64)
        )
        self.judge_phi_cols = list(judge_phi_cols)
        if len(self.judge_phi_cols) != len(self.w_true_judge):
            raise ValueError("judge_phi_cols and w_true_judge have different dimensions.")
        if aggregation not in {"uniform", "learner_alpha"}:
            raise ValueError("aggregation must be 'uniform' or 'learner_alpha'.")
        self.gamma = float(gamma)
        self.noise_std = float(noise_std)
        self.seed = int(seed)
        self.threshold = (
            float(threshold)
            if threshold is not None
            else float(1.0 / len(self.w_true_judge))
        )
        self.aggregation = aggregation
        self.records: list[dict] = []

    def _noise_for_round(self, t: Optional[int]) -> float:
        round_id = 0 if t is None else int(t)
        sequence = np.random.SeedSequence([self.seed, round_id, 314159])
        rng = np.random.default_rng(sequence)
        return float(rng.normal(0.0, self.noise_std))

    def _aggregate(self, selected_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        missing = [c for c in self.judge_phi_cols if c not in selected_df.columns]
        if missing:
            raise ValueError(
                "Selected sentences do not contain independent judge columns: "
                f"{missing[:5]}. Run build_cross_embedding_judge.py first."
            )
        phi = selected_df[self.judge_phi_cols].to_numpy(dtype=np.float64)
        if len(phi) == 0:
            raise ValueError("Cannot evaluate an empty selected set.")
        phi = np.vstack([normalize_simplex(row) for row in phi])

        if self.aggregation == "learner_alpha":
            if "alpha" not in selected_df.columns:
                raise ValueError(
                    "learner_alpha aggregation requested, but selected_df has no alpha column."
                )
            weights = normalize_simplex(selected_df["alpha"].to_numpy(dtype=np.float64))
        else:
            weights = np.ones(len(phi), dtype=np.float64) / len(phi)

        z_judge = normalize_simplex(weights @ phi)
        return z_judge, weights

    def get_feedback(
        self,
        *,
        user_id,
        product_id,
        z_t,
        selected_df,
        summary_text=None,
        t=None,
    ) -> float:
        del z_t, summary_text  # The judge never evaluates learner-space z_t.
        z_judge, weights = self._aggregate(selected_df)
        utility = float(np.dot(self.w_true_judge, z_judge))
        noise = self._noise_for_round(t)
        feedback = float(
            np.clip(sigmoid(self.gamma * (utility + noise - self.threshold)), 0.0, 1.0)
        )
        self.records.append(
            {
                "t": 0 if t is None else int(t),
                "user_id": str(user_id),
                "product_id": str(product_id),
                "judge_utility": utility,
                "judge_threshold": self.threshold,
                "judge_noise": noise,
                "judge_feedback": feedback,
                "judge_z": z_judge.copy(),
                "judge_weights": weights.copy(),
                "judge_aggregation": self.aggregation,
            }
        )
        return feedback

"""Held-out-judge utilities for PREFER.

The learner and judge use the same latent aspect coordinates, but their user
profiles are estimated from disjoint groups of the target user's historical
text.  This tests data-level feedback generalization; it does not remove the
shared aspect-space assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .feedback import FeedbackProvider
from .utils import normalize_simplex, sigmoid, softmax


@dataclass(frozen=True)
class UserHistorySplit:
    """A disjoint learner/judge split of one user's historical text."""

    fit_rows: pd.DataFrame
    holdout_rows: pd.DataFrame
    split_unit: str
    fit_group_ids: tuple[str, ...]
    holdout_group_ids: tuple[str, ...]


def _make_split_groups(
    user_rows: pd.DataFrame,
    *,
    group_col: Optional[str],
    product_col: str,
    allow_row_split: bool,
) -> tuple[pd.Series, str]:
    """Return a stable group label per row and a description of the split unit."""

    if group_col is not None:
        if group_col not in user_rows.columns:
            raise ValueError(f"Requested group column {group_col!r} is absent.")
        labels = user_rows[group_col].fillna("__MISSING__").astype(str)
        if labels.nunique() >= 2:
            return labels, group_col
        raise ValueError(
            f"Column {group_col!r} has fewer than two groups for this user."
        )

    # Prefer a true review identifier when one exists.
    for candidate in ("review_index", "review_id", "review_idx"):
        if candidate in user_rows.columns:
            labels = user_rows[candidate].fillna("__MISSING__").astype(str)
            if labels.nunique() >= 2:
                return labels, candidate

    # A product-time pair is a useful proxy for a review identifier.
    timestamp_col = next(
        (c for c in ("timestamp", "time", "review_timestamp") if c in user_rows.columns),
        None,
    )
    if product_col in user_rows.columns and timestamp_col is not None:
        labels = (
            user_rows[product_col].fillna("__MISSING_PRODUCT__").astype(str)
            + "::"
            + user_rows[timestamp_col].fillna("__MISSING_TIME__").astype(str)
        )
        if labels.nunique() >= 2:
            return labels, f"{product_col}+{timestamp_col}"

    # Product-level splitting is conservative: all text about a product stays
    # on one side of the split.
    if product_col in user_rows.columns:
        labels = user_rows[product_col].fillna("__MISSING_PRODUCT__").astype(str)
        if labels.nunique() >= 2:
            return labels, product_col

    if not allow_row_split:
        raise ValueError(
            "Could not find at least two review/product groups for the user. "
            "Provide --group-col or explicitly enable row-level splitting."
        )

    labels = pd.Series(
        [f"row::{i}" for i in range(len(user_rows))],
        index=user_rows.index,
        dtype=str,
    )
    return labels, "row"


def split_user_history(
    df_sent: pd.DataFrame,
    user_id: str,
    *,
    holdout_fraction: float = 0.5,
    seed: int = 0,
    user_col: str = "user_id",
    product_col: str = "parent_asin",
    group_col: Optional[str] = None,
    allow_row_split: bool = False,
) -> UserHistorySplit:
    """Split one user's history at the review/product-group level.

    The split is performed over groups, not sentences, so sentences from the
    same review cannot appear in both learner and judge partitions.
    """

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be strictly between 0 and 1.")
    if user_col not in df_sent.columns:
        raise ValueError(f"User column {user_col!r} is absent.")

    user_rows = df_sent[df_sent[user_col].astype(str) == str(user_id)].copy()
    if user_rows.empty:
        raise ValueError(f"User {user_id!r} does not occur in the dataset.")

    labels, split_unit = _make_split_groups(
        user_rows,
        group_col=group_col,
        product_col=product_col,
        allow_row_split=allow_row_split,
    )
    user_rows["__heldout_group__"] = labels

    groups = user_rows["__heldout_group__"].unique().astype(str)
    if len(groups) < 2:
        raise ValueError("At least two groups are required for a held-out split.")

    rng = np.random.default_rng(seed)
    groups = rng.permutation(groups)
    n_holdout = int(round(holdout_fraction * len(groups)))
    n_holdout = min(max(n_holdout, 1), len(groups) - 1)

    holdout_group_ids = tuple(sorted(groups[:n_holdout].tolist()))
    fit_group_ids = tuple(sorted(groups[n_holdout:].tolist()))
    holdout_set = set(holdout_group_ids)

    holdout_rows = user_rows[user_rows["__heldout_group__"].isin(holdout_set)].copy()
    fit_rows = user_rows[~user_rows["__heldout_group__"].isin(holdout_set)].copy()

    if fit_rows.empty or holdout_rows.empty:
        raise RuntimeError("The held-out split unexpectedly produced an empty side.")

    fit_rows.drop(columns="__heldout_group__", inplace=True)
    holdout_rows.drop(columns="__heldout_group__", inplace=True)

    return UserHistorySplit(
        fit_rows=fit_rows,
        holdout_rows=holdout_rows,
        split_unit=split_unit,
        fit_group_ids=fit_group_ids,
        holdout_group_ids=holdout_group_ids,
    )


def fit_aspect_preference(
    rows: pd.DataFrame,
    phi_cols: Sequence[str],
    *,
    beta: float = 8.0,
    balance_col: Optional[str] = None,
) -> np.ndarray:
    """Estimate a simplex-valued preference from aspect-score rows.

    If ``balance_col`` is supplied, first average within each group and then
    average across groups.  This prevents a long review from receiving more
    weight merely because it contains more sentences.
    """

    phi_cols = list(phi_cols)
    missing = [c for c in phi_cols if c not in rows.columns]
    if missing:
        raise ValueError(f"Missing aspect columns: {missing[:5]}")
    if rows.empty:
        raise ValueError("Cannot estimate a preference from zero rows.")

    if balance_col is not None and balance_col in rows.columns:
        group_profiles = rows.groupby(balance_col, dropna=False)[phi_cols].mean()
        empirical_profile = group_profiles.to_numpy(dtype=np.float64).mean(axis=0)
    else:
        empirical_profile = rows[phi_cols].to_numpy(dtype=np.float64).mean(axis=0)

    if not np.all(np.isfinite(empirical_profile)):
        raise ValueError("The empirical aspect profile contains NaN or infinity.")

    return normalize_simplex(softmax(float(beta) * empirical_profile, axis=0))


def cosine(a: Sequence[float], b: Sequence[float], eps: float = 1e-12) -> float:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    return float(
        np.dot(a_arr, b_arr)
        / (max(np.linalg.norm(a_arr), eps) * max(np.linalg.norm(b_arr), eps))
    )


def jensen_shannon_divergence(
    p: Sequence[float], q: Sequence[float], eps: float = 1e-12
) -> float:
    p_arr = normalize_simplex(np.asarray(p, dtype=np.float64))
    q_arr = normalize_simplex(np.asarray(q, dtype=np.float64))
    m_arr = 0.5 * (p_arr + q_arr)
    kl_pm = np.sum(p_arr * (np.log(p_arr + eps) - np.log(m_arr + eps)))
    kl_qm = np.sum(q_arr * (np.log(q_arr + eps) - np.log(m_arr + eps)))
    return float(0.5 * (kl_pm + kl_qm))


class HeldOutJudgeFeedbackProvider(FeedbackProvider):
    """Generate scalar feedback from a hidden held-out preference vector.

    For round ``t``:

        q_t = w_judge^T z_t + epsilon_t
        f_t = sigmoid(gamma * (q_t - threshold))

    The Gaussian noise is generated deterministically from ``(seed, t)``.  Two
    policies evaluated with the same seed therefore receive common random
    numbers at the same round, making paired comparisons less noisy.
    """

    def __init__(
        self,
        w_true_judge: Sequence[float],
        *,
        gamma: float = 12.0,
        noise_std: float = 0.0,
        seed: int = 42,
        threshold: Optional[float] = None,
    ) -> None:
        w = normalize_simplex(np.asarray(w_true_judge, dtype=np.float64))
        if w.ndim != 1 or len(w) == 0:
            raise ValueError("w_true_judge must be a nonempty one-dimensional vector.")
        if not np.all(np.isfinite(w)):
            raise ValueError("w_true_judge contains NaN or infinity.")
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        if noise_std < 0:
            raise ValueError("noise_std must be nonnegative.")

        self.w_true = w.astype(np.float32)
        self.gamma = float(gamma)
        self.noise_std = float(noise_std)
        self.seed = int(seed)
        self.threshold = float(1.0 / len(w) if threshold is None else threshold)
        self._fallback_rng = np.random.default_rng(seed)

    def get_true_preference(self, t: Optional[int] = None) -> np.ndarray:
        return self.w_true.copy()

    def utility(self, z_t: Sequence[float]) -> float:
        z = np.asarray(z_t, dtype=np.float64)
        if z.shape != self.w_true.shape:
            raise ValueError(
                f"z_t has shape {z.shape}, but judge has shape {self.w_true.shape}."
            )
        return float(np.dot(self.w_true, z))

    def _noise_for_round(self, t: Optional[int]) -> float:
        if self.noise_std == 0.0:
            return 0.0
        if t is None:
            return float(self._fallback_rng.normal(0.0, self.noise_std))
        round_seed = np.random.SeedSequence([self.seed, int(t), 104729])
        rng = np.random.default_rng(round_seed)
        return float(rng.normal(0.0, self.noise_std))

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
        score = self.utility(z_t)
        noisy_score = score + self._noise_for_round(t)
        feedback = sigmoid(self.gamma * (noisy_score - self.threshold))
        return float(np.clip(feedback, 0.0, 1.0))

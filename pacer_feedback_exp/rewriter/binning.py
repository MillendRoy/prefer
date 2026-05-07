from __future__ import annotations

from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd



def bin_by_relevance_with_stats(
    df: pd.DataFrame,
    score_col: str = "utility",
    high_q: float = 0.67,
    low_q: float = 0.33,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, dict], Dict[str, float | None]]:
    if df is None or len(df) == 0:
        empty = df.iloc[0:0].copy()
        bins = {"high": empty, "mid": empty, "low": empty}
        stats = {
            "high": {"count": 0, "pct": 0.0, "mean_score": 0.0},
            "mid": {"count": 0, "pct": 0.0, "mean_score": 0.0},
            "low": {"count": 0, "pct": 0.0, "mean_score": 0.0},
        }
        thresholds = {"low_thr": None, "high_thr": None}
        return bins, stats, thresholds

    work = df.copy()
    scores = work[score_col].astype(float)
    low_thr = float(scores.quantile(low_q))
    high_thr = float(scores.quantile(high_q))

    low_bin = work[scores <= low_thr].copy()
    high_bin = work[scores >= high_thr].copy()
    mid_bin = work[(scores > low_thr) & (scores < high_thr)].copy()

    bins = {"high": high_bin, "mid": mid_bin, "low": low_bin}
    total = len(work)

    def _make_stats(subdf: pd.DataFrame) -> dict:
        if len(subdf) == 0:
            return {"count": 0, "pct": 0.0, "mean_score": 0.0}
        return {
            "count": int(len(subdf)),
            "pct": 100.0 * len(subdf) / total,
            "mean_score": float(subdf[score_col].mean()),
        }

    stats = {name: _make_stats(subdf) for name, subdf in bins.items()}
    thresholds = {"low_thr": low_thr, "high_thr": high_thr}
    return bins, stats, thresholds

# import numpy as np
# import pandas as pd


def bin_by_support_with_stats(
    df: pd.DataFrame,
    phi_cols: Optional[List[str]] = None,
    theme_col: Optional[str] = None,
    reviewer_col: Optional[str] = None,
    high_q: float = 0.67,
    low_q: float = 0.33,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, dict], Dict[str, float | None]]:
    """
    Bin already-selected relevant sentences by how commonly their theme is discussed.

    Interpretation:
        high: sentences from themes mentioned by many people/sentences
        mid : sentences from themes mentioned by some people/sentences
        low : sentences from themes mentioned by a few people/sentences

    Parameters
    ----------
    df:
        Selected relevant evidence dataframe.
    phi_cols:
        Aspect-membership columns, e.g. ["asp_0", ..., "asp_9"].
        If theme_col is not provided, the dominant aspect argmax_k phi_{i,k}
        is used as the theme.
    theme_col:
        Optional existing column containing a theme/aspect/cluster label.
    reviewer_col:
        Optional column identifying reviewer/user/review source.
        If provided, support is computed by number of unique reviewers per theme.
        If None, support is computed by number of sentences per theme.
    high_q, low_q:
        Quantiles used to split theme support into high/mid/low bins.

    Returns
    -------
    bins:
        Dict with keys {"high", "mid", "low"}.
    stats:
        Summary statistics for each bin.
    thresholds:
        Low and high support thresholds.
    """

    if df is None or len(df) == 0:
        empty = df.iloc[0:0].copy() if df is not None else pd.DataFrame()
        bins = {"high": empty, "mid": empty, "low": empty}
        stats = {
            "high": {"count": 0, "pct": 0.0, "mean_support": 0.0, "num_themes": 0},
            "mid": {"count": 0, "pct": 0.0, "mean_support": 0.0, "num_themes": 0},
            "low": {"count": 0, "pct": 0.0, "mean_support": 0.0, "num_themes": 0},
        }
        thresholds = {"low_thr": None, "high_thr": None}
        return bins, stats, thresholds

    work = df.copy()

    # ---------------------------------------------------------
    # 1. Determine theme/aspect label for each selected sentence
    # ---------------------------------------------------------
    # If theme_col is provided, use it directly. 
    # Otherwise infer from phi_cols by argmax.
    if theme_col is not None and theme_col in work.columns:
        work["_theme"] = work[theme_col].astype(str)

    else:
        if phi_cols is None:
            phi_cols = [c for c in work.columns if c.startswith("asp_")]

        if len(phi_cols) == 0:
            raise ValueError(
                "Either provide theme_col or provide phi_cols / aspect columns "
                "so that dominant themes can be inferred."
            )

        phi_mat = work[phi_cols].astype(float).to_numpy()
        dom_idx = np.argmax(phi_mat, axis=1)
        work["_theme"] = [phi_cols[j] for j in dom_idx]

    # ---------------------------------------------------------
    # 2. Compute how commonly each theme appears
    # ---------------------------------------------------------
    # Support can be defined in two ways:
    # 1) by number of unique reviewers mentioning the theme, or
    # 2) by number of sentences mentioning the theme.
    if reviewer_col is not None and reviewer_col in work.columns:
        # Support = number of unique reviewers/reviews mentioning the theme.
        theme_support = (
            work.groupby("_theme")[reviewer_col]
            .nunique()
            .rename("_theme_support")
        )
    else:
        # Support = number of selected sentences mentioning the theme.
        theme_support = (
            work.groupby("_theme")
            .size()
            .rename("_theme_support")
        )

    work = work.merge(
        theme_support.reset_index(),
        on="_theme",
        how="left",
    )

    total_support = float(theme_support.sum())
    if total_support > 0:
        work["_theme_support_pct"] = 100.0 * work["_theme_support"] / total_support
    else:
        work["_theme_support_pct"] = 0.0

    # ---------------------------------------------------------
    # 3. Bin by support quantiles
    # ---------------------------------------------------------
    supports = work["_theme_support"].astype(float)

    low_thr = float(supports.quantile(low_q))
    high_thr = float(supports.quantile(high_q))

    low_bin = work[supports <= low_thr].copy()
    high_bin = work[supports >= high_thr].copy()
    mid_bin = work[(supports > low_thr) & (supports < high_thr)].copy()

    # Sort within bins by support first, then utility if available.
    sort_cols = ["_theme_support"]
    ascending = [False]

    if "utility" in work.columns:
        sort_cols.append("utility")
        ascending.append(False)

    bins = {
        "high": high_bin.sort_values(sort_cols, ascending=ascending),
        "mid": mid_bin.sort_values(sort_cols, ascending=ascending),
        "low": low_bin.sort_values(sort_cols, ascending=ascending),
    }

    total = len(work)

    def _make_stats(subdf: pd.DataFrame) -> dict:
        if len(subdf) == 0:
            return {
                "count": 0,
                "pct": 0.0,
                "mean_score": 0.0,
                "num_themes": 0,
            }

        return {
            "count": int(len(subdf)),
            "pct": 100.0 * len(subdf) / total,
            "mean_score": float(subdf.get("utility", []).mean()),
            "num_themes": int(subdf["_theme"].nunique()),
        }

    stats = {name: _make_stats(subdf) for name, subdf in bins.items()}

    thresholds = {
        "low_thr": low_thr,
        "high_thr": high_thr,
    }

    return bins, stats, thresholds
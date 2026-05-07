from __future__ import annotations

from typing import Callable, Dict, Any

import pandas as pd

from .binning import bin_by_relevance_with_stats, bin_by_support_with_stats
from .models import dedup_sentences, fallback_summarizer
from .prompts import basic_prompt, make_instruct_stitch_prompt, make_final_third_person_prompt



def contextual_rewrite_baseline(
    selected_df: pd.DataFrame,
    text_col: str = "review_text",
    score_col: str = "utility",
    high_q: float = 0.67,
    low_q: float = 0.33,
    max_high: int = 8,
    max_mid: int = 8,
    max_low: int = 6,
    use_low: bool = True,
    summarizer: Callable[[str], str] | None = None,
) -> Dict[str, Any]:
    if summarizer is None:
        summarizer = fallback_summarizer

    if selected_df is None or len(selected_df) == 0:
        return {
            "S_H": "",
            "S_M": "",
            "S_L": "",
            "final": "",
            "bin_stats": {},
            "thresholds": {},
            "bins": {"high": None, "mid": None, "low": None},
            "prompts": {"high": "", "mid": "", "low": "", "final": ""},
        }

    df_sorted = selected_df.sort_values(score_col, ascending=False).copy()
    phi_cols = [c for c in selected_df.columns if c.startswith("asp_")]

    bins, stats, thresholds = bin_by_support_with_stats(
        selected_df,
        phi_cols=phi_cols,
        reviewer_col="user_id",   # or "review_id" if you have it
        high_q=high_q,
        low_q=low_q,
    )
    # bins, stats, thresholds = bin_by_relevance_with_stats(df_sorted, score_col=score_col, high_q=high_q, low_q=low_q)

    high_sents = dedup_sentences(bins["high"][text_col].tolist())[:max_high]
    mid_sents = dedup_sentences(bins["mid"][text_col].tolist())[:max_mid]
    low_sents = dedup_sentences(bins["low"][text_col].tolist())[:max_low]

    pH = basic_prompt(high_sents)
    pM = basic_prompt(mid_sents)
    pL = basic_prompt(low_sents)

    S_H = summarizer(pH) if high_sents else ""
    S_M = summarizer(pM) if mid_sents else ""
    S_L = summarizer(pL) if (use_low and low_sents) else ""

    final_prompt_seed = basic_prompt([x for x in [S_H, S_M, S_L] if x])
    final_seed = summarizer(final_prompt_seed) if final_prompt_seed else ""

    return {
        "S_H": S_H,
        "S_M": S_M,
        "S_L": S_L,
        "final": final_seed,
        "bin_stats": stats,
        "thresholds": thresholds,
        "bins": bins,
        "prompts": {"high": pH, "mid": pM, "low": pL, "final": final_prompt_seed},
    }



def rewrite_selected_df(
    selected_df: pd.DataFrame,
    *,
    text_col: str = "review_text",
    score_col: str = "utility",
    bin_summarizer: Callable[[str], str] | None = None,
    stitch_rewriter: Callable[[str], str] | None = None,
    final_rewriter: Callable[[str], str] | None = None,
    high_q: float = 0.67,
    low_q: float = 0.33,
    use_low: bool = True,
) -> Dict[str, Any]:
    if bin_summarizer is None:
        bin_summarizer = fallback_summarizer
    if stitch_rewriter is None:
        stitch_rewriter = fallback_summarizer
    if final_rewriter is None:
        final_rewriter = fallback_summarizer

    stage1 = contextual_rewrite_baseline(
        selected_df,
        text_col=text_col,
        score_col=score_col,
        high_q=high_q,
        low_q=low_q,
        use_low=use_low,
        summarizer=bin_summarizer,
    )

    bin_payloads = {
        "high": {"summary": stage1["S_H"], "stats": stage1["bin_stats"]["high"]},
        "mid": {"summary": stage1["S_M"], "stats": stage1["bin_stats"]["mid"]},
        "low": {"summary": stage1["S_L"], "stats": stage1["bin_stats"]["low"]},
    }

    stitch_prompt = make_instruct_stitch_prompt(bin_payloads, score_col=score_col)
    stitched_summary = stitch_rewriter(stitch_prompt)

    final_prompt = make_final_third_person_prompt(stitched_summary)
    final_summary = final_rewriter(final_prompt)

    return {
        "stage1": stage1,
        "bin_payloads": bin_payloads,
        "stitch_prompt": stitch_prompt,
        "stitched_summary": stitched_summary,
        "final_prompt": final_prompt,
        "final_summary": final_summary,
    }

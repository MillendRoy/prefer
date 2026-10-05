#!/usr/bin/env python3
"""Export PREFER summaries with the exact evidence used for rewriting.

The final summary must be evaluated against the selected HIGH/MID/LOW review
sentences, not against the complete product corpus and not against the fallback
summary alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def _text_list(frame: pd.DataFrame | None, text_col: str) -> list[str]:
    if frame is None or len(frame) == 0 or text_col not in frame.columns:
        return []
    values = (
        frame[text_col]
        .dropna()
        .astype(str)
        .map(lambda text: " ".join(text.split()))
        .tolist()
    )
    return [text for text in values if text]


def evidence_from_rewrite(
    selected_df: pd.DataFrame,
    rewrite_out: Mapping[str, Any] | None,
    *,
    text_col: str = "review_text",
) -> dict[str, list[str]]:
    rewrite_out = dict(rewrite_out or {})
    stage1 = dict(rewrite_out.get("stage1") or {})
    bins = dict(stage1.get("bins") or {})

    high = _text_list(bins.get("high"), text_col)
    mid = _text_list(bins.get("mid"), text_col)
    low = _text_list(bins.get("low"), text_col)
    selected = _text_list(selected_df, text_col)
    combined = high + mid + low
    if not combined:
        combined = selected

    return {
        "evidence": combined,
        "evidence_high": high,
        "evidence_mid": mid,
        "evidence_low": low,
        "selected_evidence": selected,
    }


def make_extractive_summary(evidence: Mapping[str, list[str]]) -> str:
    return (
        "HIGH: " + " ".join(evidence.get("evidence_high", []))
        + " MID: " + " ".join(evidence.get("evidence_mid", []))
        + " LOW: " + " ".join(evidence.get("evidence_low", []))
    ).strip()


def build_quality_record(
    *,
    example_id: str,
    method: str,
    summary: str,
    selected_df: pd.DataFrame,
    rewrite_out: Mapping[str, Any] | None,
    text_col: str = "review_text",
    round_id: int | None = None,
    unit_id: str | None = None,
    profile_name: str | None = None,
    target_profile: str | None = None,
    product_id: str | None = None,
    seed: int | None = None,
    reference: str | None = None,
) -> dict[str, Any]:
    """Create one serializable quality-evaluation row."""
    evidence = evidence_from_rewrite(
        selected_df, rewrite_out, text_col=text_col
    )
    rewrite_out = dict(rewrite_out or {})
    return {
        "id": str(example_id),
        "method": str(method),
        "round": round_id,
        "unit_id": unit_id,
        "profile_name": profile_name,
        "target_profile": target_profile,
        "product_id": product_id,
        "seed": seed,
        "evidence": json.dumps(evidence["evidence"], ensure_ascii=False),
        "evidence_high": json.dumps(
            evidence["evidence_high"], ensure_ascii=False
        ),
        "evidence_mid": json.dumps(evidence["evidence_mid"], ensure_ascii=False),
        "evidence_low": json.dumps(evidence["evidence_low"], ensure_ascii=False),
        "selected_evidence": json.dumps(
            evidence["selected_evidence"], ensure_ascii=False
        ),
        "summary": str(summary).strip(),
        "fallback_summary": str(
            (rewrite_out.get("stage1") or {}).get("final") or ""
        ).strip(),
        "stitched_summary": str(
            rewrite_out.get("stitched_summary") or ""
        ).strip(),
        "reference": "" if reference is None else str(reference).strip(),
    }


def quality_log_fields(
    *,
    selected_df: pd.DataFrame,
    summary_payload: Mapping[str, Any] | None,
    summary_text: str | None,
    user_id: str,
    seed: int,
    round_id: int,
    product_id: str,
    method: str,
    text_col: str = "review_text",
    target_profile: str | None = None,
) -> dict[str, Any]:
    """Fields to add to one ``run_online_experiment`` log dictionary."""
    record = build_quality_record(
        example_id=f"{user_id}:{seed}:{round_id}:{product_id}:{method}",
        method=method,
        summary=summary_text or "",
        selected_df=selected_df,
        rewrite_out=summary_payload,
        text_col=text_col,
        round_id=round_id,
        unit_id=f"{user_id}:{seed}",
        profile_name=user_id,
        target_profile=target_profile,
        product_id=product_id,
        seed=seed,
    )
    # The log already has summary_text; expose the evaluator's canonical name too.
    record["summary_text"] = record["summary"]
    record["user_id"] = user_id
    return record


def append_quality_record(record: Mapping[str, Any], output_csv: str | Path) -> None:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([dict(record)])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)

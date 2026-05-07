from __future__ import annotations

from typing import Iterable, List


def dedup_sentences(sentences: Iterable[str], min_chars: int = 10) -> List[str]:
    """
    Deduplicate sentences while preserving order. Sentences that are shorter than min_chars after stripping are filtered out.
    """
    out: List[str] = []
    seen: set[str] = set()
    for s in sentences:
        s = "" if s is None else str(s).strip()
        s = " ".join(s.split())
        if len(s) < min_chars:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def fallback_summarizer(prompt_or_sentences, max_sentences: int = 4) -> str:
    """
    Lightweight deterministic fallback.
    Useful when you want the rewriter pipeline wired up before attaching an LLM.
    """
    if isinstance(prompt_or_sentences, str):
        text = prompt_or_sentences.strip()
        if not text:
            return ""
        sents = [x.strip() for x in text.split(".") if x.strip()]
        sents = [f"{x}." if not x.endswith((".", "!", "?")) else x for x in sents]
    else:
        sents = list(prompt_or_sentences)

    sents = dedup_sentences(sents)
    return " ".join(sents[:max_sentences])

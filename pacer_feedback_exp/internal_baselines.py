"""Internal-oracle baselines for PREFER.

These baselines deliberately use PREFER's original latent aspect representation and
SyntheticFeedbackProvider.  No external LLM judge is used.

The long-context LLM is used only as an evidence selector. It returns sentence IDs,
not a generated summary. The original SyntheticFeedbackProvider scores the aspect
profile of those selected evidence sentences exactly as in the original experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from pacer_feedback_exp.preference import aspect_profile
from pacer_feedback_exp.schemas import ExtractionResult
from pacer_feedback_exp.utils import ensure_global_index, get_phi_matrix, normalize_simplex


_EPS = 1e-12


def clean_text(value: Any) -> str:
    return " ".join(str(value).split()).strip()


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    temperature = max(float(temperature), 1e-6)
    x = x / temperature
    x = x - np.max(x)
    values = np.exp(np.clip(x, -50.0, 50.0))
    return normalize_simplex(values)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= _EPS:
        return float("nan")
    return float(np.dot(a, b) / denominator)


def infer_text_column(df: pd.DataFrame, requested: Optional[str] = None) -> str:
    if requested and requested in df.columns:
        return requested
    for column in ("review_text", "sentence", "text"):
        if column in df.columns:
            return column
    raise ValueError("Could not find a review-text column. Pass --text-col explicitly.")


def build_user_profile_text(
    df_sent: pd.DataFrame,
    *,
    user_id: str,
    text_col: str,
    max_sentences: int = 40,
    max_chars: int = 12000,
) -> str:
    """Build the LLM/semantic profile from the user's existing review sentences.

    This intentionally follows the same circular setup as make_synthetic_true_preference:
    the user's own review text is the source of the hidden synthetic preference and the
    textual profile supplied to the strong baselines.
    """
    rows = df_sent.loc[df_sent["user_id"].astype(str) == str(user_id), text_col]
    texts: list[str] = []
    used = 0
    for value in rows.tolist():
        text = clean_text(value)
        if not text or text in texts:
            continue
        if len(texts) >= int(max_sentences):
            break
        if used + len(text) > int(max_chars):
            remaining = int(max_chars) - used
            if remaining > 40:
                texts.append(text[:remaining].rstrip())
            break
        texts.append(text)
        used += len(text) + 1
    if not texts:
        return "The user has no available review history; select broad, balanced product evidence."
    return "\n".join(f"- {text}" for text in texts)


def build_item_profile(candidates: pd.DataFrame, product_id: str) -> str:
    lines = [f"product_id: {product_id}"]
    metadata_columns = (
        "product_title",
        "title",
        "name",
        "brand",
        "category",
        "store",
        "average_rating",
        "rating_number",
        "price",
    )
    for column in metadata_columns:
        if column not in candidates.columns:
            continue
        values = [clean_text(value) for value in candidates[column].dropna().tolist()]
        value = next((item for item in values if item), "")
        if value:
            lines.append(f"{column}: {value}")
    if "rating" in candidates.columns:
        rating = pd.to_numeric(candidates["rating"], errors="coerce").dropna()
        if len(rating):
            lines.append(f"mean_review_rating: {float(rating.mean()):.3f}")
            lines.append(f"review_rating_count: {int(len(rating))}")
    lines.append(f"available_review_sentences: {int(len(candidates))}")
    return "\n".join(lines)


def _lengths(frame: pd.DataFrame, text_col: str, length_col: str) -> np.ndarray:
    if length_col in frame.columns:
        values = pd.to_numeric(frame[length_col], errors="coerce").fillna(0).astype(int)
        return values.to_numpy()
    return frame[text_col].fillna("").astype(str).str.split().str.len().to_numpy(dtype=int)


def _selected_result(
    *,
    method: str,
    candidates: pd.DataFrame,
    selected_local_idx: Sequence[int],
    phi_cols: Sequence[str],
    phi_rows: np.ndarray,
    user_id: str,
    product_id: str,
    alpha: Optional[np.ndarray],
    text_col: str,
    length_col: str,
    scores: Optional[Mapping[str, np.ndarray]] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> ExtractionResult:
    selected_local_idx = np.asarray(selected_local_idx, dtype=int)
    selected_df = candidates.iloc[selected_local_idx].copy()
    selected_global_idx = selected_df["global_idx"].to_numpy(dtype=int)
    if alpha is None:
        alpha = np.ones(len(selected_df), dtype=np.float32) / max(1, len(selected_df))
    alpha = normalize_simplex(np.asarray(alpha, dtype=float)).astype(np.float32)
    selected_df["alpha"] = alpha
    z_t = aspect_profile(phi_rows, selected_global_idx, alpha=alpha)
    total_len = int(_lengths(selected_df, text_col, length_col).sum())
    return ExtractionResult(
        method=method,
        user_id=user_id,
        product_id=product_id,
        selected_df=selected_df,
        selected_local_idx=selected_local_idx,
        selected_global_idx=selected_global_idx,
        alpha=alpha,
        z_t=np.asarray(z_t, dtype=np.float32),
        total_len=total_len,
        n_candidates=int(len(candidates)),
        scores=dict(scores or {}),
        meta=dict(meta or {}),
    )


class SentenceEmbeddingModel:
    """Thin wrapper around the sentence encoder already used by PREFER."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers to run the semantic baseline."
            ) from exc
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


@dataclass
class SemanticSimilarityPolicy:
    embedder: SentenceEmbeddingModel
    profile_text: str
    k: int = 8
    length_budget: int = 800
    redundancy_penalty: float = 0.20
    update_rate: float = 0.25
    baseline_mode: str = "running_mean"
    ema_alpha: float = 0.10
    text_col: str = "review_text"
    length_col: str = "len"
    alpha_temperature: float = 0.20

    def __post_init__(self) -> None:
        self.profile_embedding = self.embedder.encode([self.profile_text])[0]
        self.feedback_sum = 0.0
        self.feedback_count = 0
        self.baseline = 0.0

    def _baseline_before_update(self) -> float:
        if self.feedback_count == 0:
            return 0.0
        return float(self.baseline)

    def select(
        self,
        *,
        df_sent: pd.DataFrame,
        user_id: str,
        product_id: str,
        phi_cols: Sequence[str],
        phi_rows: np.ndarray,
    ) -> Optional[ExtractionResult]:
        df_sent = ensure_global_index(df_sent)
        candidates = df_sent.loc[df_sent["parent_asin"].astype(str) == str(product_id)].copy()
        if len(candidates) == 0:
            return None
        texts = [clean_text(value) for value in candidates[self.text_col].fillna("").tolist()]
        embeddings = self.embedder.encode(texts)
        relevance = embeddings @ self.profile_embedding
        lengths = _lengths(candidates, self.text_col, self.length_col)

        selected: list[int] = []
        used_len = 0
        greedy_scores: list[float] = []
        for _ in range(int(self.k)):
            best_index = None
            best_score = -np.inf
            for index in range(len(candidates)):
                if index in selected:
                    continue
                if used_len + int(lengths[index]) > int(self.length_budget):
                    continue
                redundancy = 0.0
                if selected:
                    redundancy = float(np.max(embeddings[selected] @ embeddings[index]))
                score = float(relevance[index] - self.redundancy_penalty * redundancy)
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index is None:
                break
            selected.append(best_index)
            greedy_scores.append(best_score)
            used_len += int(lengths[best_index])

        if not selected:
            return None
        selected_relevance = relevance[np.asarray(selected, dtype=int)]
        alpha = softmax(selected_relevance, temperature=self.alpha_temperature)
        return _selected_result(
            method="SemanticSimilarity",
            candidates=candidates,
            selected_local_idx=selected,
            phi_cols=phi_cols,
            phi_rows=phi_rows,
            user_id=user_id,
            product_id=product_id,
            alpha=alpha,
            text_col=self.text_col,
            length_col=self.length_col,
            scores={"semantic_similarity": selected_relevance},
            meta={
                "selected_embeddings": embeddings[np.asarray(selected, dtype=int)],
                "greedy_scores": np.asarray(greedy_scores, dtype=float),
            },
        )

    def update(self, extraction: ExtractionResult, feedback: float) -> dict[str, float]:
        baseline_t = self._baseline_before_update()
        centered = float(feedback) - baseline_t
        selected_embeddings = np.asarray(extraction.meta["selected_embeddings"], dtype=float)
        evidence_embedding = selected_embeddings.mean(axis=0)
        candidate = self.profile_embedding + self.update_rate * centered * evidence_embedding
        norm = float(np.linalg.norm(candidate))
        if norm > _EPS:
            self.profile_embedding = candidate / norm

        self.feedback_count += 1
        self.feedback_sum += float(feedback)
        if self.baseline_mode == "ema":
            if self.feedback_count == 1:
                self.baseline = float(feedback)
            else:
                self.baseline = (
                    (1.0 - self.ema_alpha) * self.baseline
                    + self.ema_alpha * float(feedback)
                )
        else:
            self.baseline = self.feedback_sum / self.feedback_count
        return {"baseline_t": baseline_t, "centered_feedback": centered}


@dataclass
class LinearThompsonSamplingPolicy:
    n_aspects: int
    prior_precision: float = 1.0
    observation_variance: float = 0.05
    exploration_scale: float = 1.0
    softmax_temperature: float = 0.20
    baseline_mode: str = "running_mean"
    ema_alpha: float = 0.10
    seed: int = 0

    def __post_init__(self) -> None:
        self.A = float(self.prior_precision) * np.eye(self.n_aspects, dtype=float)
        self.b = np.zeros(self.n_aspects, dtype=float)
        self.rng = np.random.default_rng(self.seed)
        self.feedback_sum = 0.0
        self.feedback_count = 0
        self.baseline = 0.0

    def _baseline_before_update(self) -> float:
        if self.feedback_count == 0:
            return 0.0
        return float(self.baseline)

    def posterior_mean(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    def sampled_preference(self) -> np.ndarray:
        covariance = np.linalg.inv(self.A)
        covariance = 0.5 * (covariance + covariance.T)
        jitter = 1e-9 * np.eye(self.n_aspects)
        sample = self.rng.multivariate_normal(
            self.posterior_mean(),
            (self.exploration_scale ** 2) * covariance + jitter,
        )
        return softmax(sample, temperature=self.softmax_temperature)

    def mean_preference(self) -> np.ndarray:
        return softmax(self.posterior_mean(), temperature=self.softmax_temperature)

    def update(self, z_t: np.ndarray, feedback: float) -> dict[str, float]:
        z_t = np.asarray(z_t, dtype=float)
        baseline_t = self._baseline_before_update()
        centered = float(feedback) - baseline_t
        variance = max(float(self.observation_variance), 1e-8)
        self.A += np.outer(z_t, z_t) / variance
        self.b += z_t * centered / variance

        self.feedback_count += 1
        self.feedback_sum += float(feedback)
        if self.baseline_mode == "ema":
            if self.feedback_count == 1:
                self.baseline = float(feedback)
            else:
                self.baseline = (
                    (1.0 - self.ema_alpha) * self.baseline
                    + self.ema_alpha * float(feedback)
                )
        else:
            self.baseline = self.feedback_sum / self.feedback_count
        return {"baseline_t": baseline_t, "centered_feedback": centered}


def make_long_context_prompt(
    *,
    user_profile: str,
    item_profile: str,
    candidates: pd.DataFrame,
    text_col: str,
    k: int,
    length_budget: int,
    max_prompt_chars: int,
    interaction_history: Optional[Sequence[Mapping[str, Any]]] = None,
    history_limit: int = 5,
) -> tuple[str, dict[str, int]]:
    """Build the evidence-selection prompt.

    Static baseline: ``interaction_history=None``.
    Online baseline: provide previous selected evidence and scalar feedback. The
    hidden synthetic preference and aspect vectors are never exposed to the LLM.
    """
    header = f"""You are a personalized review-evidence selector.

USER PROFILE FROM THE USER'S EXISTING REVIEW HISTORY:
{user_profile}

CURRENT ITEM PROFILE:
{item_profile}

Select at most {int(k)} review sentences whose total word count is at most {int(length_budget)}.
Choose evidence that is most relevant to this user's preferences and avoid redundancy.
Return strict JSON only in this exact schema:
{{"selected_ids": ["S0", "S4", "S7"]}}
The selected_ids must refer only to the numbered sentences below.
Do not generate a summary, explanation, score, or any text outside the JSON object.
"""

    if interaction_history:
        history_lines: list[str] = []
        recent = list(interaction_history)[-max(1, int(history_limit)):]
        for event in recent:
            evidence = event.get("selected_evidence", [])
            evidence_text = " | ".join(clean_text(value) for value in evidence if clean_text(value))
            if len(evidence_text) > 1200:
                evidence_text = evidence_text[:1200].rstrip() + "..."
            history_lines.append(
                f"- product_id={event.get('product_id', '')}; "
                f"feedback={float(event['feedback']):.4f}; "
                f"selected_evidence={evidence_text or '[not available]'}"
            )
        header += (
            "\nPREVIOUS INTERACTIONS:\n"
            "Higher feedback means the previous evidence better matched the user's preference.\n"
            + "\n".join(history_lines)
            + "\nUse these interactions to improve the current selection.\n"
        )

    pieces: list[str] = []
    id_to_local: dict[str, int] = {}
    used = len(header)
    for local_index, (_, row) in enumerate(candidates.iterrows()):
        text = clean_text(row.get(text_col, ""))
        if not text:
            continue
        piece = f"[S{local_index}] {text}"
        if used + len(piece) + 1 > int(max_prompt_chars):
            break
        pieces.append(piece)
        id_to_local[f"S{local_index}"] = local_index
        used += len(piece) + 1

    if not pieces:
        raise ValueError("No review sentences fit inside --long-context-max-chars.")

    prompt = header + "\nAVAILABLE REVIEW SENTENCES:\n" + "\n".join(pieces)
    return prompt, id_to_local

def parse_long_context_response(text: str) -> list[str]:
    text = str(text).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise ValueError("Long-context selector did not return a JSON object.")
    payload = json.loads(match.group(0))
    raw_ids = payload.get("selected_ids", [])
    if not isinstance(raw_ids, list):
        raise ValueError("Long-context JSON requires a selected_ids list.")
    normalized_ids: list[str] = []
    for value in raw_ids:
        token = str(value).strip().upper()
        if token.isdigit():
            token = f"S{token}"
        if not token.startswith("S"):
            token = f"S{token}"
        if token not in normalized_ids:
            normalized_ids.append(token)
    return normalized_ids


def run_long_context_selection(
    *,
    generator: Callable[[str], str],
    df_sent: pd.DataFrame,
    user_id: str,
    product_id: str,
    phi_cols: Sequence[str],
    phi_rows: np.ndarray,
    user_profile: str,
    text_col: str,
    length_col: str,
    k: int,
    length_budget: int,
    max_prompt_chars: int,
    interaction_history: Optional[Sequence[Mapping[str, Any]]] = None,
    history_limit: int = 5,
) -> Optional[ExtractionResult]:
    df_sent = ensure_global_index(df_sent)
    candidates = df_sent.loc[df_sent["parent_asin"].astype(str) == str(product_id)].copy()
    if len(candidates) == 0:
        return None
    prompt, id_to_local = make_long_context_prompt(
        user_profile=user_profile,
        item_profile=build_item_profile(candidates, product_id),
        candidates=candidates,
        text_col=text_col,
        k=k,
        length_budget=length_budget,
        max_prompt_chars=max_prompt_chars,
        interaction_history=interaction_history,
        history_limit=history_limit,
    )
    response = generator(prompt)
    selected_ids = parse_long_context_response(response)
    lengths = _lengths(candidates, text_col, length_col)
    selected: list[int] = []
    used_len = 0
    for token in selected_ids:
        if token not in id_to_local:
            continue
        local_index = int(id_to_local[token])
        if len(selected) >= int(k):
            break
        if used_len + int(lengths[local_index]) > int(length_budget):
            continue
        selected.append(local_index)
        used_len += int(lengths[local_index])
    if not selected:
        raise ValueError("Long-context LLM returned no valid evidence IDs within the budget.")
    result = _selected_result(
        method="LongContextLLM",
        candidates=candidates,
        selected_local_idx=selected,
        phi_cols=phi_cols,
        phi_rows=phi_rows,
        user_id=user_id,
        product_id=product_id,
        alpha=None,
        text_col=text_col,
        length_col=length_col,
        meta={
            "prompt_chars": len(prompt),
            "raw_response": str(response),
            "selected_ids": selected_ids,
        },
    )
    return result

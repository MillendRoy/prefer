"""Utilities for creating reproducible mixed-aspect simulated-user personas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .llm_judge import UserPersona


DETAIL_LEVELS = ("concise", "balanced", "detailed")
SKEPTICISM_LEVELS = ("low", "medium", "high")
DECISION_STYLES = ("decisive", "balanced", "cautious")
PREFERENCE_MODES = ("must_cover", "positive_priority", "risk_averse")


def load_aspect_catalog(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("aspects", [])
    if not rows:
        raise ValueError(f"No aspects were found in {path}.")
    required = {"aspect_id", "label", "description"}
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"Aspect catalog row is missing {sorted(missing)}: {row}")
    return rows


def _persona_description(
    preferences: Sequence[Mapping[str, Any]],
    *,
    detail_preference: str,
    evidence_skepticism: str,
    decision_style: str,
) -> str:
    ordered = sorted(preferences, key=lambda item: -float(item["weight"]))
    topics = ", ".join(
        f"{item['label']} ({float(item['weight']):.0%})" for item in ordered
    )
    return (
        f"This shopper has a mixed preference profile centered on {topics}. "
        f"They prefer {detail_preference} summaries, have {evidence_skepticism} "
        f"skepticism toward unsupported claims, and use a {decision_style} decision style."
    )


def generate_mixed_personas(
    catalog: Sequence[Mapping[str, Any]],
    *,
    n_personas: int,
    seed: int = 2026,
    min_aspects: int = 2,
    max_aspects: int = 5,
    dirichlet_concentration: float = 0.65,
    risk_averse_probability: float = 0.25,
    positive_priority_probability: float = 0.35,
) -> list[UserPersona]:
    """Sample heterogeneous mixtures over human-readable latent-aspect labels.

    A concentration below one creates nonuniform mixtures in which one or two
    interests often dominate, while the remaining interests still matter.
    """

    if n_personas < 1:
        raise ValueError("n_personas must be positive.")
    if not 1 <= min_aspects <= max_aspects <= len(catalog):
        raise ValueError("Require 1 <= min_aspects <= max_aspects <= len(catalog).")
    if dirichlet_concentration <= 0.0:
        raise ValueError("dirichlet_concentration must be positive.")

    rng = np.random.default_rng(seed)
    personas: list[UserPersona] = []

    for index in range(n_personas):
        count = int(rng.integers(min_aspects, max_aspects + 1))
        chosen_indices = rng.choice(len(catalog), size=count, replace=False)
        weights = rng.dirichlet(np.full(count, dirichlet_concentration))

        preferences: list[dict[str, Any]] = []
        for aspect_index, weight in zip(chosen_indices, weights):
            aspect = dict(catalog[int(aspect_index)])
            draw = float(rng.random())
            if draw < risk_averse_probability:
                mode = "risk_averse"
            elif draw < risk_averse_probability + positive_priority_probability:
                mode = "positive_priority"
            else:
                mode = "must_cover"

            preferences.append(
                {
                    "aspect_id": str(aspect["aspect_id"]),
                    "label": str(aspect["label"]),
                    "description": str(aspect["description"]),
                    "weight": float(weight),
                    "mode": mode,
                    "representative_examples": [
                        str(value)
                        for value in aspect.get("representative_examples", [])[:3]
                    ],
                }
            )

        detail = str(rng.choice(DETAIL_LEVELS, p=[0.30, 0.45, 0.25]))
        skepticism = str(rng.choice(SKEPTICISM_LEVELS, p=[0.20, 0.50, 0.30]))
        decision_style = str(rng.choice(DECISION_STYLES, p=[0.25, 0.45, 0.30]))

        persona_payload = {
            "persona_id": f"llm_persona_{index:03d}",
            "description": _persona_description(
                preferences,
                detail_preference=detail,
                evidence_skepticism=skepticism,
                decision_style=decision_style,
            ),
            "preferences": preferences,
            "detail_preference": detail,
            "evidence_skepticism": skepticism,
            "decision_style": decision_style,
            "generation_seed": seed,
        }
        personas.append(UserPersona.model_validate(persona_payload))

    return personas


def save_personas(personas: Sequence[UserPersona], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"personas": [persona.model_dump(mode="json") for persona in personas]}
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

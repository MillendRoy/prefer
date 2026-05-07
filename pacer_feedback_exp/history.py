from typing import Optional
import json
import numpy as np
from .schemas import InteractionEvent
import os

def _to_jsonable(obj):
    """
    Recursively convert numpy / pandas-friendly objects into JSON-serializable types.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]

    return str(obj)

def build_history_event(
    t: int,
    selected_df,
    z_t,
    feedback: float,
    product_id: str,
    user_id: str,
    method: str,
    alpha: Optional[np.ndarray] = None,
    meta: Optional[dict] = None,
) -> InteractionEvent:
    if "global_idx" not in selected_df.columns:
        raise ValueError("selected_df must contain a 'global_idx' column.")

    return InteractionEvent(
        t=t,
        user_id=str(user_id),
        product_id=str(product_id),
        method=str(method),
        S_idx=selected_df["global_idx"].to_numpy(dtype=int),
        z_t=np.asarray(z_t, dtype=np.float32),
        f=float(feedback),
        alpha=None if alpha is None else np.asarray(alpha, dtype=np.float32),
        meta={} if meta is None else meta,
    )


def event_to_jsonable(ev: InteractionEvent):
    return {
        "t": ev.t,
        "user_id": ev.user_id,
        "product_id": ev.product_id,
        "method": ev.method,
        "S_idx": ev.S_idx.tolist(),
        "z_t": ev.z_t.tolist(),
        "f": float(ev.f),
        "alpha": None if ev.alpha is None else ev.alpha.tolist(),
        "meta": _to_jsonable(ev.meta),
    }


def save_history_jsonl(history, path: str):
    """
    Save interaction history to a JSONL file.
    Creates parent directories automatically if they do not exist.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for ev in history:
            f.write(json.dumps(event_to_jsonable(ev)) + "\n")
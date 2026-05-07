from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

@dataclass
class ExtractionResult:
    method: str
    user_id: Optional[str]
    product_id: str

    selected_df: pd.DataFrame
    selected_local_idx: np.ndarray
    selected_global_idx: np.ndarray

    alpha: np.ndarray
    z_t: np.ndarray

    total_len: int
    n_candidates: int

    scores: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InteractionEvent:
    t: int
    user_id: str
    product_id: str
    method: str
    S_idx: np.ndarray
    z_t: np.ndarray
    f: float
    alpha: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ExperimentResult:
    history: List[InteractionEvent]
    logs: List[Dict[str, Any]]
    w_init: np.ndarray
    w_true: Optional[np.ndarray]
    w_boltz_final: np.ndarray
    w_omd_final: np.ndarray
    w_omd_avg_final: np.ndarray
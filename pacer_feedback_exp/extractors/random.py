"""Uniform random evidence baseline for independent-judge experiments."""

from __future__ import annotations

import numpy as np

from .base import BaseExtractor


class RandomExtractor(BaseExtractor):
    def __init__(self, *, seed: int = 42, k: int = 10, L: int = 1000, **kwargs):
        super().__init__(alpha_mode="uniform", **kwargs)
        self.rng = np.random.default_rng(seed)
        self.k = int(k)
        self.L = int(L)

    def _select_local_indices(self, cand, utility_scores, phi, w_u, tau_ext, tau_alpha):
        del utility_scores, phi, w_u, tau_ext, tau_alpha
        lengths = cand[self.len_col].to_numpy(dtype=np.int64)
        selected: list[int] = []
        total_len = 0
        for index in self.rng.permutation(len(cand)):
            length = int(lengths[index])
            if total_len + length > self.L:
                continue
            selected.append(int(index))
            total_len += length
            if len(selected) >= self.k:
                break
        return {
            "selected_local_idx": np.asarray(selected, dtype=int),
            "selection_order_scores": np.zeros(len(cand), dtype=np.float32),
            "total_len": total_len,
            "meta": {},
        }

    def _finalize_selected_df(self, selected_df, cand, selection_output):
        del cand, selection_output
        return selected_df

    def _alpha_scores_for_selected_df(self, selected_df):
        return np.zeros(len(selected_df), dtype=np.float32)

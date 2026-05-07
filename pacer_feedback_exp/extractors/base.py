from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from .common import (
    prepare_candidates,
    compute_sentence_utilities,
    compute_alpha,
)
from pacer_feedback_exp.schemas import ExtractionResult
from pacer_feedback_exp.preference import aspect_profile
from pacer_feedback_exp.utils import get_phi_matrix, ensure_global_index, ensure_length_col

class BaseExtractor(ABC):
    def __init__(
        self,
        *,
        text_col="review_text",
        len_col="len",
        alpha_mode="utility_softmax",
        tau_alpha=1.0,
        rank_decay=0.3,
    ):
        """
        Base class for sentence extractors.
        text_col: column name in df_sent that contains the sentence text
        len_col: column name in df_sent that contains the sentence length (for length-constrained extraction)
        alpha_mode: how to compute alpha weights over selected sentences for aspect aggregation ()
        tau_alpha: temperature for computing alpha weights
        rank_decay: decay factor for utility-based alpha weighting (if applicable)
        """
        self.text_col = text_col
        self.len_col = len_col
        self.alpha_mode = alpha_mode
        self.tau_alpha = tau_alpha
        self.rank_decay = rank_decay

    @abstractmethod
    def _select_local_indices(
        self,
        cand: pd.DataFrame,
        utility_scores: np.ndarray,
        phi_cols,
        w_u: np.ndarray,
        tau_ext: float,
    ):
        """
        Must return a dict with at least:
          {
            "selected_local_idx": np.ndarray,
            "selection_order_scores": np.ndarray,   # aligned with candidate rows OR selected rows
            "total_len": int,
            "meta": dict,
          }
        """
        raise NotImplementedError

    def extract(
        self,
        *,
        df_sent: pd.DataFrame,
        user_id: str | None,
        product_id: str,
        phi_cols,
        w_u: np.ndarray,
        phi_rows: np.ndarray | None = None,
        utility_lam: float = 0.0,
        tau_ext : float = 1.0,
    ):
        """
        What are phi_cols and phi_rows?
            - phi_cols: list of column names in df_sent that correspond to aspect features
            - phi_rows: matrix of aspect features for all sentences (if None, will be computed)

        1. Filter candidate sentences for the given product_id
        2. Compute utility scores for each candidate sentence
        3. Select a subset of sentences using some extraction strategy
        4. Reorder/finalize the selected sentences for display
        5. Compute weights alpha over the selected sentences
        6. Aggregate aspect profile z_t from the selected sentences
        Return everything in an ExtractionResult
        """

        # Step 1. select candidate sentences for this product
        cand = prepare_candidates(
            df_sent=df_sent,
            product_id=product_id,
            phi_cols=phi_cols,
            text_col=self.text_col,
            len_col=self.len_col,
        )
        if cand is None or len(cand) == 0:
            return None

        if phi_rows is None:
            # use whole df_sent ordering
            phi_rows = get_phi_matrix(df_sent, phi_cols)
            #phi_rows represents the aspect features for all sentences in df_sent, aligned by global index

        # Step 2: compute utility scores for each candidate sentence : Section 3.1 Rel_i(u,p) 
        U, Phi = compute_sentence_utilities(
            cand,
            phi_cols=phi_cols,
            w_u=w_u,
            lam=utility_lam,
            len_col=self.len_col,
        )
        # Phi shape : (n_candidates, K) represents the aspect features for each candidate sentence
        # U shape : (n_candidates,)

        # Step 3: select sentences based on utility scores and extractor-specific strategy
        # Not Implemented in Base Class, look into Child Classes gumbel and MMR extractors for examples.
        sel = self._select_local_indices(
            cand=cand,
            utility_scores=U,
            phi = Phi,
            w_u=w_u,
            tau_ext=tau_ext, # large tau_ext -> more deterministic selection of top utility sentences, small tau_ext -> more random exploration
            # when tau_ext = 0, selection is uniform random among feasible candidates, when tau_ext -> inf, selection is deterministic top-k by utility
            tau_alpha = self.tau_alpha, # large tau_alpha -> more focused alpha weights on highest utility sentences, small tau_alpha -> more uniform alpha weights
        )
        selected_local_idx = np.asarray(sel["selected_local_idx"], dtype=int)
        if len(selected_local_idx) == 0:
            return None
        selected_df = cand.iloc[selected_local_idx].copy()
        # attach utility
        selected_df["utility"] = U[selected_local_idx]

        # Step 4: extractor-specific reordering/finalization of selected sentences for display
        # extractor-specific displayed ordering
        selected_df = self._finalize_selected_df(
            selected_df=selected_df,
            cand=cand,
            # selected_local_idx=selected_local_idx,
            selection_output=sel,
        )

        selected_global_idx = selected_df["global_idx"].to_numpy(dtype=int)

        # Step 5: compute alpha weights over the selected sentences for aspect aggregation
        # alpha should align with final selected_df order (Section 3.3)
        # alpha is based on the utility scores of the selected sentences by default, but can be customized by overriding _alpha_scores_for_selected_df in child classes.
        aligned_scores = self._alpha_scores_for_selected_df(selected_df)
        alpha = compute_alpha(
            aligned_scores,
            mode=self.alpha_mode,
            tau_alpha=self.tau_alpha,
            rank_decay=self.rank_decay,
        )

        # Step 6: aggregate aspect profile z_t from the selected sentences using the computed alpha weights
        z_t = aspect_profile(phi_rows, selected_global_idx, alpha=alpha)
        selected_df["alpha"] = alpha

        # what is the difference between selected_local_idx and selected_global_idx?
        # selected_local_idx refers to the indices of the selected sentences within the filtered candidate set (cand), while selected_global_idx refers to the indices of the selected sentences within the original df_sent dataframe. The global indices are important for correctly aligning with the aspect matrix phi_rows and for any downstream processing
        return ExtractionResult(
            method=self.__class__.__name__,
            user_id=user_id,
            product_id=product_id,
            selected_df=selected_df,
            selected_local_idx=selected_local_idx,
            selected_global_idx=selected_global_idx,
            alpha=alpha,
            z_t=z_t,
            total_len=int(sel["total_len"]),
            n_candidates=len(cand),
            scores=self._collect_scores(selected_df, sel),
            meta=sel.get("meta", {}),
        )

    def _finalize_selected_df(self, selected_df, cand, selected_local_idx, selection_output):
        return selected_df

    def _alpha_scores_for_selected_df(self, selected_df):
        """
        Default: use utility for alpha.
        """
        return selected_df["utility"].to_numpy(dtype=np.float32)

    def _collect_scores(self, selected_df, selection_output):
        return {}
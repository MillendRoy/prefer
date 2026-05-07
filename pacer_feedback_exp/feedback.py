from abc import ABC, abstractmethod
from typing import Callable, Optional
import numpy as np
from .utils import sigmoid, normalize_simplex

class FeedbackProvider(ABC):
    @abstractmethod
    def get_feedback(self, *, user_id, product_id, z_t, selected_df, summary_text=None, t=None) -> float:
        """
        Return scalar feedback in [0,1].
        """
        raise NotImplementedError


class SyntheticFeedbackProvider(FeedbackProvider):
    """
    Synthetic feedback based on hidden true preference vector w_true.
    """
    def __init__(self, w_true, gamma=10.0, noise_std=0.05, seed=42):
        """
        w_true: hidden oracle user preference vector (for feedback generation)
        gamma: sensitivity parameter for feedback generation (higher means more sensitive to differences in z_t)
        noise_std: standard deviation of Gaussian noise added to the feedback signal (for realism)
        seed: random seed for reproducibility
        """
        self.w_true = np.asarray(w_true, dtype=np.float32)
        self.gamma = float(gamma)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)

    def get_feedback(self, *, user_id, product_id, z_t, selected_df, summary_text=None, t=None) -> float:
        score = float(np.dot(self.w_true, np.asarray(z_t, dtype=np.float32)))
        noisy_score = score + self.rng.normal(0.0, self.noise_std)
        f_t = sigmoid(self.gamma * (noisy_score - float(np.mean(self.w_true))))
        return float(np.clip(f_t, 0.0, 1.0))

class TimeVaryingSyntheticFeedbackProvider(FeedbackProvider):
    """
    Synthetic feedback based on a hidden time-varying true preference vector.

    The oracle preference starts at w_start and gradually or abruptly changes
    toward w_end. At each round t, feedback is generated using w_true_t.
    """

    def __init__(
        self,
        w_start,
        w_end,
        drift_start: int = 30,
        drift_end: int = 70,
        gamma: float = 10.0,
        noise_std: float = 0.05,
        seed: int = 42,
        mode: str = "linear",
    ):
        """
        w_start:
            Initial hidden oracle preference vector.

        w_end:
            Final hidden oracle preference vector after drift.

        drift_start:
            Round at which the preference drift begins.

        drift_end:
            Round at which the preference drift ends.

        gamma:
            Sensitivity parameter for feedback generation.

        noise_std:
            Standard deviation of Gaussian noise added to the feedback signal.

        seed:
            Random seed for reproducibility.

        mode:
            "linear" for gradual drift from w_start to w_end.
            "abrupt" for sudden switch from w_start to w_end at drift_start.
        """
        self.w_start = normalize_simplex(np.asarray(w_start, dtype=np.float32))
        self.w_end = normalize_simplex(np.asarray(w_end, dtype=np.float32))

        self.drift_start = int(drift_start)
        self.drift_end = int(drift_end)

        if self.drift_end < self.drift_start:
            raise ValueError("drift_end must be greater than or equal to drift_start.")

        self.gamma = float(gamma)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)
        self.mode = mode

        # This keeps compatibility with existing code that expects feedback_provider.w_true.
        # It represents the initial preference, but the actual feedback uses get_true_preference(t).
        self.w_true = self.w_start.copy()


    def get_true_preference(self, t: int):
        """
        Return the current oracle preference vector w_true_t.
        """
        if t is None:
            t = 0

        t = int(t)

        if self.mode == "abrupt":
            if t < self.drift_start:
                return self.w_start.copy()
            return self.w_end.copy()

        if self.mode == "linear":
            if t <= self.drift_start:
                rho_t = 0.0
            elif t >= self.drift_end:
                rho_t = 1.0
            else:
                rho_t = (t - self.drift_start) / max(1, self.drift_end - self.drift_start)

            w_t = (1.0 - rho_t) * self.w_start + rho_t * self.w_end
            return normalize_simplex(w_t)

        raise ValueError(f"Unknown drift mode: {self.mode}")

    def get_feedback(self, *, user_id, product_id, z_t, selected_df, summary_text=None, t=None) -> float:
        """
        Return scalar feedback in [0,1] using the current drifting preference.

        Feedback equation:
            score_t = w_true_t^T z_t
            noisy_score_t = score_t + epsilon_t
            f_t = sigmoid(gamma * (noisy_score_t - mean(w_true_t)))
        """
        w_true_t = self.get_true_preference(t)
        z_t = np.asarray(z_t, dtype=np.float32)

        score = float(np.dot(w_true_t, z_t))
        noisy_score = score + self.rng.normal(0.0, self.noise_std)

        f_t = sigmoid(self.gamma * (noisy_score - float(np.mean(w_true_t))))

        return float(np.clip(f_t, 0.0, 1.0))
    

class ManualFeedbackProvider(FeedbackProvider):
    """
    For real experiments in notebook / terminal:
    asks a human for a number in [0,1].
    """
    def get_feedback(self, *, user_id, product_id, z_t, selected_df, summary_text=None, t=None) -> float:
        print(f"\nUser: {user_id} | Product: {product_id} | Round: {t}")
        if summary_text is not None:
            print("\nSummary shown to user:\n")
            print(summary_text)

        while True:
            raw = input("\nEnter feedback in [0,1]: ").strip()
            try:
                val = float(raw)
                if 0.0 <= val <= 1.0:
                    return val
            except Exception:
                pass
            print("Invalid input. Please enter a number between 0 and 1.")


class CallbackFeedbackProvider(FeedbackProvider):
    """
    Lets you plug in web app / API / UI callbacks later.
    """
    def __init__(self, fn: Callable):
        self.fn = fn

    def get_feedback(self, *, user_id, product_id, z_t, selected_df, summary_text=None, t=None) -> float:
        val = self.fn(
            user_id=user_id,
            product_id=product_id,
            z_t=z_t,
            selected_df=selected_df,
            summary_text=summary_text,
            t=t,
        )
        val = float(val)
        return max(0.0, min(1.0, val))
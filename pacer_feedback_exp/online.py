import numpy as np
from .utils import softmax, normalize_simplex


def init_boltz_state(K: int):
    """
    Initialize the state for the online Boltzmann estimation of user preferences.
        K: number of aspects
    """
    return {
        "num": np.zeros(K, dtype=np.float32),
        "den": 0.0,
    }


def online_boltz_update(state, z_t, f_t, beta=5.0, eps=1e-8, prior=None):
    z_t = np.asarray(z_t, dtype=np.float32)

    state["num"] += float(f_t) * z_t
    state["den"] += float(f_t)

    z_tilde = state["num"] / (state["den"] + eps)
    w = softmax(beta * z_tilde, axis=0)

    if prior is not None:
        prior = normalize_simplex(prior)
        w = 0.9 * w + 0.1 * prior
        w = normalize_simplex(w)

    return w, state


# def init_omd_state(w_init):
#     return {"w": normalize_simplex(w_init).copy()}

def init_omd_state(w_init, baseline_mode="running_mean", ema_alpha=0.1):

    return {
        "w": normalize_simplex(w_init).copy(),
        "y": np.log(np.clip(w_init, 1e-12, None)),
        "t": 0,
        "feedback_sum": 0.0,
        "baseline": 0.0,
        "baseline_mode": baseline_mode,
        "ema_alpha": ema_alpha,
        "w_avg": normalize_simplex(w_init).copy(),  # for debugging, track average w over time
    }


def online_omd_update(state, z_t, f_t, eta=1.0):
    """
    Entropic OMD / exponentiated gradient:
        w_{t+1,k} ∝ w_{t,k} exp(eta * f_t * z_{t,k})
    """
    w_t = normalize_simplex(state["w"])
    z_t = np.asarray(z_t, dtype=np.float32)

    mult = np.exp(eta * float(f_t) * z_t)
    w_next = normalize_simplex(w_t * mult)

    state["w"] = w_next
    return w_next, state


def online_omd_update_centered(
    state,
    *,
    z_t,
    f_t,
    eta=0.3,
    clip_centered_feedback=None,
):
    """
    Entropic OMD / exponentiated gradient update with centered feedback.

    Uses:
        f_eff = f_t - baseline_t

    Update:
        y_{t+1} = y_t + eta * f_eff * z_t
        w_{t+1} = softmax(y_{t+1})

    because gradient is:
        g_t = -(f_eff * z_t)
    and OMD uses:
        y_{t+1} = y_t - eta * g_t = y_t + eta * f_eff * z_t
    """
    z_t = np.asarray(z_t, dtype=float)
    y = state["y"]
    baseline_mode = state["baseline_mode"]

    # baseline BEFORE incorporating current feedback
    if state["t"] == 0:
        baseline_t = 0.0
    else:
        baseline_t = state["baseline"]

    f_eff = float(f_t) - float(baseline_t)

    if clip_centered_feedback is not None:
        c = float(clip_centered_feedback)
        f_eff = np.clip(f_eff, -c, c)

    # OMD dual update
    y_new = y + eta * f_eff * z_t
    w_new = softmax(y_new)

    t_new = state["t"] + 1

    w_avg_old = state["w_avg"]
    w_avg_new = ((t_new - 1) / t_new) * w_avg_old + (1.0 / t_new) * w_new
    w_avg_new = w_avg_new / w_avg_new.sum()

    # update baseline AFTER using current feedback
    if baseline_mode == "running_mean":
        new_feedback_sum = state["feedback_sum"] + float(f_t)
        new_t = state["t"] + 1
        new_baseline = new_feedback_sum / new_t

    elif baseline_mode == "ema":
        alpha = state["ema_alpha"]
        if state["t"] == 0:
            new_baseline = float(f_t)
        else:
            new_baseline = (1 - alpha) * state["baseline"] + alpha * float(f_t)
        new_feedback_sum = state["feedback_sum"] + float(f_t)
        new_t = state["t"] + 1

    else:
        raise ValueError(f"Unknown baseline_mode: {baseline_mode}")

    new_state = {
        **state,
        "w": w_new,
        "y": y_new,
        "t": new_t,
        "feedback_sum": new_feedback_sum,
        "baseline": new_baseline,
        "w_avg": w_avg_new,
    }

    return w_new, new_state, {
        "baseline_t": baseline_t,
        "f_eff": f_eff,
        "w_avg": w_avg_new,
    }
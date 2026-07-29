from typing import Callable, List, Optional
import numpy as np

from .schemas import ExperimentResult
from .history import build_history_event
from .preference import (
    estimate_user_pref_from_own_text,
    make_synthetic_true_preference,
)
from .feedback import FeedbackProvider
from .online import (
    init_boltz_state,
    online_boltz_update,
    init_omd_state,
    online_omd_update,
    online_omd_update_centered
)
from .utils import kl_divergence, get_phi_matrix,ensure_global_index, normalize_simplex

from sklearn.metrics.pairwise import cosine_similarity

from .extractors.base import BaseExtractor

def run_online_experiment(
    *,
    df_sent,
    user_id: str,
    product_ids: List[str],
    phi_cols,
    extractor:  BaseExtractor,
    feedback_provider: FeedbackProvider,
    method_name: str, #options are "gumbel", "mmr", "random"
    beta_init: float = 8.0,
    beta_boltz: float = 8.0,
    eta_omd: float = 3.0,
    delta_omd: Optional[float] = None,
    c_eta: float = 0.05,
    baseline_mode: str = "ema",  # "ema" or "last" or None
    ema_alpha: float = 0.1,  # for EMA baseline if baseline_mode == "ema"
    clip_centered_feedback : float = 0.15,   
    utility_lam: float = 0.0,
    use_policy: str = "omd",
    synthetic_oracle: bool = False,
    oracle_noise_scale: float = 0.03,
    seed: int = 42,
    w_init_override: Optional[np.ndarray] = None,
    verbose: bool = True,
):
    """
    Runs an online learning experiment with the given extractor and feedback provider.
        - df_sent: DataFrame containing sentences with columns for user_id, product_id, and
            aspect features (phi_cols).
        - user_id: the user for whom we are running the experiment
        - product_ids: list of product_ids to iterate over (rounds)
        - phi_cols: list of column names in df_sent that correspond to aspect features
        - extractor: an instance of BaseExtractor that will select sentences and compute z_t
        - feedback_provider: an instance of FeedbackProvider that will provide feedback f_t based on selected
            sentences and their aspect profile z_t
        - method_name: string name for the method (for logging)
        - beta_init: initial inverse temperature for preference estimation from own text
        - beta_boltz: inverse temperature for Boltzmann updates
        - eta_omd: learning rate for OMD updates
        - use_policy: which preference vector to use for extraction ("boltz" or "omd")
        - synthetic_oracle: whether to create a synthetic oracle preference w_true for testing
        - oracle_noise_scale: noise scale for creating synthetic oracle preference
        - seed: random seed for reproducibility
    The experiment proceeds in rounds, iterating over the given product_ids. 
    Initialize w_init -> interact -> update w -> track learning dynamics and logs.
    At round t, it does:
    1. pick current preference w_current
    2. call extract(...) for that product_id
    3. get feedback f_t
    4. update w_boltz and w_omd
    5. store logs
    """

    # Preprocessing: ensure global index and get phi_rows for aspect profile computations.
    df_sent = ensure_global_index(df_sent)
    phi_rows = get_phi_matrix(df_sent, phi_cols)
    K = len(phi_cols)

    # Step 1: Estimate initial user preference w_init from their own text (bootstrapping).
    # w_init = estimate_user_pref_from_own_text(
    #     df_sent=df_sent,
    #     user_id=user_id,
    #     phi_cols=phi_cols,
    #     beta=beta_init,
    # )
    # Use the paper's uniform initialization unless an experiment explicitly
    # supplies a learner-side profile, e.g. from the fit half of a held-out split.
    if w_init_override is None:
        w_init = np.ones(K, dtype=np.float32) / K
    else:
        candidate_w_init = np.asarray(w_init_override, dtype=np.float32)
        if candidate_w_init.shape != (K,):
            raise ValueError(
                f"w_init_override must have shape {(K,)}, got {candidate_w_init.shape}."
            )
        if not np.all(np.isfinite(candidate_w_init)):
            raise ValueError("w_init_override contains NaN or infinity.")
        w_init = normalize_simplex(candidate_w_init)


    # Step 2: We don't necessarily need it here, since feedback_provider can use phi_rows and w_true directly, but we can create a synthetic oracle w_true for testing if desired.
    w_true = getattr(feedback_provider, "w_true", None) if synthetic_oracle else None
    # if synthetic_oracle:
    #     w_true = make_synthetic_true_preference(
    #         df_sent=df_sent,
    #         user_id=user_id,
    #         phi_cols=phi_cols,
    #         beta=beta_init,
    #         noise_scale=oracle_noise_scale,
    #         seed=seed,
    #     )

    # Step 3: Initialize online learning states for Boltzmann and OMD updates.
    boltz_state = init_boltz_state(K)
    omd_state = init_omd_state(w_init, baseline_mode=baseline_mode, ema_alpha=ema_alpha)
    w_boltz = w_init.copy()
    w_omd = w_init.copy()

    z_sum = np.zeros(K, dtype=float)
    num_valid_rounds = 0
    history = []
    logs = []

    # Step 4: Main online learning loop over product_ids (rounds).
    for t, product_id in enumerate(product_ids):
        if use_policy == "omd":
            w_current = w_omd
        elif use_policy == "boltz":
            w_current = w_boltz
        elif use_policy == "static":
            w_current = w_init
        else:
            raise ValueError(f"Unknown use_policy={use_policy}")

        # w_current = w_omd if use_policy == "omd" else w_boltz
        w_before_update = w_current.copy()
        
        # how did we come up with this schedule for tau_ext? 
        # It's a common practice to increase exploration (higher tau_ext) in early rounds and 
        # decrease it over time to allow for more exploitation as we learn more about the user's preferences. The specific parameters (1.0, 50.0, log(t+2)) can be tuned based on the desired exploration-exploitation tradeoff and the expected number of rounds. You could experiment with different schedules (e.g., linear decay, exponential decay) to see what works best in your setting.
        tau_t = min(200.0, 1.0+50.0*np.log(t + 2))


        out = extractor.extract(
            df_sent=df_sent,
            user_id=user_id,
            product_id=product_id,
            phi_cols=phi_cols,
            w_u=w_current,
            phi_rows=phi_rows,
            utility_lam=utility_lam,
            tau_ext=tau_t,
        ) 
        # out represents the ExtractionResult containing selected sentences, 
        # their aspect profile z_t, and other metadata for this round's extraction based on the current preference w_current.
        # The extractor will use w_current to compute utility scores and select sentences accordingly.
        
        
        # actual_reward = float(np.dot(w_true, out.z_t))

        # oracle_out = extractor.extract(
        #     df_sent=df_sent,
        #     user_id=user_id,
        #     product_id=product_id,
        #     phi_cols=phi_cols,
        #     w_u=w_true,
        #     phi_rows=phi_rows,
        #     utility_lam=utility_lam,
        #     tau_ext=tau_t,
        # )
        # oracle_reward = float(np.dot(w_true, oracle_out.z_t))
        # instant_regret = oracle_reward - actual_reward


        if out is None or len(out.selected_df) == 0:
            continue

        z_sum += np.asarray(out.z_t, dtype=float)
        num_valid_rounds += 1
        z_avg_t = z_sum / num_valid_rounds

        f_t = feedback_provider.get_feedback(
            user_id=user_id,
            product_id=product_id,
            z_t=out.z_t,
            selected_df=out.selected_df,
            summary_text=out.meta.get("summary_text", None),
            t=t,
        )

        if hasattr(feedback_provider, "get_true_preference"):
            w_true_t = feedback_provider.get_true_preference(t)
        else:
            w_true_t = w_true

        event = build_history_event(
            t=t,
            selected_df=out.selected_df,
            z_t=out.z_t,
            feedback=f_t,
            product_id=product_id,
            user_id=user_id,
            method=out.method,
            alpha=out.alpha,
            meta=out.meta,
        )
        history.append(event)

        w_boltz, boltz_state = online_boltz_update(
            boltz_state,
            z_t=out.z_t,
            f_t=f_t,
            beta=beta_boltz,
            prior=w_init,
        )

        # w_omd, omd_state = online_omd_update(
        #     omd_state,
        #     z_t=out.z_t,
        #     f_t=f_t,
        #     eta=eta_omd,
        # )

        # choice of eta_t is important for stability and convergence of OMD updates. 
        # A common choice is to use a decaying learning rate that decreases over time, 
        # such as eta_t = eta_omd / sqrt(t+1) or eta_t = eta_omd / log(t+2). 
        # This allows for larger updates in early rounds when we are more uncertain about 
        # the user's preferences, and smaller updates in later rounds as we gather
        # more information and want to fine-tune the preference vector. 
        # You can experiment with different decay schedules to see what works best in your setting.
        round_id = t + 1
        eta_t = eta_omd / np.sqrt(c_eta*round_id+1)
        w_omd, omd_state, omd_dbg = online_omd_update_centered(
            omd_state,
            z_t=out.z_t,
            f_t=f_t,
            eta=eta_t,
            clip_centered_feedback= clip_centered_feedback,   # optional, helps stability
        )

        # print(
        #     "t=", t,
        #     "feedback=", round(float(f_t), 4),
        #     "w_before_top=", np.argsort(-w_before_update)[:3],
        #     "w_after_top=", np.argsort(-w_omd)[:3],
        #     "w_after=", np.round(w_omd, 4),
        #     "z_top=", np.argsort(-out.z_t)[:3],
        #     "z=", np.round(out.z_t, 4),
        # )

        if delta_omd is not None:
            if K * delta_omd >= 1.0:
                raise ValueError("delta_omd must satisfy K * delta_omd < 1.")

            w_omd = np.asarray(w_omd, dtype=float)
            w_omd = np.maximum(w_omd, 0.0)
            w_omd = w_omd / w_omd.sum()

            # Shrink toward the uniform interior floor.
            w_omd = delta_omd + (1.0 - K * delta_omd) * w_omd
            w_omd = w_omd / w_omd.sum()

            omd_state["w"] = w_omd.copy()

        if use_policy == "static":
            # For static baselines, the policy remains fixed at initialization.
            # We still compute feedback for evaluation, but do not let the logged
            # preference vector drift.
            w_logged = w_init.copy()
            w_logged_avg = w_init.copy()
        else:
            w_logged = w_omd.copy()
            w_logged_avg = omd_state["w_avg"].copy()

        # w_omd_avg = omd_state["w_avg"].copy()

        log = {
            "t": t,
            "product_id": product_id,
            "feedback": float(f_t),
            "z_t": out.z_t.copy(),
            "alpha": out.alpha.copy(),
            "selected_global_idx": out.selected_global_idx.copy(),
            "w_boltz": w_boltz.copy(),
            "w_omd": w_logged.copy(),
        }
        log["omd_baseline"] = omd_dbg["baseline_t"]
        log["omd_f_eff"] = omd_dbg["f_eff"]
        log["w_omd_avg"] = w_logged_avg

        if w_true_t is not None:

            log["w_true"] = w_true_t.copy()
            # log["A_pref"] = float(np.dot(w_true, w_logged))
            
            # log["A_evid"] = float(np.dot(w_true, out.z_t))
            log["cos_boltz"] = float(
                cosine_similarity(w_boltz.reshape(1, -1), w_true_t.reshape(1, -1))[0, 0]
            )

            log["cos_omd"] = float(
                cosine_similarity(w_logged.reshape(1, -1), w_true_t.reshape(1, -1))[0, 0]
            )

            log["cos_omd_avg"] = float(
                cosine_similarity(w_logged_avg.reshape(1, -1), w_true_t.reshape(1, -1))[0, 0]
            )

            log["cos_evid"] = float(
                cosine_similarity(np.asarray(out.z_t).reshape(1, -1), w_true_t.reshape(1, -1))[0, 0]
            )

            log["cos_evid_avg"] = float(
                cosine_similarity(z_avg_t.reshape(1, -1), w_true_t.reshape(1, -1))[0, 0]
            )
            log["kl_boltz_to_true"] = kl_divergence(w_boltz, w_true_t)
            log["kl_omd_to_true"] = kl_divergence(w_logged, w_true_t)

            f_eff = float(omd_dbg["f_eff"])
            # Surrogate losses:
            # ell_t(w) = - f_eff * <z_t, w>
            loss_policy = -f_eff * float(np.dot(out.z_t, w_before_update))
            loss_true = -f_eff * float(np.dot(out.z_t, w_true_t))

            inst_surrogate_regret = loss_policy - loss_true

            prev_cum_surrogate_regret = (
                logs[-1]["cum_surrogate_regret"]
                if len(logs) > 0 and "cum_surrogate_regret" in logs[-1]
                else 0.0
            )

            cum_surrogate_regret = prev_cum_surrogate_regret + inst_surrogate_regret
            avg_surrogate_regret = cum_surrogate_regret / round_id

            # Theoretical bound from Theorem
            delta_for_bound = delta_omd if delta_omd is not None else float(np.min(w_init))
            if clip_centered_feedback is not None:
                c_for_bound = float(clip_centered_feedback)
            else:
                c_for_bound = 1.0

            theory_bound = (
                (np.log(1.0 / delta_for_bound) / eta_omd)
                +
                (c_for_bound**2 * eta_omd / c_eta)
            ) * np.sqrt(1.0 + c_eta * round_id)

            theory_avg_bound = theory_bound / round_id
            log["A_pref"] = float(np.dot(w_true_t, w_before_update))
            log["A_pref_avg"] = float(np.dot(w_true_t, w_logged_avg))
            log["A_evid"] = float(np.dot(w_true_t, out.z_t))
            log["A_evid_avg"] = float(np.dot(w_true_t, z_avg_t))

            log["loss_policy"] = loss_policy
            log["loss_true"] = loss_true
            log["inst_surrogate_regret"] = inst_surrogate_regret
            log["cum_surrogate_regret"] = cum_surrogate_regret
            log["avg_surrogate_regret"] = avg_surrogate_regret

            log["theory_regret_bound"] = theory_bound
            log["theory_avg_regret_bound"] = theory_avg_bound

            log["eta_t"] = float(eta_t)
            log["min_w_before_update"] = float(np.min(w_before_update))
            log["min_w_after_update"] = float(np.min(w_omd))


        
        # log["actual_reward"] = actual_reward
        # log["oracle_reward"] = oracle_reward
        # log["inst_regret"] = instant_regret

        # prev_cum_regret = logs[-1]["cum_regret"] if len(logs) > 0 and logs[-1].get("cum_regret") is not None else 0.0
        # if instant_regret is not None:
        #     log["cum_regret"] = prev_cum_regret + instant_regret
        #     log["avg_regret"] = log["cum_regret"] / (len(logs) + 1)
        # else:
        #     log["cum_regret"] = prev_cum_regret
        #     log["avg_regret"] = log["cum_regret"] / max(1, (len(logs) + 1))

        logs.append(log)

        if verbose:
            if use_policy == "omd":
                print(f"Round {t+1}/{len(product_ids)} - Product ID: {product_id}, Feedback: {f_t:.4f}, Cosine OMD: {log.get('cos_omd', 'N/A')}, Selected idx: {out.selected_global_idx}")
                # print("w_omd top aspects", np.argsort(-w_omd)[:5])
                # print("w_omd", np.round(w_omd, 2))
            else:
                print(f"Round {t+1}/{len(product_ids)} - Product ID: {product_id}, Feedback: {f_t:.4f}, Cosine Static: {log.get('cos_boltz', 'N/A')}, Selected idx: {out.selected_global_idx}")
                # print("w_boltz top aspects", np.argsort(-w_boltz)[:5])
                # print("w_boltz", np.round(w_boltz, 2))
            # print("top z aspects", np.argsort(-out.z_t)[:5])
            # print("z_t", np.round(out.z_t, 2))


    return ExperimentResult(
        history=history,
        logs=logs,
        w_init=w_init,
        w_true=w_true_t,
        w_boltz_final=w_boltz,
        w_omd_final=w_omd,
        w_omd_avg_final=omd_state["w_avg"].copy(),
    )
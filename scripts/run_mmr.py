import pandas as pd
from pacer_feedback_exp.feedback import SyntheticFeedbackProvider
from pacer_feedback_exp.experiment import run_online_experiment
from pacer_feedback_exp.preference import make_synthetic_true_preference
from pacer_feedback_exp.history import save_history_jsonl
from pacer_feedback_exp.extractors.mmr import MMRExtractor

data = pd.read_csv("data/train_with_aspect_scores.csv")

phi_cols = [c for c in data.columns if c.startswith("asp_")]
user_id = "AG73BVBKUOH22USSFJA5ZWL7AKXA"
product_ids = ["B085BB7B1M"]*10

w_true = make_synthetic_true_preference(
    df_sent=data,
    user_id=user_id,
    phi_cols=phi_cols,
    beta=8.0,
    noise_scale=0.03,
    seed=42,
)

feedback_provider = SyntheticFeedbackProvider(
    w_true=w_true,
    gamma=12.0,
    noise_std=0.0,
    seed=42,
)

extractor = MMRExtractor(
    k=10,
    L=900,
    lam_mmr=0.7,
    alpha_mode="utility_softmax",
    tau_alpha=1.5,
)

result = run_online_experiment(
    df_sent=data,
    user_id=user_id,
    product_ids=product_ids,
    phi_cols=phi_cols,
    extractor=extractor,
    # extractor_fn=lambda **kwargs: run_mmr_extractor(
    #     extract_mmr_for_user_product,
    #     **kwargs,
    # ),
    # extractor_kwargs={
    #     "k": 10,
    #     "lambda_div": 0.3,
    # },
    feedback_provider=feedback_provider,
    method_name="mmr",
    beta_init=8.0,
    beta_boltz=1.0,
    eta_omd=0.3,
    use_policy="omd",
    synthetic_oracle=True,
    seed=42,

    utility_lam=0.0,
)

print("w_init:\n", result.w_init)
print("\nw_true:\n", result.w_true)
print("\nw_boltz_final:\n", result.w_boltz_final)
# print("\nw_omd_final:\n", result.w_omd_final)
print("\nnum_events:", len(result.history))

save_history_jsonl(result.history, "outputs/history_mmr.jsonl")
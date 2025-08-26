import pandas as pd
import numpy as np
from scipy.special import rel_entr

# === 파일 불러오기 ===

preferred_path_scalar = '/home/hail/Distribution_RLHF/summarize_from_feedback_details/eval_tables/reward_basic__1__1752221539/scalar_logs/preferred_rewards_scalar_step_validation_test.csv'
rejected_path_scalar = '/home/hail/Distribution_RLHF/summarize_from_feedback_details/eval_tables/reward_basic__1__1752221539/scalar_logs/rejected_rewards_scalar_step_validation_test.csv'
preferred_path_quantile = '/home/hail/Distribution_RLHF/summarize_from_feedback_details/eval_tables/reward_qr__1__1752132064/quantile_logs/preferred_rewards_quantile_step_validation_test.csv'
rejected_path_quantile = '/home/hail/Distribution_RLHF/summarize_from_feedback_details/eval_tables/reward_qr__1__1752132064/quantile_logs/rejected_rewards_quantile_step_validation_test.csv'

# === Scalar ===
preferred_scalar = pd.read_csv(preferred_path_scalar, header=None, skiprows=1).iloc[:, 0].astype(float)
rejected_scalar = pd.read_csv(rejected_path_scalar, header=None, skiprows=1).iloc[:, 0].astype(float)

# === Quantile ===
preferred_quantile = pd.read_csv(preferred_path_quantile)
rejected_quantile = pd.read_csv(rejected_path_quantile)

# Quantile 평균으로 Scalar화
preferred_quantile_mean = preferred_quantile.mean(axis=1).astype(float)
rejected_quantile_mean = rejected_quantile.mean(axis=1).astype(float)

# === KL Divergence 함수 ===
def calculate_kl_divergence(preferred, rejected, bins=10000, epsilon=1e-8):
    counts_pref, bin_edges = np.histogram(preferred, bins=bins, density=False)
    counts_rej, _ = np.histogram(rejected, bins=bin_edges, density=False)
    
    P = counts_pref / counts_pref.sum()
    Q = counts_rej / counts_rej.sum()
    
    P = P + epsilon
    Q = Q + epsilon
    
    kl = np.sum(rel_entr(P, Q))
    return kl

# === KL Divergence 계산 ===
kl_scalar = calculate_kl_divergence(preferred_scalar, rejected_scalar)
kl_quantile = calculate_kl_divergence(preferred_quantile_mean, rejected_quantile_mean)

print(f"KL Divergence (Preferred Scalar || Rejected Scalar) = {kl_scalar}")
print(f"KL Divergence (Preferred Quantile || Rejected Quantile) = {kl_quantile}")

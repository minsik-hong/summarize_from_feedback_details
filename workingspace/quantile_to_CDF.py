import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 저장 폴더
save_dir = "./"
os.makedirs(save_dir, exist_ok=True)

# 파일 경로
preferred_path_scalar = '/home/hail/Distribution_RLHF/summarize_from_feedback_details/eval_tables/reward_basic__1__1752221539/scalar_logs/preferred_rewards_scalar_step_validation_test.csv'
rejected_path_scalar = '/home/hail/Distribution_RLHF/summarize_from_feedback_details/eval_tables/reward_basic__1__1752221539/scalar_logs/rejected_rewards_scalar_step_validation_test.csv'
preferred_path_quantile = '/home/hail/Distribution_RLHF/summarize_from_feedback_details/eval_tables/reward_qr__1__1752132064/quantile_logs/preferred_rewards_quantile_step_validation_test.csv'
rejected_path_quantile = '/home/hail/Distribution_RLHF/summarize_from_feedback_details/eval_tables/reward_qr__1__1752132064/quantile_logs/rejected_rewards_quantile_step_validation_test.csv'

# 데이터 불러오기
preferred_scalar = pd.read_csv(preferred_path_scalar, skiprows=1, header=None).iloc[:, 0].astype(float)
rejected_scalar = pd.read_csv(rejected_path_scalar, skiprows=1, header=None).iloc[:, 0].astype(float)

preferred_quantile_df = pd.read_csv(preferred_path_quantile)
rejected_quantile_df = pd.read_csv(rejected_path_quantile)

preferred_quantile = preferred_quantile_df.mean(axis=1).astype(float)
rejected_quantile = rejected_quantile_df.mean(axis=1).astype(float)

# 히스토그램 및 CDF 계산
bins = 10000
counts_pref_scalar, bin_edges = np.histogram(preferred_scalar, bins=bins)
prob_pref_scalar = counts_pref_scalar / counts_pref_scalar.sum()
counts_rej_scalar, _ = np.histogram(rejected_scalar, bins=bin_edges)
prob_rej_scalar = counts_rej_scalar / counts_rej_scalar.sum()
counts_pref_quantile, _ = np.histogram(preferred_quantile, bins=bin_edges)
prob_pref_quantile = counts_pref_quantile / counts_pref_quantile.sum()
counts_rej_quantile, _ = np.histogram(rejected_quantile, bins=bin_edges)
prob_rej_quantile = counts_rej_quantile / counts_rej_quantile.sum()
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# CDF 계산
cdf_pref_scalar = np.cumsum(prob_pref_scalar)
cdf_rej_scalar = np.cumsum(prob_rej_scalar)
cdf_pref_quantile = np.cumsum(prob_pref_quantile)
cdf_rej_quantile = np.cumsum(prob_rej_quantile)

# CDF plot
plt.figure(figsize=(12, 7))
plt.plot(bin_centers, cdf_pref_scalar, color='blue', alpha=0.8, label='Preferred (Scalar)')
plt.plot(bin_centers, cdf_rej_scalar, color='blue', alpha=0.3, label='Rejected (Scalar)')
plt.plot(bin_centers, cdf_pref_quantile, color='orange', alpha=0.8, label='Preferred (Quantile)')
plt.plot(bin_centers, cdf_rej_quantile, color='orange', alpha=0.3, label='Rejected (Quantile)')

plt.xlim(0.0, 10.0)
plt.xlabel('Reward Value', fontsize=18)
plt.ylabel('Cumulative Probability (CDF)', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)

save_path = os.path.join(save_dir, 'cdf_plot.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"CDF 저장 완료: {save_path}")

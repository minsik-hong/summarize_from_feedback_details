import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === 저장 폴더 생성 ===
save_dir = "./"
os.makedirs(save_dir, exist_ok=True)

# === 파일 경로 설정 ===
preferred_path_scalar = '/home/hail/Distribution_RLHF/summarize_from_feedback_details/eval_tables/reward_basic__1__1752221539/scalar_logs/preferred_rewards_scalar_step_validation_test.csv'
rejected_path_scalar = '/home/hail/Distribution_RLHF/summarize_from_feedback_details/eval_tables/reward_basic__1__1752221539/scalar_logs/rejected_rewards_scalar_step_validation_test.csv'
preferred_path_quantile = '/home/hail/Distribution_RLHF/summarize_from_feedback_details/eval_tables/reward_qr__1__1752132064/quantile_logs/preferred_rewards_quantile_step_validation_test.csv'
rejected_path_quantile = '/home/hail/Distribution_RLHF/summarize_from_feedback_details/eval_tables/reward_qr__1__1752132064/quantile_logs/rejected_rewards_quantile_step_validation_test.csv'

# === 데이터 불러오기 ===
preferred_scalar = pd.read_csv(preferred_path_scalar, skiprows=1, header=None).iloc[:, 0].astype(float)
rejected_scalar = pd.read_csv(rejected_path_scalar, skiprows=1, header=None).iloc[:, 0].astype(float)

preferred_quantile_df = pd.read_csv(preferred_path_quantile)
rejected_quantile_df = pd.read_csv(rejected_path_quantile)

# === Quantile 평균으로 Scalar화 ===
preferred_quantile = preferred_quantile_df.mean(axis=1).astype(float)
rejected_quantile = rejected_quantile_df.mean(axis=1).astype(float)

# === 히스토그램 데이터 수집 ===
bins = 10000
 
# Preferred Scalar 기준으로 bin_edges 고정
counts_pref_scalar, bin_edges = np.histogram(preferred_scalar, bins=bins)
prob_pref_scalar = counts_pref_scalar / counts_pref_scalar.sum()

# Rejected Scalar
counts_rej_scalar, _ = np.histogram(rejected_scalar, bins=bin_edges)
prob_rej_scalar = counts_rej_scalar / counts_rej_scalar.sum()

# Preferred Quantile
counts_pref_quantile, _ = np.histogram(preferred_quantile, bins=bin_edges)
prob_pref_quantile = counts_pref_quantile / counts_pref_quantile.sum()

# Rejected Quantile
counts_rej_quantile, _ = np.histogram(rejected_quantile, bins=bin_edges)
prob_rej_quantile = counts_rej_quantile / counts_rej_quantile.sum()

# === x축: bin 중심 구하기 ===
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 고정 막대 폭 설정
fixed_bar_width = 0.2

# === 확률 히스토그램 그리기 ===
plt.figure(figsize=(12, 7))

# Scalar pairs - 파란색 계열
plt.bar(bin_centers, prob_pref_scalar, width=fixed_bar_width, color='blue', alpha=0.8, label='Preferred (Scalar)')
plt.bar(bin_centers, prob_rej_scalar, width=fixed_bar_width, color='blue', alpha=0.3, label='Rejected (Scalar)')

# Quantile pairs - 주황색 계열
plt.bar(bin_centers, prob_pref_quantile, width=fixed_bar_width, color='orange', alpha=0.8, label='Preferred (Quantile)')
plt.bar(bin_centers, prob_rej_quantile, width=fixed_bar_width, color='orange', alpha=0.3, label='Rejected (Quantile)')

plt.xlim(0.0, 10.0)
plt.xlabel('Reward Value', fontsize=18)
plt.ylabel('Probability', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)

# 저장
save_path = os.path.join(save_dir, 'all_fixedwidth_histogram_10000.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print("Bin start:", bin_edges[0], "Bin end:", bin_edges[-1])
print("Fixed bar width:", fixed_bar_width)
print(f"확률 히스토그램 저장 완료: {save_path}")

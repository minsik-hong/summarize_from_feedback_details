import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 저장 폴더
save_dir = "./"
os.makedirs(save_dir, exist_ok=True)

# 데이터 로드
preferred_path_scalar = '/home/hail/Distribution_RLHF/summarize_from_feedback_details/eval_tables/reward_basic__1__1752221539/scalar_logs/preferred_rewards_scalar_step_validation_test.csv'
rejected_path_scalar = '/home/hail/Distribution_RLHF/summarize_from_feedback_details/eval_tables/reward_basic__1__1752221539/scalar_logs/rejected_rewards_scalar_step_validation_test.csv'
preferred_path_quantile = '/home/hail/Distribution_RLHF/summarize_from_feedback_details/eval_tables/reward_qr__1__1752132064/quantile_logs/preferred_rewards_quantile_step_validation_test.csv'
rejected_path_quantile = '/home/hail/Distribution_RLHF/summarize_from_feedback_details/eval_tables/reward_qr__1__1752132064/quantile_logs/rejected_rewards_quantile_step_validation_test.csv'

preferred_scalar = pd.read_csv(preferred_path_scalar, skiprows=1, header=None).iloc[:, 0].astype(float)
rejected_scalar = pd.read_csv(rejected_path_scalar, skiprows=1, header=None).iloc[:, 0].astype(float)

preferred_quantile_df = pd.read_csv(preferred_path_quantile)
rejected_quantile_df = pd.read_csv(rejected_path_quantile)

preferred_quantile_q0 = preferred_quantile_df.iloc[:, 4].astype(float)
rejected_quantile_q0 = rejected_quantile_df.iloc[:, 4].astype(float)

preferred_quantile_q9 = preferred_quantile_df.iloc[:, 9].astype(float)
rejected_quantile_q9 = rejected_quantile_df.iloc[:, 9].astype(float)


# KDE plot
plt.figure(figsize=(12, 7))

# sns.kdeplot(preferred_scalar, bw_adjust=0.5, color='blue', label='Preferred (Scalar)')
# sns.kdeplot(rejected_scalar, bw_adjust=0.5, color='blue', linestyle='--', label='Rejected (Scalar)')
sns.kdeplot(preferred_quantile_q0, bw_adjust=0.5, color='orange', label='Preferred (Quantile4)')
sns.kdeplot(rejected_quantile_q0, bw_adjust=0.5, color='orange', linestyle='--', label='Rejected (Quantile4)')
sns.kdeplot(preferred_quantile_q9, bw_adjust=0.5, color='purple', label='Preferred (Quantile9)')
sns.kdeplot(rejected_quantile_q9, bw_adjust=0.5, color='purple', linestyle='--', label='Rejected (Quantile9)')

plt.xlim(- 10.0, 12.0)
plt.xlabel('Reward Value', fontsize=18)
plt.ylabel('Density (KDE)', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=9, loc='upper left')
plt.grid(True)

save_path = os.path.join(save_dir, 'kde_plot_scaclar&q4&9.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"KDE 그래프 저장 완료: {save_path}")


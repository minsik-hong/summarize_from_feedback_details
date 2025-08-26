import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 파일 불러오기
df = pd.read_csv('/home/hail/Distribution_RLHF/summarize_from_feedback_details/eval_tables/reward_qr__1__1752132064/quantile_logs/rejected_rewards_quantile_step_validation_test.csv')

q0 = df['q0'].values
q9 = df['q9'].values

plt.figure(figsize=(8, 5))

# q0 히스토그램 + KDE
sns.histplot(q0, color='blue', label='q0', stat='density', kde=True, element='step', fill=True, alpha=0.3)

# q9 히스토그램 + KDE
sns.histplot(q9, color='red', label='q9', stat='density', kde=True, element='step', fill=True, alpha=0.3)

plt.title('q0 vs q9: Histogram + KDE')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('q0_q9_combined_hist_kde.png', dpi=300)
plt.show()
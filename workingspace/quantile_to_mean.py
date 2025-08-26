import pandas as pd
preferred_path_quantile = '/home/hail/Distribution_RLHF/summarize_from_feedback_details/eval_tables/reward_qr__1__1752132064/quantile_logs/preferred_rewards_quantile_step_validation_test.csv'

# 파일 경로
preferred_input = '/home/hail/Distribution_RLHF/summarize_from_feedback_details/eval_tables/reward_qr__1__1752132064/quantile_logs/preferred_rewards_quantile_step_validation_test.csv'
rejected_input = '/home/hail/Distribution_RLHF/summarize_from_feedback_details/eval_tables/reward_qr__1__1752132064/quantile_logs/rejected_rewards_quantile_step_validation_test.csv'

preferred_output = 'preferred_quantile_mean.csv'
rejected_output = 'rejected_quantile_mean.csv'

# 데이터 불러오기 (첫 줄은 header, 두 번째 줄부터 사용)
preferred_df = pd.read_csv(preferred_input).iloc[1:]
rejected_df = pd.read_csv(rejected_input).iloc[1:]

# 평균 계산
preferred_mean = preferred_df.mean(axis=1)
rejected_mean = rejected_df.mean(axis=1)

# CSV 저장
preferred_mean.to_csv(preferred_output, index=False, header=['mean'])
rejected_mean.to_csv(rejected_output, index=False, header=['mean'])

print(f"Preferred 평균 저장 완료: {preferred_output}")
print(f"Rejected 평균 저장 완료: {rejected_output}")


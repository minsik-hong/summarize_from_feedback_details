mkdir -p logs  # 로그 디렉토리 없으면 생성

SEED=1
if [ -z "$MODEL" ]; then
    # MODEL=EleutherAI/pythia-6.9b-deduped
    # MODEL=EleutherAI/pythia-2.8b-deduped
    # MODEL=EleutherAI/pythia-1b-deduped
    # MODEL=EleutherAI/pythia-410m-deduped
    MODEL="openai-community/gpt2-medium"
fi
LR=3e-6
REWARD_MODEL_PATH=models/$MODEL/reward_model_qr_gpt2_$SEED
SFT_MODEL_PATH=models/$MODEL/sft_model_gpt2_$SEED
POLICY_MODEL_PATH=models/$MODEL/policy_model_gpt2_$SEED

# 로그 이름 자동 설정
NOW=$(date +"%Y%m%d_%H%M%S")
MODEL_NAME_SAFE=${MODEL//\//_}
LOGFILE_SFT="logs/sft_${MODEL_NAME_SAFE}_seed${SEED}_${NOW}.txt"
LOGFILE_RM="logs/reward_basic_${MODEL_NAME_SAFE}_seed${SEED}_${NOW}.txt"
LOGFILE_PPO="logs/ppo_basic_${MODEL_NAME_SAFE}_seed${SEED}_${NOW}.txt"

# vary the following parameters to fit your GPU memory
local_rollout_forward_batch_size=2 # smaller fits better on GPU
gradient_accumulation_steps=128 # bigger fits better on GPU, 기존 64로 설정되어 있음
local_micro_batch_size=4 # smaller fits better on GPU
local_eval_batch_size=1 # smaller fits better on GPU

# 1. you want to make sure gradient_accumulation_steps * local_micro_batch_size = 64
# so you have the same hyperparameters as the paper
# 2. if you are running on a single GPU, you want to make sure 
# gradient_accumulation_steps * local_micro_batch_size = 512 to have the same hyperparameters

#CUDA_VISIBLE_DEVICES=0 poetry run accelerate launch --config_file deepspeed.yaml \
#    --main_process_port=29512 \
#    summarize_from_feedback_details/sft_gpt2_newData.py \
#    --base_model=$MODEL \
#    --lr=$LR \
#    --deepspeed \
#    --track \
#    --output_dir=$SFT_MODEL_PATH \
#    --no-push-to-hub \
#    --run_eval \
#    --seed=$SEED 2>&1 | tee "$LOGFILE_SFT"

CUDA_VISIBLE_DEVICES=0 poetry run accelerate launch --config_file deepspeed.yaml \
     --main_process_port=29513 \
     summarize_from_feedback_details/reward_qr_gpt2_newData_constraint.py \
     --base_model=$MODEL \
     --sft_model_path=$SFT_MODEL_PATH \
     --lr=$LR \
     --deepspeed \
     --run_eval \
     --track \
     --output_dir=$REWARD_MODEL_PATH \
     --no-push-to-hub \
     --local_eval_batch_size=$local_eval_batch_size \
     --seed=$SEED 2>&1 | tee "$LOGFILE_RM"

# CUDA_VISIBLE_DEVICES=2 poetry run accelerate launch --config_file deepspeed.yaml \
#    --main_process_port=29513 \
#    summarize_from_feedback_details/ppo_basic_gpt2_newData.py \
#    --local_rollout_forward_batch_size=$local_rollout_forward_batch_size \
#    --gradient_accumulation_steps=$gradient_accumulation_steps \
#    --local_micro_batch_size=$local_micro_batch_size \
#    --base_model=$MODEL \
#    --sft_model_path=$SFT_MODEL_PATH \
#    --reward_model_path=$REWARD_MODEL_PATH \
#    --lr=$LR \
#    --deepspeed \
#    --run_eval \
#    --track \
#    --output_dir=$POLICY_MODEL_PATH \
#    --no-push-to-hub \
#    --seed=$SEED 2>&1 | tee "$LOGFILE_PPO"
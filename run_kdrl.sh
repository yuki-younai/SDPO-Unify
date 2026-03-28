#!/bin/bash
set -x

# =============================================================================
# KDRL: Knowledge Distillation Reinforced Learning
# 论文最优配置: L_KDRL = L_RL(GRPO) + β(t) * L_KD(k2)
#   - KL 估计器: k2 (梯度无偏)
#   - β 调度: linear_decay (βinit=5e-3, δ=5e-5, βmin=1e-3)
#   - 屏蔽模式: response (正确回答不做 KD)
#   - RL 损失: GRPO
# =============================================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3
# Count number of GPUs separated by commas
export NUM_GPU=$(expr length "$CUDA_VISIBLE_DEVICES" / 2 + 1)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/lib')")
# WandB API Key Login
export WANDB_API_KEY="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"
wandb login --relogin $WANDB_API_KEY
export swanlab_api_key=FpaObNsPY6wSpfCM0Cu60
swanlab login -k $swanlab_api_key
# ray stop 
ray stop --force
# 为了保险，手动杀掉所有相关的 python 和 ray 进程（注意不要误杀其他人的任务）
pkill -f ray

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export USER=${USER:-$(whoami)}

ulimit -c 0

# =============================================================================
# CONFIGURATION
# =============================================================================
# Dataset and Model
DATA_PATH="datasets/sciknoweval/chemistry"
TASKS="chemistry"
MODEL_PATH="/data/szs/share/Qwen2.5-7B-Instruct"
CONFIG_NAME="kdrl"

# Hyperparameters (通用)
TRAIN_BATCH_SIZE=32
PPO_MINI_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
LR=1e-5
LR_WARMUP_STEPS=10
VAL_ROLLOUT_N=16

# SDPO / Self-Distillation 参数
ALPHA=0.5
DISTILLATION_TOPK=100
DONTS_REPROMPT_ON_SELF_SUCCESS=True

# KDRL 论文最优参数
KL_ESTIMATOR="k2"                    # k2 (梯度无偏) | k3 (低方差) | topk (原SDPO)
KDRL_RL_LOSS_MODE="grpo"             # grpo | vanilla | gpg | reinforce
KDRL_BETA_SCHEDULE="linear_decay"    # none | constant | linear_decay
KDRL_BETA_INIT=5e-3                  # 线性衰减初始 β
KDRL_BETA_MIN=1e-3                   # 线性衰减最小 β
KDRL_BETA_DELTA=5e-5                 # 每步衰减 δ
KDRL_BETA=2e-3                       # 恒定模式下的 β (仅 constant 模式使用)
KDRL_MASKING_MODE="response"         # none | response | group

# Experiment Name Construction
MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
SUFFIX=${1:-"single_run"}
EXP_NAME="KDRL-${MODEL_NAME}-${TASKS}-${KL_ESTIMATOR}-${KDRL_BETA_SCHEDULE}-mask_${KDRL_MASKING_MODE}-${SUFFIX}"
model_save_dir="/data/szs/share/gwy/SDPO-RL"
HDFS_CHECKPOINT_PATH=$model_save_dir/$EXP_NAME
# Export environment variables required by user.yaml to avoid InterpolationResolutionError
export TASK=$DATA_PATH
export EXPERIMENT=$EXP_NAME

# =============================================================================
# EXECUTION
# =============================================================================
echo "================================================================"
echo "Starting KDRL Training (Knowledge Distillation Reinforced Learning)"
echo "================================================================"
echo "Experiment: $EXP_NAME"
echo "Data:       $DATA_PATH"
echo "Model:      $MODEL_PATH"
echo "KL Estimator:     $KL_ESTIMATOR"
echo "Beta Schedule:    $KDRL_BETA_SCHEDULE"
echo "Beta Init/Min/δ:  $KDRL_BETA_INIT / $KDRL_BETA_MIN / $KDRL_BETA_DELTA"
echo "Masking Mode:     $KDRL_MASKING_MODE"
echo "RL Loss Mode:     $KDRL_RL_LOSS_MODE"
echo "================================================================"

python -m verl.trainer.main_ppo \
    --config-name $CONFIG_NAME \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.train_files="['$PROJECT_ROOT/$DATA_PATH/train.parquet']" \
    data.val_files="['$PROJECT_ROOT/$DATA_PATH/test.parquet']" \
    custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py \
    trainer.group_name=$WANDB_PROJECT \
    actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS \
    actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
    actor_rollout_ref.actor.self_distillation.distillation_topk=$DISTILLATION_TOPK \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=$DONTS_REPROMPT_ON_SELF_SUCCESS \
    actor_rollout_ref.actor.self_distillation.kl_estimator=$KL_ESTIMATOR \
    actor_rollout_ref.actor.self_distillation.kdrl_beta=$KDRL_BETA \
    actor_rollout_ref.actor.self_distillation.kdrl_beta_schedule=$KDRL_BETA_SCHEDULE \
    actor_rollout_ref.actor.self_distillation.kdrl_beta_init=$KDRL_BETA_INIT \
    actor_rollout_ref.actor.self_distillation.kdrl_beta_min=$KDRL_BETA_MIN \
    actor_rollout_ref.actor.self_distillation.kdrl_beta_delta=$KDRL_BETA_DELTA \
    actor_rollout_ref.actor.self_distillation.kdrl_masking_mode=$KDRL_MASKING_MODE \
    actor_rollout_ref.actor.policy_loss.kdrl_rl_loss_mode=$KDRL_RL_LOSS_MODE \
    algorithm.rollout_correction.rollout_is=token \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_ROLLOUT_N \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    trainer.default_local_dir=$HDFS_CHECKPOINT_PATH \
    trainer.n_gpus_per_node=${NUM_GPU} \
    trainer.project_name=sdpo-rl \
    trainer.experiment_name=${EXP_NAME} \
    trainer.save_freq=1000 \
    trainer.nnodes=1 \
    trainer.logger=['console','swanlab'] >> training_${EXP_NAME}_$(date +%Y%m%d%H%M%S).log 2>&1

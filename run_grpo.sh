#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3
# Count number of GPUs separated by commas
export NUM_GPU=$(expr length "$CUDA_VISIBLE_DEVICES" / 2 + 1)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/lib')")
# WandB API Key Login
export WANDB_API_KEY="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"
wandb login --relogin $WANDB_API_KEY

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export USER=${USER:-$(whoami)}

# Disable VLLM V1 because installed version (0.5.3) does not support it
# export VLLM_USE_V1=1
# export PYTHONBUFFERED=1
# export VLLM_USE_MODELSCOPE=0
# export VLLM_WORKER_MULTIPROC_METHOD=spawn
# export VLLM_ATTENTION_BACKEND=XFORMERS
# export PYTHONBUFFERED=1
# export RAY_DEBUG=1
ulimit -c 0

# =============================================================================
# CONFIGURATION
# =============================================================================
# Dataset and Model
DATA_PATH="datasets/sciknoweval/chemistry"
MODEL_PATH="/data/szs/share/Qwen3-8B"
CONFIG_NAME="baseline_grpo"
TASKS="chemistry"
# Hyperparameters (Referencing run_local_grpo.sh)
TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=8
ROLLOUT_BATCH_SIZE=8
LR=1e-5
LR_WARMUP_STEPS=10
VAL_ROLLOUT_N=16

# Experiment Name Construction
MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
SUFFIX=${1:-"single_run"}
EXP_NAME="SDPO-${MODEL_NAME}-${TASKS}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-alpha${ALPHA}-${SUFFIX}"
model_save_dir="/data/szs/share/gwy/SDPO-RL"
HDFS_CHECKPOINT_PATH=$model_save_dir/$EXP_NAME
# Export environment variables required by user.yaml to avoid InterpolationResolutionError
export TASK=$DATA_PATH
export EXPERIMENT=$EXP_NAME

# =============================================================================
# EXECUTION
# =============================================================================
echo "----------------------------------------------------------------"
echo "Starting GRPO Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "----------------------------------------------------------------"

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
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_ROLLOUT_N \
    algorithm.rollout_correction.rollout_is=token \
    trainer.default_local_dir=${model_save_path} \
    trainer.n_gpus_per_node=${NUM_GPU} \
    trainer.project_name=sdpo-rl \
    trainer.experiment_name=${EXP_NAME} \
    trainer.default_local_dir=$HDFS_CHECKPOINT_PATH \
    trainer.save_freq=1000 \
    trainer.nnodes=1 \
    trainer.logger=['console','wandb']  >> training_${EXP_NAME}_$(date +%Y%m%d%H%M%S).log 2>&1

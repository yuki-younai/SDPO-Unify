export CUDA_VISIBLE_DEVICES=0,1,2,3
# Count number of GPUs separated by commas
export NUM_GPU=$(expr length "$CUDA_VISIBLE_DEVICES" / 2 + 1)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/lib')")
# WandB API Key Login
export WANDB_API_KEY="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"
wandb login --relogin $WANDB_API_KEY
export swanlab_api_key=FpaObNsPY6wSpfCM0Cu60
swanlab login -k $swanlab_api_key

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
#"datasets/tooluse"
#datasets/sciknoweval/chemistry
DATA_PATH="datasets/sciknoweval/chemistry"
TASKS="chemistry"
#/data/szs/share/Qwen2.5-7B-Instruct
#/data/szs/250010072/public/models/Qwen2.5-3B-Instruct
#/data/szs/share/Qwen3-4B
#/data/szs/share/Qwen3-1.7B
#/data/szs/share/Qwen3-8B
MODEL_PATH="/data/szs/share/Qwen2.5-7B-Instruct"
CONFIG_NAME="sdpo"

# Hyperparameters
TRAIN_BATCH_SIZE=32
PPO_MINI_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
LR=1e-5
LAMBDA=0.0
CLIP_ADV_HIGH=null
DONTS_REPROMPT_ON_SELF_SUCCESS=True
ALPHA=0.5
DISTILLATION_TOPK=100
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
echo "Starting SDPO Training"
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
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.self_distillation.distillation_topk=$DISTILLATION_TOPK \
    algorithm.rollout_correction.rollout_is=token \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
    actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_ROLLOUT_N \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    trainer.default_local_dir=${model_save_path} \
    trainer.n_gpus_per_node=${NUM_GPU} \
    trainer.project_name=sdpo-rl \
    trainer.experiment_name=${EXP_NAME} \
    trainer.default_local_dir=$HDFS_CHECKPOINT_PATH \
    trainer.save_freq=1000 \
    trainer.nnodes=1 \
    trainer.logger=['console','swanlab'] >> training_${EXP_NAME}_$(date +%Y%m%d%H%M%S).log 2>&1

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_GPU=$(expr length "$CUDA_VISIBLE_DEVICES" / 2 + 1)
export CUDA_DEVICE_MAX_CONNECTIONS=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/lib')")
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export USER=${USER:-$(whoami)}
ulimit -c 0

export swanlab_api_key=FpaObNsPY6wSpfCM0Cu60
swanlab login -k $swanlab_api_key
# ray stop 
ray stop --force
# 为了保险，手动杀掉所有相关的 python 和 ray 进程（注意不要误杀其他人的任务）
pkill -f ray

current_time=$(date +"%Y%m%d_%H%M%S")

DATA_PATH=datasets/sciknoweval/chemistry
MODEL_PATH=/data/szs/share/Qwen3-4B
TASKS=chemistry

TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
VAL_ROLLOUT_N=16
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=2048
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="GRPO-${MODEL_NAME}-${TASKS}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr1e-5-${current_time}"

RUN_NAME=${EXP_NAME}
export TASK="$TASKS"
export EXPERIMENT="$RUN_NAME"
PROJECT_NAME=sdpo-rl
WANDB_PROJECT=sdpo-rl
OUTPUT_ROOT=/data/szs/share/gwy/SDPO-RL
OUTPUT_PATH=$OUTPUT_ROOT/$RUN_NAME
HDFS_LOG_PATH=$OUTPUT_PATH/log
HDFS_CHECKPOINT_PATH=$OUTPUT_PATH/model_output
mkdir -p "$OUTPUT_PATH" "$HDFS_LOG_PATH" "$HDFS_CHECKPOINT_PATH" 
SCRIPT_NAME=$(basename "$0")
DESTINATION_PATH="$OUTPUT_PATH/$SCRIPT_NAME"
cp "$0" "$DESTINATION_PATH"

python -m verl.trainer.main_ppo \
    --config-name baseline_grpo \
    algorithm.adv_estimator=grpo \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.train_files="['$PROJECT_ROOT/$DATA_PATH/train.parquet']" \
    data.val_files="['$PROJECT_ROOT/$DATA_PATH/test.parquet']" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    custom_reward_function.path="$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py" \
    trainer.group_name="$WANDB_PROJECT" \
    actor_rollout_ref.rollout.n="$ROLLOUT_BATCH_SIZE" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model'] \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.val_kwargs.n="$VAL_ROLLOUT_N" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=token \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node="$NUM_GPU" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$RUN_NAME" \
    trainer.default_local_dir="$HDFS_CHECKPOINT_PATH" \
    trainer.save_freq=1000 \
    trainer.test_freq=100 \
    trainer.total_epochs=15 \
    trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.logger="['console','swanlab']" > "${EXP_NAME}.log" 2>&1








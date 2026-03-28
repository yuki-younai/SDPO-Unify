export CUDA_VISIBLE_DEVICES=6,7
export NUM_GPUS=$(expr length "$CUDA_VISIBLE_DEVICES" / 2 + 1)

set -x 

current_time=$(date +"%Y%m%d_%H%M%S")
export NAME=dapo
export RUN_NAME=${NAME}_${current_time}
export PROJECT_NAME=${NAME}
export OUTPUT_PATH=model_output/$RUN_NAME
export TMPDIR=/data3/gwy/tmp
mkdir -p $TMPDIR
export HDFS_LOG_PATH=$OUTPUT_PATH/log
export HDFS_CHECKPOINT_PATH=$OUTPUT_PATH/model_output
export TENSORBOARD_DIR=$OUTPUT_PATH/tensorboard_log
if [ ! -d "$OUTPUT_PATH" ]; then
    mkdir -p "$OUTPUT_PATH"
    echo "目录 $OUTPUT_PATH 已创建"
else
    echo "目录 $OUTPUT_PATH 已存在"
fi

SCRIPT_NAME=$(basename "$0")
DESTINATION_PATH="$OUTPUT_PATH/$SCRIPT_NAME"
cp "$0" "$DESTINATION_PATH"

### Separated Clip Epsilons (-> Clip-Higher)
clip_ratio_low=0.2
clip_ratio_high=0.28

### Dynamic Sampling (with Group Filtering)
enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10

### Flexible Loss Aggregation Mode (-> Token-level Loss)
loss_agg_mode="token-mean"

### Overlong Reward Shaping
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 2))
overlong_penalty_factor=1.0

fsdp_size=-1
micro_batch_size=4
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 6))
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * $micro_batch_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * $micro_batch_size))
#/data3/public/model/Qwen2.5-3B-Instruct
#/data3/public/model/Qwen2.5-3B
export MODEL_PATH=/data3/gwy/Agent-RL/model_output/Qwen2.5-1.5B

#'swanlab'
python3 -m recipe.dapo.main_dapo --config-name=dapo_trainer \
    algorithm.adv_estimator=grpo \
    data.train_files=data/simplerl-verl/train.parquet \
    data.val_files=data/simplerl-verl/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.loss_agg_mode=$loss_agg_mode \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.n=5 \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    ++algorithm.filter_groups.enable=${enable_filter_groups} \
    ++algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    ++algorithm.filter_groups.metric=${filter_groups_metric} \
    custom_reward_function.path=examples/reward_function/math_reward.py \
    custom_reward_function.name=compute_score_math \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    actor_rollout_ref.actor.checkpoint.save_contents=['model'] \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.val_before_train=False\
    trainer.test_freq=20 \
    trainer.save_freq=200 \
    trainer.default_local_dir=$HDFS_CHECKPOINT_PATH \
    trainer.total_epochs=5
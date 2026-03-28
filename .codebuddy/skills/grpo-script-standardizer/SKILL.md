---
name: grpo-script-standardizer
description: Standardizes run_grpo.sh-style training scripts (structure, naming, params, logging, config checks). Invoke when creating/refactoring RL training launch scripts.
---

# GRPO Training Script Standardizer

## 适用场景

当用户提出以下需求时使用本技能：

- 模仿 `example.sh` 整理训练脚本
- 重构 `run_grpo.sh` 的结构与命名
- 新增训练参数并检查是否被当前配置支持
- 统一日志命名、输出目录与实验命名规则
- 让脚本更易维护（减少二次赋值、分层清晰）

## 目标

将训练脚本整理为稳定、可复现、便于 AI 持续维护的标准形态，且不改变核心训练语义。

## 强制规范

### 1) 整体结构分层（固定顺序）

1. 环境初始化层：`CUDA_VISIBLE_DEVICES`、`NUM_GPU`、`PROJECT_ROOT`、`PYTHONPATH`、`ulimit`
2. 关键训练变量层：数据、模型、batch、长度、rollout 数
3. 实验命名层：`MODEL_NAME`、`EXP_NAME`、`current_time`、`RUN_NAME`
4. 输出目录层：`OUTPUT_PATH`、`HDFS_LOG_PATH`、`HDFS_CHECKPOINT_PATH`、`TENSORBOARD_DIR`、脚本快照
5. 训练入口层：单条 `python -m verl.trainer.main_ppo` 命令
6. 日志落盘层：`> "${EXP_NAME}.log" 2>&1`

### 2) 命名规则

- `MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')`
- `EXP_NAME` 必须带时间戳：  
  `GRPO-${MODEL_NAME}-${TASKS}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr1e-5-${current_time}`
- `RUN_NAME=${EXP_NAME}`
- `trainer.experiment_name="$RUN_NAME"`
- 日志名用 `EXP_NAME`，目录名用 `RUN_NAME`

### 3) 参数组织规则

- 关键参数变量化：`DATA_PATH`、`MODEL_PATH`、`TRAIN_BATCH_SIZE`、`MINI_BATCH_SIZE`、`ROLLOUT_BATCH_SIZE`、`VAL_ROLLOUT_N`、`MAX_PROMPT_LENGTH`、`MAX_RESPONSE_LENGTH`
- 稳定参数直接写死：如 `use_kl_loss`、`kl_loss_coef`、`entropy_coeff`、`warmup`、固定 logger 与部分固定频率
- 禁止“参数二次赋值泛滥”：不要把几乎所有参数都先定义变量再回填命令行

### 4) 配置兼容性规则

新增参数前必须在 `verl/trainer/config` 中确认键路径存在。优先使用已验证类别：

- `actor_rollout_ref.model.*`：`use_remove_padding`、`enable_gradient_checkpointing`
- `actor_rollout_ref.actor.*`：`use_dynamic_bsz`、`ppo_max_token_len_per_gpu`、`grad_clip`、`checkpoint.save_contents`
- `actor_rollout_ref.rollout.*`：`temperature`、`top_p`、`top_k`、`max_model_len`、`enable_chunked_prefill`、`log_prob_use_dynamic_bsz`、`log_prob_max_token_len_per_gpu`
- `actor_rollout_ref.ref.*`：`log_prob_use_dynamic_bsz`、`log_prob_max_token_len_per_gpu`
- `trainer.*`：`val_before_train`

### 5) 语义保护规则

- 无明确需求时，不改默认任务、模型、batch、学习率、训练轮数等实验语义
- 保留脚本快照复制逻辑（`cp "$0" "$DESTINATION_PATH"`）
- 训练命令保持单入口，避免拆成多段不可追踪命令

## 模板示例

### A. 命名与目录块模板（核心结构）

```bash
current_time=$(date +"%Y%m%d_%H%M%S")

DATA_PATH=datasets/sciknoweval/chemistry
MODEL_PATH=/data/szs/share/Qwen3-4B
TASKS=chemistry

TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
VAL_ROLLOUT_N=16
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=1024
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="GRPO-${MODEL_NAME}-${TASKS}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr1e-5-${current_time}"

RUN_NAME=${EXP_NAME}
PROJECT_NAME=sdpo-rl
WANDB_PROJECT=sdpo-rl
OUTPUT_ROOT=/data/szs/share/gwy/SDPO-RL
OUTPUT_PATH=$OUTPUT_ROOT/$RUN_NAME
HDFS_LOG_PATH=$OUTPUT_PATH/log
HDFS_CHECKPOINT_PATH=$OUTPUT_PATH/model_output
TENSORBOARD_DIR=$OUTPUT_PATH/tensorboard_log
mkdir -p "$OUTPUT_PATH" "$HDFS_LOG_PATH" "$HDFS_CHECKPOINT_PATH" "$TENSORBOARD_DIR"
SCRIPT_NAME=$(basename "$0")
DESTINATION_PATH="$OUTPUT_PATH/$SCRIPT_NAME"
cp "$0" "$DESTINATION_PATH"
```

### B. 训练入口与日志模板

```bash
python -m verl.trainer.main_ppo \
  --config-name baseline_grpo \
  algorithm.adv_estimator=grpo \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH_SIZE" \
  actor_rollout_ref.rollout.n="$ROLLOUT_BATCH_SIZE" \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$RUN_NAME" \
  trainer.default_local_dir="$HDFS_CHECKPOINT_PATH" \
  trainer.logger="['console','swanlab']" > "${EXP_NAME}.log" 2>&1
```

## 正反例

### 推荐

- `EXP_NAME` 含时间戳，`RUN_NAME` 与 `EXP_NAME` 一致
- 日志名用 `EXP_NAME`，输出目录用 `RUN_NAME`
- 参数分层清晰，关键变量少而精

### 不推荐

- 使用 `>>` 追加日志导致同名日志内容混杂
- 所有参数全部变量化，出现大量冗余二次赋值
- 未校验配置键就直接添加新参数

## 执行步骤（给 AI）

1. 先识别并修复结构块（命名、目录、训练入口、日志）
2. 再整理参数层（关键参数变量化，其余写死）
3. 再做配置键存在性校验（`verl/trainer/config`）
4. 保持语义不变，仅做结构化和可维护性增强
5. 修改后执行 `bash -n run_grpo.sh` 做语法校验

## 完成标准

- 脚本结构满足六层顺序
- 命名/目录/日志规则一致
- 新增键均可在配置中找到
- 命令可读、可复现、语法通过

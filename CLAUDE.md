# 训练脚本规范（run_grpo.sh）

## 总体原则

- 以 `run_grpo.sh` 作为主配置来源，改造时不改变其核心训练语义与默认实验设定。
- 脚本结构模仿 `example.sh` 的工程化组织方式：环境初始化、实验命名、输出目录、参数分层、单一训练入口。
- 参数策略遵循“稳定参数写死、关键参数变量化”。
- 避免“参数二次赋值”写法：不建议先定义一批变量再逐个镜像赋给命令行同名项。

## 命名与输出规范

- `MODEL_NAME` 由 `MODEL_PATH` 转换得到：将 `/` 替换为 `-`，用于构造可读模型标识。
- `EXP_NAME` 使用固定结构且不带时间戳：  
  `GRPO-${MODEL_NAME}-${TASKS}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr1e-5`
- `RUN_NAME` 用于区分具体运行实例：`RUN_NAME=${EXP_NAME}_${current_time}`。
- 输出目录基于 `RUN_NAME`：`OUTPUT_PATH=$OUTPUT_ROOT/$RUN_NAME`，并包含 `log/model_output/tensorboard_log` 子目录。
- `trainer.experiment_name` 使用 `RUN_NAME`，保证平台侧每次运行唯一。
- 训练日志文件名使用 `EXP_NAME`，放在当前执行目录：`>> "${EXP_NAME}.log" 2>&1`。

## 脚本整体结构规范

- 训练脚本必须按固定分层组织，推荐顺序如下：
  1. 环境初始化层：`CUDA_VISIBLE_DEVICES`、`NUM_GPU`、`PROJECT_ROOT`、`PYTHONPATH`、`ulimit`
  2. 关键训练变量层：数据、模型、batch、长度、rollout 数
  3. 实验命名层：`MODEL_NAME`、`EXP_NAME`、`current_time`、`RUN_NAME`
  4. 输出目录层：`OUTPUT_PATH`、`HDFS_LOG_PATH`、`HDFS_CHECKPOINT_PATH`、`TENSORBOARD_DIR`、脚本快照
  5. 训练入口层：单条 `python -m verl.trainer.main_ppo` + 分层参数
  6. 日志落盘层：重定向到 `${EXP_NAME}.log`
- 其中“实验命名层 + 输出目录层”是必须保留的核心结构，不能省略。

### 实验命名与目录块模板（对应 run_grpo.sh 中关键结构）

```bash
MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="GRPO-${MODEL_NAME}-${TASKS}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr1e-5"

RUN_NAME=${EXP_NAME}_${current_time}
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

### 训练入口与日志模板

```bash
python -m verl.trainer.main_ppo \
  --config-name baseline_grpo \
  ... \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$RUN_NAME" \
  trainer.default_local_dir="$HDFS_CHECKPOINT_PATH" \
  trainer.logger="['console','swanlab']" >> "${EXP_NAME}.log" 2>&1
```

## AI 整理脚本执行指引（给后续模型）

- 第一步先锁定“关键结构块”是否存在：命名块、目录块、训练入口块、日志块。
- 第二步再做参数层整理：先区分“变量化参数”与“写死参数”，避免全量变量化。
- 第三步补充参数时，先查 `verl/trainer/config`，确认键路径存在后再写入。
- 第四步保持实验语义不变：若无明确要求，不改默认模型、默认任务、默认 batch、默认学习率。
- 第五步修改后做基础校验：至少保证脚本语法正确（`bash -n run_grpo.sh`）。

## 正反例

- 推荐：`EXP_NAME` 稳定、`RUN_NAME` 带时间戳、日志名用 `EXP_NAME`、目录名用 `RUN_NAME`。
- 不推荐：把日志名改成时间戳随机名导致同配置不可聚合；或把所有参数都改成变量造成维护冗余。

## 参数组织规范

- 变量化（保留为脚本变量）的参数：任务路径、模型路径、核心 batch、rollout 数、最大输入输出长度等主要调参项。
- 写死（直接写在命令行）的参数：一般不变或默认稳定的项，如固定学习率、warmup、logger、测试/保存频率、固定布尔开关等。
- 采用分层组织：`data`、`actor_rollout_ref.model`、`actor_rollout_ref.actor`、`actor_rollout_ref.rollout`、`actor_rollout_ref.ref`、`algorithm`、`trainer`。

## 示例对齐规范

- 需要参考 `example.sh` 的可观测与工程组织，但不照搬其任务配置与模型配置。
- 需要参考 `examples/grpo_trainer/run_deepseek7b_llm_math_megatron.sh` 补充参数时，必须以当前仓库 `verl/trainer/config` 的可用键为准。
- 新增参数前需确认键存在，避免写入当前版本不支持的配置路径。

## 配置兼容性检查清单

- 新增键必须在 `verl/trainer/config` 下可检索到，或可由默认配置链解析。
- 推荐优先使用已验证可用的键（如以下类别）：
  - `actor_rollout_ref.model.*`：`use_remove_padding`、`enable_gradient_checkpointing`
  - `actor_rollout_ref.actor.*`：`use_dynamic_bsz`、`ppo_max_token_len_per_gpu`、`grad_clip`、`checkpoint.save_contents`
  - `actor_rollout_ref.rollout.*`：`temperature`、`top_p`、`top_k`、`max_model_len`、`enable_chunked_prefill`、`log_prob_use_dynamic_bsz`、`log_prob_max_token_len_per_gpu`
  - `actor_rollout_ref.ref.*`：`log_prob_use_dynamic_bsz`、`log_prob_max_token_len_per_gpu`
  - `trainer.*`：`val_before_train`

## 执行与可复现规范

- 训练前保留环境初始化：`CUDA_VISIBLE_DEVICES`、`NUM_GPU`、`LD_LIBRARY_PATH`、`PROJECT_ROOT`、`PYTHONPATH`、`ulimit`。
- 运行时保留脚本快照：将当前脚本复制到实验输出目录，便于复现实验配置。
- 训练入口保持单条 `python -m verl.trainer.main_ppo` 命令，参数显式可读。

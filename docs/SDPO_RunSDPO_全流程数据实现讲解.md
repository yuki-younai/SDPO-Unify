# SDPO 从 run_sdpo.sh 出发：改动点详细实现说明（相对 verl 原生）

本文只展开 **SDPO 新增/改造** 的实现。  
原生 verl 能力（通用 Ray 启动、常规 Dataset 读取、标准 PPO 训练骨架）不重复解释。

## 1. 入口层：run_sdpo.sh 如何把训练切换到 SDPO 轨道

`run_sdpo.sh` 的关键不是“启动 Python”，而是覆盖了会改变训练语义的参数：

- `--config-name sdpo`：载入 SDPO 基线配置。
- `actor_rollout_ref.actor.policy_loss.loss_mode=sdpo`：让 actor 走自蒸馏分支，而非 vanilla PPO/GRPO 分支。
- `actor_rollout_ref.actor.self_distillation.alpha=$ALPHA`：设置蒸馏散度形态。
- `actor_rollout_ref.actor.self_distillation.distillation_topk=$DISTILLATION_TOPK`：使用 top-k 蒸馏近似。
- `algorithm.rollout_correction.rollout_is=token`：启用 token 级 rollout 修正权重。
- `actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True`：避免样本把“自己本轮成功答案”直接作为演示。

同时，`sdpo.yaml` 固化了默认策略：

- `loss_mode: sdpo`
- `self_distillation.is_clip: 2.0`
- `algorithm.adv_estimator: grpo`（注释里已写“disables critic”）
- `rollout_is_threshold: 2.0`

代码位置：

- `run_sdpo.sh` 第 65-90 行
- `verl/trainer/config/sdpo.yaml` 第 13-32 行

## 2. 训练器装配层：SDPO 对 worker 选择加了硬约束

`TaskRunner.add_actor_rollout_worker` 里有 SDPO 专属约束：

1. 若 SDPO 需要 ref policy，同时又打开 KL 参考策略共享，会直接报错：  
   `SDPO cannot share the reference policy with KL regularization.`
2. 若禁用了 legacy worker，也会报错：  
   `SDPO requires the legacy worker implementation to colocate the teacher.`

这说明 SDPO 依赖“teacher 与 actor/ref 的特定共置关系”，不是任意 worker 组合都可跑。

代码位置：

- `verl/trainer/main_ppo.py` 第 128-137 行

## 3. 数据侧第一处改造：reward 附带 feedback，供 reprompt 使用

SDPO 要把环境反馈塞进教师条件，所以 reward 阶段不只返回标量分数。

在 `NaiveRewardManager.__call__`：

1. `compute_score(...)` 可能返回 dict（包含 `score`、`feedback` 等）。
2. 当返回 dict 时，所有键值都会写入 `reward_extra_info[key].append(value)`。
3. 最终返回：
   - `reward_tensor`
   - `reward_extra_info`（其中可能含 `feedback`）

这就是后面 trainer 构造教师输入时 `reward_extra_infos_dict["feedback"]` 的来源。

代码位置：

- `verl/workers/reward_manager/naive.py` 第 86-121 行

## 4. trainer 核心改造：把“成功轨迹 + 反馈”变成教师输入

核心函数：`RayPPOTrainer._maybe_build_self_distillation_batch(...)`。

### 4.1 触发条件

仅当两者同时满足时执行：

- `self_distillation_cfg is not None`
- `loss_mode == "sdpo"`

否则直接 `return None`，不影响原生链路。

代码位置：

- `verl/trainer/ppo/ray_trainer.py` 第 678-681 行

### 4.2 反馈提取逻辑

`_collect_feedback(...)` 的行为很严格：

- 只有 `include_environment_feedback=True` 才会读 `reward_extra_infos_dict`。
- 只接受非空字符串 feedback（`strip()` 后非空）。
- 对齐 batch 大小，不足部分补 `None`。

代码位置：

- `verl/trainer/ppo/ray_trainer.py` 第 611-634 行

### 4.3 成功解筛选逻辑

`_collect_solutions_by_uid(...)`：

- 先用 `reward_tensor.sum(dim=-1)` 得到序列分数。
- 按 `uid` 分组。
- 分数 `>= success_reward_threshold` 的样本索引记为可演示成功轨迹。

后续 `_get_solution(...)` 支持：

- `dont_reprompt_on_self_success`：剔除当前样本自身索引。
- `remove_thinking_from_demonstration`：移除 `<think>...</think>` 内容。

代码位置：

- `verl/trainer/ppo/ray_trainer.py` 第 636-669 行

### 4.4 reprompt 拼接规则（最关键）

`_build_teacher_message(i)` 里按配置做三层决策：

1. 是否有成功解 `has_solution`。
2. 是否有环境反馈 `has_feedback`。
3. 若 `environment_feedback_only_without_solution=True`，当有解时忽略反馈。

然后拼装：

- `solution_section = solution_template.format(...)`
- `feedback_section = feedback_template.format(...)`
- `reprompt_text = reprompt_template.format(prompt, solution, feedback)`

若两者都没有，直接退化为原始 prompt。

代码位置：

- `verl/trainer/ppo/ray_trainer.py` 第 710-745 行

### 4.5 教师输入张量化与监督掩码

构造流程：

1. `tokenizer.apply_chat_template(..., max_length=self_distillation_cfg.max_reprompt_len, truncation=True, padding=True)`
2. `teacher_input_ids = concat(teacher_prompt_ids, responses)`
3. `teacher_attention_mask = concat(teacher_prompt_mask, response_mask)`
4. `teacher_position_ids = compute_position_id_with_mask(...)`
5. `self_distillation_mask`：仅当“有成功解或反馈被使用”才置 1

返回给 actor 的新增字段：

- `teacher_input_ids`
- `teacher_attention_mask`
- `teacher_position_ids`
- `self_distillation_mask`

并上报关键比率指标：

- `success_group_fraction`
- `success_sample_fraction`
- `feedback_available_fraction`
- `feedback_used_fraction`
- `reprompt_sample_fraction`

代码位置：

- `verl/trainer/ppo/ray_trainer.py` 第 748-796 行

## 5. actor 核心改造：SDPO 分支的前向、损失和更新门控

核心函数：`DataParallelPPOActor.update_policy(...)`。

### 5.1 入参校验与字段选择

当 `loss_mode=="sdpo"`，强制要求 batch 含四个键：

- `teacher_input_ids`
- `teacher_attention_mask`
- `teacher_position_ids`
- `self_distillation_mask`

否则直接断言失败。  
这确保 trainer 的 teacher batch 构造不会被静默绕过。

代码位置：

- `verl/workers/actor/dp_actor.py` 第 684-694、708-710 行

### 5.2 学生/教师双前向

每个 micro-batch：

1. 学生侧常规前向得到：
   - `log_probs`
   - （可选）`all_logps` 或 `topk_logps/topk_indices`
2. 教师侧使用 `teacher_*` 张量前向：
   - teacher 使用 `self.teacher_module or self.actor_module`
   - trust-region 模式要求 teacher 必须是独立模块

代码位置：

- `verl/workers/actor/dp_actor.py` 第 778-833 行

### 5.3 SDPO 损失调用参数

`compute_self_distillation_loss(...)` 会收到：

- 学生/教师 token log prob
- full-logit 或 top-k logits
- `self_distillation_mask`
- `old_log_probs`（用于 distillation IS）
- `rollout_is_weights`（用于 rollout 修正）

并额外记录：

- `self_distillation/empty_target_batch = (mask.sum()==0)`

代码位置：

- `verl/workers/actor/dp_actor.py` 第 833-849 行

### 5.4 更新提交与 teacher 同步门控

在你关注的尾段逻辑中：

1. `_optimizer_step()` 做梯度裁剪 + step。
2. 只有 `grad_norm` 有限时，`did_update=True`。
3. 全部 mini-batch 完成后，只有 `did_update` 为真才触发 `_update_teacher()`。

这避免 NaN/Inf 步导致 teacher 也被错误推进。

代码位置：

- `verl/workers/actor/dp_actor.py` 第 913-921 行

## 6. 损失函数核心改造：compute_self_distillation_loss 逐项拆解

核心函数：`verl/trainer/ppo/core_algos.py::compute_self_distillation_loss`。

### 6.1 监督样本选择

- 初始 `loss_mask = response_mask`
- 若有 `self_distillation_mask`，变为 `response_mask * self_distillation_mask[:, None]`

即“非目标样本完全不贡献蒸馏梯度”。

代码位置：

- `core_algos.py` 第 1102-1105 行

### 6.2 full-logit / top-k 蒸馏

当 `full_logit_distillation=True`：

- 若配置 `distillation_topk`：使用 top-k log probs。
  - `distillation_add_tail=True` 时，会额外构造 tail bucket（把 top-k 之外概率质量并成一个桶）。
  - 否则对 top-k 子分布重新归一化。
- 若未配 top-k：直接用全词表 logits。

代码位置：

- `core_algos.py` 第 1106-1137 行

### 6.3 alpha 对应的散度形态

- `alpha == 0.0`：forward KL（teacher 作为 target）
- `alpha == 1.0`：reverse KL
- `0<alpha<1`：通过 mixture 分布构造 generalized JSD

代码位置：

- `core_algos.py` 第 1138-1161 行

### 6.4 is_clip 的真实作用路径

`is_clip` 不是对 KL 值截断，而是对 **蒸馏权重比率** 截断：

1. 先算 `negative_approx_kl = (student_log_probs - old_log_probs).detach()`
2. 再 `ratio = exp(clamp(negative_approx_kl, -20, 20)).clamp(max=is_clip)`
3. 用该 `ratio` 乘 `per_token_loss`

这相当于“只让蒸馏损失按受控 IS 比率放大”，防止过大比率导致梯度爆炸。

代码位置：

- `core_algos.py` 第 1168-1177 行

### 6.5 与 rollout IS 的叠加

若 trainer 侧给了 `rollout_is_weights`，会继续乘到 `per_token_loss` 上：

- 先蒸馏 IS（`is_clip`）  
- 再 rollout 修正权重

最后统一走 `agg_loss(...)` 聚合。

代码位置：

- `core_algos.py` 第 1178-1187 行

## 7. 教师来源与更新：初始化 + 训练中演化

### 7.1 初始化阶段（worker 构建时）

在 `fsdp_workers.py`：

- `teacher_regularization == "trust-region"`：`self.actor.teacher_module = TrustRegionTeacher(ref, student, mix_coef)`
- 否则：`self.actor.teacher_module = self.ref_module_fsdp`

代码位置：

- `verl/workers/fsdp_workers.py` 第 894-905 行

### 7.2 训练阶段（每轮更新后）

在 `dp_actor._update_teacher()`：

- 仅 `loss_mode=="sdpo"` 且 `teacher_regularization=="ema"` 且 `teacher_update_rate>0` 才执行。
- 逐参数 EMA：
  - `teacher = (1-rate)*teacher + rate*student`

代码位置：

- `verl/workers/actor/dp_actor.py` 第 132-151 行

## 8. 字段级数据流（只列 SDPO 新增链路）

1. reward 阶段写入 `reward_extra_info["feedback"]`  
2. trainer 读取 feedback + 成功样本，构造 reprompt  
3. trainer 输出 `teacher_input_ids/attention_mask/position_ids/self_distillation_mask`  
4. actor 读取这些字段，跑 teacher 前向  
5. `compute_self_distillation_loss` 用 mask + alpha + topk/full-logit + is_clip + rollout_is 计算损失  
6. 有效 step 后 teacher 再按 EMA/trust-region 机制更新

对应代码入口：

- `naive.py` 第 93-121 行
- `ray_trainer.py` 第 672-796 行
- `dp_actor.py` 第 684-849、913-921 行
- `core_algos.py` 第 1085-1188 行
- `fsdp_workers.py` 第 894-905 行

## 9. 结论（相对原生 verl 的本质变化）

SDPO 的关键变化不是“多一个 loss 名字”，而是把训练从“仅用终局分数回传”改成了“终局分数筛样 + 环境反馈条件化 + token 级蒸馏监督”三段式：

1. trainer 侧新增教师输入构造器；
2. actor 侧新增双前向蒸馏分支；
3. loss 侧新增 alpha/top-k/is_clip/rollout_is 组合；
4. teacher 侧新增可更新机制（EMA/trust-region）。

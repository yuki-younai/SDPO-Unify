# SDPO 实现分析

基于论文 "Reinforcement Learning via Self-Distillation" (SDPO) 和提供的代码库，本文档总结了 SDPO 的核心思想是如何在代码中实现的。

## 1. 概述

SDPO (Self-Distilled Policy Optimization，自蒸馏策略优化) 通过利用模型自身的高回报轨迹作为“自我教师”来增强在策 (on-policy) 强化学习。它将教师的预测（以反馈和成功演示为条件）蒸馏回学生策略中。

该实现集成在 `verl` 框架中，主要修改了 PPO 训练器 (trainer) 和执行者 (actor)。

## 2. 核心组件

### 2.1 配置
SDPO 通过 `verl/trainer/config/sdpo.yaml` 中的 `actor.policy_loss.loss_mode` 配置启用。

```yaml
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: sdpo
    self_distillation:
      max_reprompt_len: 10240
      is_clip: 2.0
      # ... 其他设置，如 alpha, 模板等
```

### 2.2 自我教师输入构建 (Reprompting)

SDPO 的关键思想是让教师以以下内容为条件：
1.  **原始提示 (Original Prompt)**
2.  **成功解决方案 (Successful Solution)**（来自之前的尝试或当前批次）
3.  **反馈 (Feedback)**（来自环境或验证器）

此逻辑在 `verl/trainer/ppo/ray_trainer.py` 的 `_maybe_build_self_distillation_batch` 方法中实现。

**关键步骤：**
1.  **识别成功案例**：`_collect_solutions_by_uid` 查找回报高于 `success_reward_threshold` 的轨迹。
2.  **收集反馈**：`_collect_feedback` 收集环境反馈（例如，编译器错误、测试结果）。
3.  **构建教师消息**：`_build_teacher_message` 使用 `SelfDistillationConfig` 中定义的模板格式化提示。

```python
# verl/trainer/ppo/ray_trainer.py

def _maybe_build_self_distillation_batch(self, batch, reward_tensor, ...):
    # ...
    # 1. 收集成功的解决方案
    success_by_uid = self._collect_solutions_by_uid(...)
    
    # 2. 收集反馈
    feedback_list = self._collect_feedback(...)

    # 3. 为每个样本构建教师消息
    messages = [_build_teacher_message(i) for i in range(batch_size)]
    
    # 4. 应用聊天模板并进行分词
    teacher_prompt = self.tokenizer.apply_chat_template(messages, ...)
    
    # 5. 与学生的响应拼接以形成教师输入
    teacher_input_ids = torch.cat([teacher_prompt["input_ids"], responses], dim=1)
    
    return DataProto.from_dict(tensors={
        "teacher_input_ids": teacher_input_ids,
        "self_distillation_mask": self_distillation_mask,
        # ...
    })
```

### 2.3 自蒸馏损失 (Self-Distillation Loss)

损失函数最小化学生策略和教师策略（以增强后的提示为条件）之间的 KL 散度。这在 `verl/trainer/ppo/core_algos.py` 中实现。

**函数**: `compute_self_distillation_loss`

**逻辑**:
-   它支持 **前向 KL (Forward KL)**、**反向 KL (Reverse KL)** 和 **Jensen-Shannon 散度 (JSD)**，由 `alpha` 控制。
-   它可以执行 **全对数蒸馏 (Full Logit Distillation)** 或 **Top-K 蒸馏**。
-   它应用 **重要性采样 (IS) 截断** (`is_clip`) 以稳定训练。

```python
# verl/trainer/ppo/core_algos.py

def compute_self_distillation_loss(student_log_probs, teacher_log_probs, ...):
    # ...
    if self_distillation_config.alpha == 0.0:
        kl_loss = F.kl_div(student, teacher, ...)
    elif self_distillation_config.alpha == 1.0:
        kl_loss = F.kl_div(teacher, student, ...)
    else:
        # JSD 插值
        mixture = logsumexp([student + log(1-alpha), teacher + log(alpha)])
        kl_loss = lerp(kl_student, kl_teacher, alpha)
    
    # 如果配置了，应用重要性采样比率
    if is_clip is not None:
        ratio = torch.exp(student_log_probs - old_log_probs).clamp(max=is_clip)
        per_token_loss = per_token_loss * ratio
        
    return loss, metrics
```

### 2.4 Actor 更新循环

`verl/workers/actor/dp_actor.py` 中的 actor 更新逻辑协调整个过程。

**逻辑**:
1.  检查 `loss_mode == "sdpo"`。
2.  **运行教师模型**：在 `teacher_input_ids`（在 trainer 中构建）上运行教师模型以获取 `teacher_log_probs`。
    -   教师模型可以是 actor 本身（自蒸馏）或缓慢更新的副本（EMA）。
3.  **计算损失**：调用 `compute_self_distillation_loss`。
4.  **更新教师**：使用 EMA (`_update_teacher`) 定期更新教师权重。

```python
# verl/workers/actor/dp_actor.py

def update_policy(self, data):
    # ...
    if loss_mode == "sdpo":
        # 在教师输入上进行前向传播
        teacher_outputs = self._forward_micro_batch(teacher_inputs, ...)
        
        # 计算 SDPO 损失
        pg_loss, metrics = compute_self_distillation_loss(
            student_log_probs=log_prob,
            teacher_log_probs=teacher_outputs["log_probs"],
            # ...
        )
    else:
        # 标准 PPO/GRPO 损失
        pg_loss, metrics = policy_loss_fn(...)
        
    # 反向传播和优化器步骤
    loss.backward()
    self._optimizer_step()
    
    # 更新教师 (EMA)
    self._update_teacher()
```

## 3. 反馈机制

反馈是在奖励计算阶段生成的。`RayPPOTrainer` 从 `reward_extra_infos_dict` 中提取此反馈并将其传递给教师输入构建过程。

-   **来源**: `verl/utils/reward_score/feedback/` 包含不同领域（数学、代码等）的逻辑。
-   **集成**: `RayPPOTrainer._collect_feedback` 提取反馈。
-   **使用**: 通过 `reprompt_template` 中的 `{feedback}` 占位符注入到提示中。

## 4. 总结

该实现紧密遵循论文内容：
1.  **自蒸馏**：通过最小化当前策略（学生）和其自身的增强版本（教师）之间的 KL 散度来实现。
2.  **丰富反馈**：通过使用来自高回报轨迹的反馈和解决方案动态构建教师的提示来整合。
3.  **在策 (On-Policy)**：学生是在策更新的，而教师提供了一个稳定的、改进的目标分布。

这种实现有效地将强化学习问题转化为自监督学习问题，其中模型学习预测其自身最佳（并经过反馈修正）尝试的 token。

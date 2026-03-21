# SDPO 将哪些样本用于蒸馏：代码级逻辑整理

本文只回答一个问题：在当前实现中，哪些样本会进入 SDPO 蒸馏损失，哪些不会。

## 1. 总判定：看 `self_distillation_mask`

在 trainer 中，样本是否参与蒸馏由这一条决定：

- `self_distillation_mask[i] = (solution_strs[i] is not None) OR (feedback_used[i] is True)`

对应代码位置：

- `verl/trainer/ppo/ray_trainer.py` 中 `_maybe_build_self_distillation_batch`
- `self_distillation_mask` 构造处约在 773-779 行

这意味着：只要样本拿到“可用成功解”或“可用反馈”之一，就参与蒸馏。

## 2. `solution_strs[i]` 如何变成可用

### 2.1 先按 uid 收集成功样本索引

- 序列分数：`seq_scores = reward_tensor.sum(dim=-1)`
- 成功条件：`seq_scores[idx] >= success_reward_threshold`
- 成功样本按 `uid` 分组记录

对应代码约在 636-643 行。

### 2.2 当前样本取演示解（`_get_solution`）

对样本 `i`：

- 先拿 `solution_idxs = success_by_uid[uid]`
- 若 `dont_reprompt_on_self_success=True`，会移除当前样本索引 `i`
- 移除后若为空，返回 `None`
- 否则取第一个成功样本作为演示解

对应代码约在 650-669 行。

因此，`dont_reprompt_on_self_success=True` 时，样本即便自己答对，也可能拿不到 solution（如果同 uid 下没有别的成功样本）。

## 3. `feedback_used[i]` 如何变成可用

### 3.1 反馈提取前置条件

反馈只在以下条件同时满足时才可能可用：

- `include_environment_feedback=True`
- `reward_extra_infos_dict["feedback"]` 里该样本是非空字符串（`strip()` 后仍非空）

对应代码约在 611-634 行。

### 3.2 反馈是否被真正使用

即便有反馈，还受 `environment_feedback_only_without_solution` 约束：

- 若该开关为 `False`：有反馈就用
- 若该开关为 `True`：只有“没有 solution 的样本”才用反馈

对应代码约在 767-771 行。

## 4. 所有样本都会建 teacher prompt，但不一定进蒸馏

`messages = [_build_teacher_message(i) for i in range(batch_size)]` 会对全样本构建 teacher prompt。

但如果某样本既无 solution 又无可用 feedback：

- 它的 `reprompt_text` 会退化为原始 `prompt`
- 且 `self_distillation_mask[i]=0`

对应代码约在 710-745、748 行、773-779 行。

## 5. 蒸馏损失如何真正“过滤样本”

在 loss 侧：

- `loss_mask = response_mask * self_distillation_mask.unsqueeze(1)`

如果某样本 `self_distillation_mask=0`，该样本所有 response token 在蒸馏项上都被 mask 掉，不贡献蒸馏梯度。

对应代码位置：

- `verl/trainer/ppo/core_algos.py` 中 `compute_self_distillation_loss` 约 1102-1105 行。

## 6. 三个高频场景速查

### 场景 A：样本自己答对，且 `dont_reprompt_on_self_success=True`

- 同 uid 下还有其他成功样本：可拿到 solution，参与蒸馏
- 同 uid 下没有其他成功样本：`solution=None`，是否参与取决于 feedback 是否可用

### 场景 B：整组都没答对

- 若有可用 feedback：可参与蒸馏（靠 feedback）
- 若无可用 feedback：该组样本 `mask=0`，蒸馏项无梯度

### 场景 C：无 rich feedback 配置（常见于 chemistry/sciknoweval）

- 常见设置下 `include_environment_feedback=False`
- 则能否参与蒸馏主要由是否有成功演示样本决定

## 7. 一句话结论

当前实现不是“按组过滤”，而是“按样本打 mask”：

- 样本级判定规则：`有 solution` 或 `有被使用的 feedback`
- 最终通过 `self_distillation_mask` 在 token 级损失中生效。

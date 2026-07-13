# SimVLA: A Simple VLA Baseline for Robotic Manipulation

tags: #paper #VLA #baseline #flow-matching #robot-manipulation

## 1. Basic Info

- Year: 2026
- Category: VLA baseline / training recipe
- Keywords: simple VLA, VLM backbone, lightweight action head, flow matching, training recipe, action normalization
- Relation to my work: 提醒在提出复杂 memory / WAM / streaming 模块前，必须先建立足够强、足够干净的 baseline。
- Reading status: read

## 2. One-sentence Summary

SimVLA 证明，一个非常简单的 VLM backbone + 轻量 action transformer + flow-matching action head，在训练 recipe 被严格标准化后，可以达到非常强的机器人操作性能；很多复杂 VLA 的收益可能来自训练细节而非结构本身。

## 3. Motivation

当前 VLA 领域有大量结构创新，例如：

- temporal augmentation；
- memory module；
- 3D / geometry prior；
- visual CoT；
- world model；
- complex action decoder。

但这些方法往往同时改变训练数据、backbone、优化策略、action normalization 和评测设置，使得我们很难判断性能提升到底来自哪里。

SimVLA 的动机是建立一个足够强的极简 baseline，让未来方法必须回答：

> 新模块到底解决了简单 baseline 解决不了的问题吗？

## 4. Method

### 4.1 Overall Architecture

SimVLA 的结构可以概括为：

```text
multi-view RGB + language instruction
        ↓
pretrained VLM backbone
        ↓
vision-language fused tokens
        ↓
lightweight action transformer
        ↓
flow matching action head
        ↓
continuous action chunk
```

它严格解耦 perception 与 control：

- VLM backbone 负责视觉-语言表征；
- action head 负责连续动作生成。

### 4.2 Action Head

SimVLA 使用 conditional flow matching 生成连续 action chunk。

直觉上：

```text
真实 action chunk x
随机噪声 ε
插值成 noisy action x_t
模型学习从 x_t 回到真实 action
```

推理时，从噪声开始，通过少量 Euler steps 生成 action chunk。

### 4.3 Training Recipe

SimVLA 最重要的不是结构，而是训练细节：

- data shuffling；
- action normalization；
- proprioception normalization；
- action horizon；
- VLM learning rate multiplier；
- optimization schedule；
- action head scale。

论文的 ablation 表明，关闭 data shuffling 或 action normalization 会导致性能近乎崩溃，而缩小 action head 反而影响较小。

## 5. Experiments

### 5.1 LIBERO

SimVLA 在 LIBERO 上报告了非常高的成功率：

| Suite | Success |
|---|---:|
| Spatial | 99.6 |
| Object | 99.8 |
| Goal | 98.6 |
| Long | 96.4 |
| Average | 98.6 |

这说明在标准 manipulation benchmark 上，极简结构已经很强。

### 5.2 Key Ablations

一些关键现象：

| Change | Effect |
|---|---|
| close data shuffling | performance collapse |
| close action normalization | performance collapse |
| increase VLM LR multiplier too much | severe degradation |
| reduce action head size | small degradation |
| change backbone | still strong |

核心结论是：训练 recipe 可能比结构创新更重要。

### 5.3 Robustness and Real Robot

SimVLA 在 LIBERO-PRO 和真实机器人上也有一定表现，但其对强空间扰动、任务组合变化、真实 long-horizon OOD generalization 仍有限。

## 6. Strengths

1. **极简但强**：给复杂 VLA 方法设置了很高的 baseline 门槛。
2. **强调训练细节**：data shuffling、normalization、LR multiplier 等隐藏变量对 VLA 性能极其关键。
3. **可复现性强**：相比复杂多模块系统，更适合作为新方法的对照组。

## 7. Weaknesses / Limitations

1. **不能证明复杂结构没用**：它只说明在一些 benchmark 上，复杂结构未必必要。
2. **缺少显式 long-history memory**：对非 Markov、状态歧义、周期任务可能不足。
3. **没有 world model / planning**：无法显式预测动作后果，也没有 value-guided planning。
4. **LIBERO 高分不等于真实 generalist robot**：标准 benchmark 可能已不足以检验真正泛化。

## 8. Connection to My Research

SimVLA 对我的研究有三个提醒：

### 8.1 必须建立强 baseline

在提出 memory、latent action、world model、streaming planner 之前，必须先确认一个简单 VLM + flow head 在同一数据和训练 recipe 下能达到什么水平。

### 8.2 控制训练变量

后续做实验时至少要控制：

```text
action normalization
proprioception normalization
data shuffling
action horizon
VLM LR multiplier
scheduler
action head scale
```

否则无法证明新模块真的有效。

### 8.3 复杂模块要解决 SimVLA 解决不了的问题

如果我的方法主打 long-history memory / WAM / streaming，需要证明它在以下场景有优势：

- 状态歧义；
- 长历史依赖；
- 重复 / cyclic task；
- 需要 future consequence reasoning；
- strong OOD spatial perturbation；
- long-horizon subtask switching。

## 9. Useful Sentences / Ideas

- A simple baseline can be surprisingly strong when training dynamics are standardized.
- Many VLA improvements may be confounded by silent implementation details.
- New architectural modules should be evaluated against strong recipe-controlled baselines.

## 10. Related Notes

- [[StarVLA]]
- [[π0.7]]
- [[Cosmos Policy]]
- [[Robot Memory]]

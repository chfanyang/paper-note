# π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities

tags: #paper #VLA #generalist-robot #prompt-conditioning #subgoal-image #cross-embodiment

## 1. Basic Info

- Year: 2026
- Category: generalist robotic foundation model / rich-context-conditioned VLA
- Keywords: π0.7, VLA, steerable policy, subgoal images, episode metadata, mixed-quality data, cross-embodiment, compositional generalization
- Relation to my work: 启发 VLA 不应只依赖语言指令，而应利用 richer context，例如 subgoal image、metadata、history、control mode 来增强泛化和长程任务能力。
- Reading status: read

## 2. One-sentence Summary

π0.7 的核心贡献是把 VLA 从 language-conditioned policy 推向 rich-context-conditioned generalist policy：通过 task / subtask instruction、subgoal images、episode metadata、control mode 和 memory，让模型能够从多源、多质量、多 embodiment 数据中学习可控、可组合的机器人技能。

## 3. Motivation

传统 VLA 通常只输入一句语言指令：

```text
π(a | observation, language)
```

但真实机器人数据高度混杂：

- 不同机器人；
- 不同控制模式；
- 不同质量 demonstrations；
- 成功和失败 rollout；
- 人类视频和非机器人数据；
- 同一任务的多种策略。

如果直接把这些数据混在一起训练，模型容易学到平均策略或失败行为。π0.7 的基本观点是：

> 不是丢掉复杂数据，而是通过更丰富的 prompt 告诉模型这段数据代表什么行为、什么质量、什么目标。

## 4. Method

### 4.1 Overall Architecture

π0.7 可以简化为：

```text
observation history + proprioception
        ↓
MEM-style history encoder
        ↓
VLM backbone
        ↓
flow matching action expert
        ↓
action chunk
```

同时，它还接收丰富 prompt：

```text
task instruction
subtask instruction
subgoal images
episode quality metadata
speed metadata
mistake label
control mode
```

### 4.2 Rich Context Conditioning

π0.7 的核心不是单纯加大模型，而是扩展条件变量：

```text
π(a | observation history, proprioception, task, subtask, subgoal image, quality, speed, mistake, control mode)
```

这相比普通 VLA 的 `π(a | observation, language)` 更适合处理混杂数据和复杂任务。

### 4.3 Subtask Instruction

长程任务不仅输入总任务，还输入当前子任务：

```text
Task: clean the kitchen
Subtask: open the fridge door
```

这样低层 policy 不需要自己完整规划，只需要在当前子任务条件下执行。

### 4.4 Subgoal Image

π0.7 使用多视角 subgoal images 描述近未来目标状态。其作用是把语言难以描述的空间细节转成视觉目标。

直觉：

```text
current image + desired future image
        ↓
inverse dynamics
        ↓
action
```

这和直接让 world model 输出动作不同。π0.7 中 world model 更像 prompt generator，生成 subgoal image 来指导 VLA。

### 4.5 Episode Metadata

π0.7 用 metadata 处理 mixed-quality data：

```text
Quality: 1-5
Speed: episode length / speed label
Mistake: true / false
Control Mode: joint / end-effector
```

训练时让模型知道每条数据的质量和策略属性；推理时用高质量 prompt steer 模型行为。

这带来一个重要思想：

> 低质量数据不一定要丢掉，可以通过条件化让模型知道它是低质量数据。

## 5. Experiments

π0.7 的实验重点不是某个标准 benchmark，而是展示 generalist robotic foundation model 的能力：

### 5.1 Out-of-the-box Performance

模型可以在无 task-specific post-training 的情况下执行一些复杂真实机器人任务，例如：

- espresso machine；
- folding laundry；
- taking out trash；
- folding a box；
- peeling vegetables。

### 5.2 Instruction Generalization

模型能在未见 kitchen / bedroom 环境中跟随复杂开放式指令，尤其是复杂 reference 和组合语义。

### 5.3 Cross-embodiment Transfer

π0.7 展示了 zero-shot cross-embodiment transfer。例如源机器人上的 folding / dexterous task，可以迁移到未见过该任务数据的目标机器人。

### 5.4 Compositional Task Generalization

通过 human coaching / high-level policy，π0.7 可以把已有技能组合到新任务中，例如 air fryer / toaster 类任务。

## 6. Main Contributions

我认为 π0.7 的主要贡献按重要性排序如下：

### 6.1 Rich-context-conditioned VLA

把 VLA prompt 从“做什么”扩展成“做什么 + 怎么做 + 目标状态长什么样 + 数据质量如何”。这是最核心的范式贡献。

### 6.2 Mixed-quality Data Conditioning

用 quality、speed、mistake 等 metadata 使模型可以利用失败数据、低质量 rollout 和 autonomous data，而不是简单过滤掉。

### 6.3 Subgoal Image as World-model-to-policy Interface

world model 不直接控制，而是生成 subgoal image，作为 low-level VLA 的视觉目标。

### 6.4 Demonstration of Generalist Capabilities

通过 instruction generalization、cross-embodiment transfer、compositional task generalization 展示了 rich context + diverse data 的潜力。

## 7. Strengths

1. **范式重要**：VLA 不再只是 language-conditioned policy，而是 rich-context-conditioned policy。
2. **数据利用思路先进**：不是只依赖高质量 demonstrations，而是利用 mixed-quality data。
3. **subgoal image 接口很适合机器人**：比纯语言 subgoal 更能表达空间目标。
4. **真实机器人能力强**：展示了更接近 foundation model 的 emergent capabilities。

## 8. Weaknesses / Limitations

1. **系统复杂度极高**：VLA、memory、world model、high-level policy、metadata annotation 全部耦合，很难拆分贡献。
2. **数据和标注依赖重**：quality、mistake、subtask、subgoal image 都需要复杂数据管线。
3. **subgoal image 依赖 world model 质量**：如果生成目标错误，policy 可能被误导。
4. **很多真实机器人实验难以复现和公平比较**。

## 9. Connection to My Research

π0.7 对我的研究有几个启发：

### 9.1 Memory / WAM 不应孤立存在

如果我要做 memory 或 WAM，它最好能作为 policy 的有效 condition，而不是只作为辅助 loss。

### 9.2 Subgoal image 是比 language 更强的中间接口

对于 long-horizon manipulation，纯语言子任务可能不够精确。视觉 subgoal 可以表达目标状态和空间细节。

### 9.3 利用失败 / rollout 数据需要 metadata

如果未来使用 autonomous rollout、online correction 或失败轨迹，不能直接混合训练，需要显式告诉模型数据质量和行为属性。

### 9.4 Long-horizon 可以拆成 high-level semantic policy + low-level rich-context VLA

这和 StreamVLA、Hi-VLA、Transition Bridge 等方向有联系。

## 10. Useful Sentences / Ideas

- VLA should be conditioned not only on what to do, but also on how to do it.
- Mixed-quality data can be useful if the model is told the quality and strategy behind each episode.
- Subgoal images provide a visually grounded interface between world models and action policies.
- Rich prompt conditioning can turn heterogeneous robot data into controllable conditional distributions.

## 11. Related Notes

- [[StarVLA]]
- [[SimVLA]]
- [[Cosmos Policy]]
- [[World Action Model]]
- [[Streaming VLA]]

# Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning

tags: #paper #World-Model #WAM #video-diffusion #robot-policy #planning #value

## 1. Basic Info

- Year: 2026
- Category: video-model-based robot policy / world action model / model-based planning
- Keywords: Cosmos Policy, video diffusion model, latent frame injection, action latent frame, future state prediction, value prediction, best-of-N planning
- Relation to my work: 与 WAM / ACR-WAM 强相关，说明 video model 可以被统一微调为 policy、world model 和 value model。
- Reading status: read

## 2. One-sentence Summary

Cosmos Policy 的核心思想是：直接把 pretrained video diffusion model 微调成机器人策略，让同一个 video model 在 latent diffusion 过程中同时生成 action、future state 和 value，从而统一 policy、world model 与 value function。

## 3. Motivation

传统 VLA 多数初始化自 VLM，而 VLM 主要从静态 image-text pair 中学习语义知识。机器人控制不仅需要识别物体和理解语言，还需要建模：

- 时序变化；
- 物理交互；
- 动作导致的未来状态；
- 复杂动作分布；
- action consequence。

视频生成模型天然学习世界如何随时间变化，因此 Cosmos Policy 假设：

> pretrained video diffusion model 的 spatiotemporal prior 可以作为 robot policy 的强初始化。

相比过去 video-based robot policy 常见的多阶段训练，Cosmos Policy 希望不改架构、不额外加 action diffuser，而是单阶段 post-training。

## 4. Method

### 4.1 Overall Architecture

Cosmos Policy 基于 Cosmos-Predict2-2B video foundation model。它接收当前状态和语言，输出：

```text
1. action chunk
2. future state: future images + future proprioception
3. value: expected rewards-to-go of future state
```

整体上可以理解为：

```text
current observation + language
        ↓
pretrained video diffusion model
        ↓
action chunk + future state + value
        ↓
direct execution or best-of-N planning
```

### 4.2 Key Idea: Latent Frame Injection

Cosmos 原本是 latent video diffusion model，输入输出是 video latent frames。

Cosmos Policy 没有新增专门 action head，而是把新模态也塞进 video latent sequence：

```text
proprioception
action chunk
future proprioception
value
```

这些非图像模态被人为构造成 latent frame，插入原本的视频 latent stream 中。

### 4.3 How Action Is Added as a Latent Frame

动作不是通过 VAE 编码的，而是这样做：

```text
action chunk: K × d_action
        ↓ normalize to [-1, 1]
        ↓ flatten
        ↓ repeat until filling H' × W' × C'
        ↓ reshape to latent volume
        ↓ replace a blank placeholder latent frame
```

也就是说，action latent frame 是人为铺满的连续 latent volume。

伪代码：

```python
a = normalize(action_chunk, range=(-1, 1))
v = a.flatten()
latent_size = H_prime * W_prime * C_prime
v_rep = repeat_until_length(v, latent_size)
z_action = v_rep[:latent_size].reshape(H_prime, W_prime, C_prime)
latent_sequence[action_position] = z_action
```

推理时，从生成的 action latent frame 中恢复 action：

```python
z = generated_latent_sequence[action_position]
copies = recover_repeated_vectors(z)
a_pred = copies.mean(dim=0)
action_chunk = unnormalize(a_pred)
```

### 4.4 Clean Condition

clean condition 指扩散模型中未加噪、作为条件输入的 latent frames。

它不是“干净数据”，而是 diffusion 里的 clean latent。

例如 latent sequence 可以写成：

```text
s, a, s', V(s')
```

其中：

- `s`: current state, 包括 current proprio + current images；
- `a`: action chunk；
- `s'`: future state, 包括 future proprio + future images；
- `V(s')`: future state value。

不同训练任务中，clean condition 和 target 不同：

| Training mode | Clean condition | Target |
|---|---|---|
| Policy training | `s` | `a, s', V(s')` |
| World model training | `s, a` | `s', V(s')` |
| Value training | `s, a, s'` | `V(s')` |

被作为 target 的 latent frames 会被加噪，模型学习 denoise。

### 4.5 Language Modality

language 没有通过 latent frame injection 加入。

语言沿用原始 Cosmos video diffusion model 的 text-conditioning 通道：

```text
language instruction
        ↓
T5 text encoder
        ↓
text embedding
        ↓
DiT cross-attention condition
```

也就是说：

- action / proprio / value 是 latent frames；
- language 是 text embedding，通过 cross-attention 进入 denoiser。

### 4.6 Direct Policy and Planning

Cosmos Policy 有两种推理方式。

#### Direct Policy

```text
current state + language
        ↓
generate action chunk
        ↓
execute
```

#### Best-of-N Planning

```text
generate N candidate action trajectories
        ↓
imagine future states for each candidate
        ↓
predict value for each future state
        ↓
execute action with highest value
```

这使得模型不仅能出动作，还能想象结果并进行简单搜索。

## 5. Experiments

论文报告 Cosmos Policy 在仿真和真实机器人上都很强：

- LIBERO average success: 98.5%；
- RoboCasa average success: 67.1%；
- real-world bimanual manipulation: 93.6%；
- model-based planning 在两个真实机器人任务上进一步提高 task completion rate。

核心实验结论：

> video diffusion model 可以同时作为 action generator、future state predictor 和 value estimator。

## 6. Strengths

1. **统一性强**：policy、world model、value function 都由同一个 video diffusion model 表达。
2. **架构改动少**：不新增复杂 action module，而是通过 latent frame injection 扩展模态。
3. **future state 和 value 参与推理**：不是只做 auxiliary prediction，而是可用于 best-of-N planning。
4. **非常适合 WAM 视角**：动作和未来视觉状态在同一生成过程中对齐。

## 7. Weaknesses / Limitations

1. **action latent frame 是人为铺开的**：`K × d_action` 被 flatten + repeat 到图像 latent volume，并没有显式 action 结构先验。
2. **video latent 的空间归纳偏置未必适合 action**：图像 latent 有空间局部性，action vector 没有。
3. **planning 仍然是短 horizon best-of-N**：不是完整 long-horizon planning。
4. **计算成本较高**：video diffusion policy 相比轻量 action head 更重。
5. **仍需要 target robot demonstration post-training**：不是完全 zero-shot policy。

## 8. Connection to My Research

Cosmos Policy 与我的 WAM / ACR-WAM 思路高度相关。

### 8.1 Action and Future State Should Be Coupled

如果只预测未来视频，可能出现“视频预测正确但动作不好”。Cosmos Policy 把 action、future state、value 放进同一生成模型中，强制它们对齐。

### 8.2 World Model Should Be Useful at Inference

world model 不应该只是 auxiliary loss。Cosmos Policy 的 future state 和 value 可以直接参与 best-of-N action selection。

### 8.3 Value Is Important for Planning

只知道未来图像是否合理还不够，还需要知道该未来状态是否有助于任务成功。value frame 是 world model 走向 planning 的关键。

### 8.4 与我的 ACR-WAM 的区别

Cosmos Policy 直接把 action 塞进视频 latent sequence；我的 ACR-WAM / dream expert 更像在 video DiT 与 action DiT 之间学习 action-centric representation。

可能的启发：

- 不要只做 video feature compression；
- compression token 应该被 action / future / value 共同监督；
- 如果有 world model 输出，应让它参与 action selection 或 policy conditioning。

## 9. Useful Sentences / Ideas

- Video models can serve as policy, world model, and value function simultaneously.
- Action can be represented as a latent frame and generated within the native video diffusion process.
- Future prediction becomes more useful when paired with value-guided action selection.
- World models should not only reconstruct the future; they should support decision making.

## 10. Related Notes

- [[π0.7]]
- [[World Action Model]]
- [[DreamZero]]
- [[Latent Action]]
- [[SimVLA]]

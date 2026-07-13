# StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing

tags: #paper #VLA #framework #benchmark #codebase

## 1. Basic Info

- Year: 2026
- Category: VLA framework / codebase / unified benchmark
- Keywords: VLA, modular framework, backbone-action head, world model backbone, multimodal co-training, cross-embodiment
- Relation to my work: 提供分析后续 VLA / WAM / memory 方法的统一坐标系。
- Reading status: read

## 2. One-sentence Summary

StarVLA 不是提出一个单一新模型，而是提出一个统一的 VLA 研发框架：把不同 VLA 方法拆成可替换的 backbone、action head、训练 recipe 和评测接口，从而缓解 VLA 研究中的架构、代码库和 benchmark 碎片化问题。

## 3. Motivation

当前 VLA 研究的主要问题不是单纯缺少新结构，而是不同方法之间太难公平比较：

- 架构各不相同；
- action decoding 方式不同；
- backbone 选择不同；
- 数据处理、训练 recipe、评测协议不统一；
- VLM-based VLA 和 world-model-based VLA 往往处在不同代码生态中。

StarVLA 的基本判断是：如果没有统一抽象，就很难知道一个方法的提升到底来自模型结构、backbone、action head，还是训练和评测细节。

## 4. Method

### 4.1 Unified VLA Formulation

StarVLA 将 VLA 统一表示为：

```text
π(a_{t:t+k}, y_aux | x_{≤t}, ℓ)
```

其中：

- `x_{≤t}`：当前及历史观测，可包含 RGB、depth、tactile、proprioception 等；
- `ℓ`：语言指令；
- `a_{t:t+k}`：未来 action chunk；
- `y_aux`：可选辅助输出，例如未来图像、语言推理、subgoal、future observation 等。

训练目标统一写成：

```text
L = L_action + L_aux
```

这使得 direct VLA、VLM-based VLA、world-model-based VLA 都可以被看成同一公式下的不同实例。

### 4.2 Backbone-action Head Decomposition

StarVLA 的核心工程抽象是：

```text
Vision / Language / World Backbone
        ↓
Action Head
        ↓
Action Chunk
```

其中 backbone 负责 perception / language / world representation，action head 负责把表征转成机器人动作。

这一区分非常重要，因为后续很多论文的创新可以被归类为：

- 改 backbone；
- 改 action head；
- 加 auxiliary loss；
- 加 memory；
- 加 world model；
- 改 training recipe；
- 改 evaluation interface。

### 4.3 Supported Action Decoding Paradigms

StarVLA 实现了多种 action head / decoding 方式：

| Variant | Mechanism | Interpretation |
|---|---|---|
| StarVLA-FAST | autoregressive action tokenization | 把动作当 token 生成 |
| StarVLA-OFT | parallel continuous regression | 轻量连续动作回归 |
| StarVLA-π | flow-matching / diffusion action expert | 类 π0 / π0.5 的连续控制 |
| StarVLA-GR00T | dual-system reasoning + fast action | 慢思考 + 快动作 |

重点不是哪一个绝对最好，而是它们可以在同一个接口下被替换和比较。

### 4.4 Training and Evaluation

StarVLA 支持：

- robot-only supervised fine-tuning；
- multimodal co-training；
- cross-embodiment co-training；
- 多 benchmark 统一评测；
- server-client 式 policy evaluation。

它整合了 LIBERO、SimplerEnv、RoboTwin 2.0、RoboCasa-GR1、BEHAVIOR-1K 等 benchmark，使得模型和环境之间通过统一 `predict_action()` 风格接口交互。

## 5. Experiments

这篇论文的实验意义主要不是单点 SOTA，而是展示：

- 同一框架可以支持 VLM backbone 与 world model backbone；
- 同一训练管线可以适配不同 action head；
- 同一评测接口可以覆盖多个仿真与真实机器人 benchmark；
- 简单 recipe 下也能得到强 baseline。

## 6. Strengths

1. **统一坐标系很有价值**：后续读 VLA / WAM 论文时，可以快速判断它到底改了哪个组件。
2. **工程复现价值高**：VLA 领域很多结果受 action normalization、controller、camera、benchmark wrapper 影响，统一框架能减少这些变量。
3. **VLM-based VLA 和 WM-based VLA 被放进同一视角**：有助于比较二者的真正差异。

## 7. Weaknesses / Limitations

1. 更像系统 / codebase paper，而不是提出一个强算法。
2. 统一接口不等于完全公平比较，因为 backbone 预训练数据、模型规模、动作空间和控制频率仍然可能不同。
3. `L_action + L_aux` 的 generalized VLA perspective 很有启发，但也可能过于宽泛。Cosmos Policy / DreamZero 这种视频生成 backbone 与普通 VLM backbone 在推理机制和时序先验上有本质差异。

## 8. Connection to My Research

StarVLA 对我的课题最重要的价值是提供分析框架。以后看任何 VLA / WAM / memory 论文，都可以问：

```text
1. 它用什么 backbone？
2. 它用什么 action head？
3. 它有没有 auxiliary output / auxiliary loss？
4. 它怎么处理历史和时间？
5. 它的提升来自架构，还是来自 training recipe？
```

这对设计自己的 memory / WAM / streaming VLA 方法很重要，因为可以避免把训练细节误认为结构创新。

## 9. Useful Sentences / Ideas

- VLA 可以被统一为 backbone-action head architecture。
- 不同 VLA 方法之间的差异，很多时候是 `L_aux` 的不同。
- VLM-based 与 world-model-based VLA 不一定是完全割裂的范式，而可能是同一 policy formulation 下的不同 inductive bias。

## 10. Related Notes

- [[SimVLA]]
- [[π0.7]]
- [[Cosmos Policy]]
- [[World Action Model]]

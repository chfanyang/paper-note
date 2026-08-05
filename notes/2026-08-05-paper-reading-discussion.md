# 近期机器人论文阅读交流笔记

> 整理日期：2026-08-05  
> 内容范围：本轮对话中共同阅读的论文、关键问题、方法流程、质疑与研究启发。

## 总体主线

这一组论文可以归纳为五条主线：

1. **如何让 action 更适合视觉/视频 backbone**：Masked Visual Actions、Lifting Embodied World Models、RynnBrain 的统一 action slot。
2. **如何让 world model / VLA 具备更强的长期规划与记忆**：WorldScape Policy 2.0。
3. **如何提高精细操作与在线控制质量**：Patch Policy、RTC、TempoVLA。
4. **如何显式表达任务约束与纠错信号**：GTA-VLA、STeP。
5. **如何从数据角度理解 embodied pretraining**：Data Pyramid、RynnBrain 1.1。

---

# 1. Masked Visual Actions for Unified World Modeling

论文：https://arxiv.org/pdf/2607.19343

## 核心问题

传统 action-conditioned video/world model 往往把低维 joint、EEF pose 或 action token直接输入视频 backbone，但 action 与 RGB/video latent 的模态差异很大。

该论文将 action 改写成：

> 机器人或物体在整段视频中的可视时空轨迹。

给定完整视频 `V` 和实体 mask `M`：

```math
C = M \odot V
```

模型输入初始帧 `I0` 和 masked visual action `C`，预测完整未来视频：

```math
p_\theta(V \mid I_0, C)
```

## 图 2：四种 action 表示的比较

### Raw joint angles

优点是控制精确；缺点是：

- 与视频模型输入模态差异最大；
- embodiment-specific；
- 模型还要隐式学会 joint → 机器人几何 → 接触 → 物体响应。

### Visualized end-effector pose

只告诉模型“手在哪里、朝向如何”，但没有提供：

- 机械臂其余连杆；
- 真实几何体积；
- 遮挡关系；
- 完整接触区域。

### Visualized skeleton

比单点信息更多，但仍然只是线框，缺少真实表面、厚度和占据区域。

### Masked Visual Actions

直接给出机器人完整视觉轨迹，使模型主要学习：

```text
已知机器人这样运动
        ↓
物体与环境会如何响应
```

最核心的思想是：

> action 条件越接近视频模型原生的像素/latent 空间，模型需要额外学习的跨模态映射越少。

## 图 3：统一 forward / inverse modeling

### Forward

给定机器人轨迹：

```math
p(\text{object motion} \mid \text{robot motion}, I_0)
```

流程：

```text
Policy 候选动作
→ URDF / FK / 相机渲染
→ Masked Visual Action
→ 视频模型预测完整未来
→ VLM / judge 选择最好结果
```

可用于：

- model-based planning；
- policy evaluation。

### Inverse

给定目标物体轨迹：

```math
p(\text{robot motion} \mid \text{object motion}, I_0)
```

流程：

```text
目标物体轨迹
→ 视频模型补全机器人操作视频
→ inverse dynamics model
→ 可执行机器人 action
```

关键观点：

> forward 和 inverse 可以被看作同一个联合视频分布在不同条件下的 completion。

## 图 4：训练数据如何构造

### Segmentation-based

```text
真实机器人视频
→ SAM 分割机器人
→ 保留机器人区域，其余设灰
→ Masked Visual Action
```

优点：真实、无需 URDF 与标定。  
缺点：推理时未来视频尚不存在，难以人为指定任意机器人轨迹；还可能通过遮挡轮廓泄露未来信息。

### Rendering-based

```text
关节状态
+ URDF
+ 相机内外参
→ FK
→ 机器人 mesh 渲染
→ Masked Visual Action
```

这个渲染不是学习模型，而是普通图形学流程：

1. 根据关节状态做 forward kinematics；
2. 得到每个 link 的 3D pose；
3. 用相机内外参投影；
4. 用 renderer 光栅化机器人；
5. 半透明机器人、红色夹爪只是视觉设计。

渲染器不负责物理仿真，也不预测物体结果。

## 评价与启发

这篇真正的新意不是“mask 控制视频”，而是：

> 把 action representation 搬到视频模型熟悉的视觉空间。

局限：

- 仍依赖 URDF、标定或 segmentation；
- 本质上是相关性生成模型，不保证 causal physics；
- 可能出现 ghost contact、乐观偏差。

与你当前方向的直接联系：

> 可以继续研究比完整机器人渲染更轻、更统一、同时保留几何和时序结构的 action representation。

---

# 2. DWM: Separating World Effects from Actions in Latent World Models

论文：https://arxiv.org/pdf/2607.18715

## 核心问题

传统 latent world model 直接学习：

```math
(z_t, a_t) \rightarrow z_{t+1}
```

但下一状态变化可能同时来自：

- action effect；
- world effect（重力、惯性、漂移等）。

论文希望分解：

```math
\hat z = \hat z^w + \hat z^a
```

其中：

- `zw`：与当前 action 无关的 world component；
- `za = z - zw`：action residual。

## 图 2 结构

原始 LeWM：

```text
observation
→ encoder
→ latent
+ action
→ predictor
→ pred head
→ 完整下一 latent
```

DWM 增加一个只在训练时使用的 world head：

```text
shared representation
├─ pred head  → 完整预测 z
└─ world head → zw

za = z - zw
```

### World consistency loss

对同一个状态历史，替换当前 action：

```math
\hat z^w(s,a_1) \approx \hat z^w(s,a_2)
```

用 InfoNCE 避免 world head 输出常数。

### Orthogonal loss

```math
\mathcal L_{orth}=|\cos(\hat z^w, \hat z^a)|
```

希望两部分少重复编码。

### 推理

推理时直接丢掉 world head，只保留原始 pred head，所以不增加 rollout 与 planning 成本。

## 关键质疑：`zw` 到底是不是 world effect？

我们的主要质疑是：

> 论文只证明 `zw` 对 action 不敏感，并没有充分证明它等于“零 action 条件下的真实下一状态”。

Action-invariant representation 可能编码：

- 当前状态；
- 场景身份；
- 与 action 无关的视觉信息；
- 其他不具有物理语义的状态函数。

因此：

```text
action-invariant
≠
真实 zero-action counterfactual
```

正交约束也只是 latent 几何约束，并不能自动赋予物理语义。

## 更有说服力的实验

应该从同一 simulator state 出发，分别执行真实 action 与 zero action：

```math
z_{t+1}^{0} = \phi(o_{t+1}^{a=0})
```

然后直接比较：

```math
\hat z^w \stackrel{?}{\approx} z_{t+1}^{0}
```

以及：

```math
\hat z^a \stackrel{?}{\approx}
\phi(o_{t+1}^{a})-\phi(o_{t+1}^{0})
```

还可以训练 linear probe，把 `zw` 解码为：

- 物体位置；
- 速度；
- 重力方向位移；
- 机器人状态。

## 最终评价

可以被实验支持的是：

> action-invariant auxiliary head + orthogonal residual regularization，能提升带外部动力环境下的 prediction 和 planning。

但“已经完成 world/action 物理分解”这个结论偏强。

---

# 3. WorldScape Policy 2.0

论文：https://arxiv.org/pdf/2607.18840

## 核心问题

如何让 joint video-action model 同时具备：

- 长时任务记忆；
- 子任务规划；
- goal image / demo video 条件；
- 多 embodiment action generation。

## 基础模型

以 joint video-action causal DiT 为主体：

```text
future video latent
+
future action tokens
→ shared DiT
→ future video + action chunk
```

不同 embodiment 共享 backbone，但使用各自 action encoder / decoder。

## 双通道历史记忆

### 短期视觉记忆

保存最近真实观测的 VAE latent：

```math
Z_t^{vis}=z_{t-S_v:t}^{obs}
```

直接作为 causal DiT 的 clean visual prefill。

作用：

- 最近接触；
- 夹爪开合；
- 局部运动方向；
- 视觉连续性。

### 长期事件语义记忆

每个 action chunk 开始时：

```text
当前 head-view
+ 高层任务
→ Qwen3-VL
```

从 Qwen3-VL 最后一层取两类 hidden states：

1. `u_t`：prefill 后的 perception tokens；
2. `r_t^{1:K}`：生成 K 个 planning token 时对应的最后层 hidden states。

拼接：

```math
q_t=[u_t;r_t^{1:K}]
```

历史队列存的是过去每个 chunk 的 `q_j`，不是原始 RGB 或动作。

## 历史压缩与三种 memory view

### Compact full history

- perception tokens 通过 attention pooling 压缩成 gist tokens；
- planning tokens 保留；
- 拼成 `H_t`。

### Global History

输入：

- 高层任务 embedding；
- `Mean(H_t)`。

作用：概括整个任务做到哪一步。

### Local Active

保留最近若干历史 chunk 的完整 `q_j`。

作用：当前子任务最近进行到哪一步。

### Event Boundary

先对每个 `q_j` 求均值，再计算相邻余弦变化：

```math
d_j=1-\cos(\bar q_j,\bar q_{j-1})
```

选变化最大的若干位置，保留对应完整 token。

作用：记住过去的重要任务切换点。

## 检索与门控

当前 `q_t` 作为 Query，历史 memory bank 作为 Key/Value：

```text
q_t
→ cross-attention 检索历史
→ memory result
→ sigmoid gate
→ residual fusion
```

```math
\hat q_t=q_t+\alpha\gamma_t\odot m_t
```

最终 `qhat_t` 作为 Joint DiT 的 cross-attention 条件，告诉策略“下一步应该做什么”。

短期视觉 latent 则通过 self-attention 告诉 DiT“最近具体怎么动”。

## Semantic forcing

训练数据有当前 event 的细粒度 caption `c_t`：

```text
caption
→ frozen T5
→ subtask semantic embedding s_t
```

同时：

```text
高层任务 + 当前图像 + 历史
→ qhat_t
```

用余弦损失让 `qhat_t` 接近真实 subtask caption 的语义。

推理时不再提供 fine-grained caption，只靠高层任务、当前图像和历史推断当前 subgoal。

## 评价

优点：

- 明确区分“任务进度记忆”和“局部动力学记忆”；
- 比简单拼接所有历史帧更有结构；
- semantic forcing 的训练逻辑合理。

保留意见：

- history 中存了 VLM 自己过去的 planning latent，可能累积错误；
- event boundary 的 latent change 可能只是视角变化；
- memory 内容本身缺乏可解释性验证；
- 三种 memory view 的独立收益没有被充分拆解。

---

# 4. Lifting Embodied World Models for Planning and Control

论文：https://arxiv.org/pdf/2604.26182

## 核心问题

已有低层 world model 需要搜索高维 joint action sequence，CEM 很难优化。

作者训练一个高层 waypoint-conditioned diffusion policy：

```text
4 个身体关键部位的 2D waypoint
→ 8 步、每步 48D 的低层关节动作
```

关键部位：

- pelvis；
- head；
- left hand；
- right hand。

总高层 action 维度：`4 × 2 = 8D`。

## 训练 waypoint 从哪里来

从未来真实人体姿态出发：

```text
未来 pose
→ FK 得到四个关键点的 3D 位置
→ 投影到当前相机
→ 四个 2D waypoint
→ 画到当前图像
```

Policy 输入：

- 最近几帧第一视角图像；
- 最近人体 pose；
- 画有 waypoint 的当前图像。

输出未来低层 joint action sequence。

## 推理时 waypoint 图片怎么来

不是模型生成，而是：

### 人工指定

用户在当前图像上点击四个关键点。

### CEM 搜索

CEM 在 8D waypoint 空间中采样：

```text
候选 waypoint
→ 画到当前图像
→ waypoint policy
→ 低层 joint actions
→ 冻结 PEVA rollout
→ 预测终点图像
→ 与 goal image 做 DreamSIM 比较
```

## Goal image 从哪里来

论文 benchmark 中，`o_g` 来自 Nymeria 同一条真实轨迹中的未来第一视角帧。

注意：

- CEM 能看到 `o_t、p_t、o_g`；
- 真实目标姿态 `p_g` 只用于最终评价；
- `o_g` 是任务定义，不是算法自动生成。

## CEM

Cross-Entropy Method：

1. 采样候选；
2. rollout；
3. 打分；
4. 选 elite；
5. 用 elite 更新高斯分布；
6. 重复。

CEM 是优化算法，MPC 是控制框架；CEM 可以作为 MPC 的求解器。

## 评价

提升来自两点：

1. 搜索空间从 384D 降到 8D；
2. diffusion waypoint policy 提供自然 motion prior。

需要区分“降维效果”和“motion manifold regularization”的贡献。

---

# 5. Guide, Think, Act: Interactive Embodied Reasoning in VLAs

论文：https://arxiv.org/pdf/2605.13632

## 核心问题

普通 VLA 出错时，用户很难低成本地告诉模型：

- 选错物体；
- 抓错位置；
- 路径不对。

GTA-VLA 提供 point / box / trace 视觉交互接口。

## Guide

用户可以提供：

- affordance point；
- bounding box；
- image-space trace。

这些提示被序列化成 Qwen3-VL 的坐标 token，而不是直接画到图像像素里。

## Think

Qwen3-VL 生成三类结构化 CoT：

1. Task CoT：任务分解；
2. Vision CoT：目标框、放置区、affordance point；
3. Robot CoT：二维末端轨迹草图。

Action head 真正读取的是这些 reasoning token 在 Qwen3-VL 最后一层的 hidden states。

## Act

慢速 VLM 循环：约 2 Hz，更新 reasoning cache。  
快速 action head：约 10 Hz，结合最新图像、wrist view、proprioception 和缓存 reasoning，生成连续 action chunk。

## 评价

优势：

> 人类提示在 reasoning 之前介入，因此可以改变目标 grounding 和后续动作生成，而不是在动作层面强行修正。

局限：

- guidance 实验较接近 oracle；
- 没解决“什么时候需要求助”；
- 2D point/trace 仍缺少深度、姿态和接触方向；
- Robot CoT 本质上很大程度来自投影后的 demonstration trajectory。

---

# 6. STeP: Signal Temporal Logic for Precise Specifications

论文：https://arxiv.org/pdf/2607.18580

## 核心问题

VLM 能理解高层语言，但难以可靠执行精确约束：

- 时间窗；
- 空间距离；
- 安全约束；
- 逻辑顺序。

STeP 在 VLM 与低层控制之间加入 STL specification layer。

## 完整流程

```text
自然语言 + 场景
→ GPT-4o 生成结构化子任务
→ deterministic compiler
→ STL formulas
→ 每个子任务选择 learned policy 或 STL-guided MPC
→ 实时计算 robustness
→ 成功 / 切换 / 重规划
```

## STL robustness

```math
\rho(z_{0:t},\varphi)
```

- `rho > 0`：满足；
- `rho < 0`：违反；
- 绝对值表示安全裕度或违反程度。

## MPC 简要理解

Model Predictive Control：

```text
当前状态
→ 用动力学模型预测未来 H 步
→ 优化动作序列
→ 只执行第一步
→ 获取新观测
→ 再次优化
```

STeP 将 STL robustness 加入 MPC objective。

## 四足中的 MPC

四足机器人里，MPC 通常优化：

- 未来机身轨迹；
- 接触脚的地面反作用力；
- 有时也优化落脚点。

再由 whole-body controller 转成关节力矩。

## 评价

优点：可解释、可监控、能明确指出违反了什么约束。  
局限：依赖人工 skills / predicate library；learned policy 只被 STL monitor，并没有被 STL 直接指导生成动作。

---

# 7. Data Pyramid for Embodied Manipulation

论文：https://arxiv.org/pdf/2607.24744

## 五层数据金字塔

从高机器人对齐、低扩展性到低机器人对齐、高扩展性：

1. Real robot；
2. UMI；
3. Egocentric / Exocentric human video；
4. Simulation；
5. General image-video-language data。

## 各层主要作用

### Real robot

提供真实 action-result pair、接触动力学、硬件噪声和失败过程。

### UMI

不占用机器人采集，但保留末端 pose、gripper 等 action proxy，需要 retargeting。

### Ego / Exo human video

提供任务结构、affordance、抓取选择、接触顺序和长时交互先验。

### Simulation

提供大规模 action/state、privileged labels、失败数据和系统性 domain randomization。

### General data

提供视觉语言、空间理解、任务语义和推理能力。

## 关键开放问题

- data recipe；
- failure / recovery data；
- tactile data；
- cross-embodiment action alignment；
- human video 到 robot action 的对齐。

## 启发

不同数据不是简单的“高质量/低质量”，而是在提供不同监督：

```text
General data       → 语义与认知
Human video        → 交互先验
Simulation         → 大规模可控 dynamics
UMI                → 真实环境 action proxy
Real robot         → 最终硬件对齐
```

---

# 8. TempoVLA

论文：https://arxiv.org/pdf/2606.06491

## 核心问题

让同一个 VLA 接收速度条件：

```math
\pi(o_t,s) \rightarrow A_t
```

其中 `s` 是 tempo command。

## VSTA

Variable-Speed Trajectory Augmentation：

先按运动模式和方向分段，再把 `q` 个原动作重新切成 `p` 个动作：

```math
s=\frac{q}{p}
```

- `q > p`：加速；
- `q < p`：减速。

夹爪事件作为硬边界，避免错误插值。

## 速度条件注入

比较三种方式：

- 文字前缀；
- RMSNorm modulation；
- soft prompt。

三者接近，说明主要增益来自正确的多速度监督，而不是复杂结构。

## 评价

优点：简单、实用，可动态调节 free-space 与 contact 阶段速度。  
局限：`s` 不是严格物理速度保证；不同速度下接触动力学未必等价；极端速度受低层 controller 饱和限制。

对 WAM 的启发：

> 不能只重采样 action，还需要同步重采样 future video 时间轴。

---

# 9. RynnBrain 1.1 / RynnBrain-VLA

论文：https://arxiv.org/pdf/2607.17977

## 两阶段体系

### RynnBrain 1.1

Embodied VLM，学习：

- 2D point / box / trajectory；
- contact point；
- 3D grounding；
- affordance；
- 操作规划。

### RynnBrain-VLA

把 backbone 作为单流 flow-matching Transformer：

```text
instruction
+ multi-camera visual tokens
+ robot state
+ noisy action tokens
→ shared Transformer
→ action flow velocity
```

## 81D unified action space

按身体部位和控制语义划分固定槽位：

- arms；
- EEF；
- gripper；
- dexterous hands；
- torso；
- head。

不同 embodiment 使用 mask 激活相应维度。

评价：

> 更像统一 action container，而不是真正统一运动语义。

Joint、EEF、hand、SONIC latent 仍是不同表示体系。

## RTC：Real-Time Chunking

每次生成 32 步 action chunk，但执行少量步后就启动下一次推理。

旧 chunk 未执行部分作为 inference-time guidance：

- 已经承诺/推理期间必然执行的前缀：强约束；
- 后续重叠区域：权重逐渐减弱；
- 远期部分：自由生成。

关键点：

- 整个过程只在推理时做；
- 不更新模型参数；
- 梯度只对当前 noisy action sample 求；
- 类似 diffusion classifier guidance。

## RTC 对操作精度的作用

直接作用：

- chunk 边界连续；
- 减少速度/位置跳变；
- 隐藏推理延迟。

间接作用：

- 更频繁使用最新真实观测重规划；
- 缩短开环长度；
- 改善低层 controller 的可跟踪性。

但 RTC 不会直接提升：

- 目标识别；
- 抓取点选择；
- 3D 定位；
- 单次 action prediction 能力。

---

# 10. Patch Policy

论文：https://arxiv.org/pdf/2607.18236

## 核心问题

很多机器人策略把 ViT 的所有 patch 压缩成 CLS 或平均池化向量，可能丢失精细控制需要的局部空间信息。

Patch Policy：

```text
冻结 ViT
→ 保留所有 patch tokens
→ 多帧 patch 展平
→ block-causal policy Transformer
→ VQ-BeT / Diffusion action head
```

## 图 2

### Observation trunk

每个相机画面经同一个冻结 ViT，保留全部 patch token。

### Goal image

Goal image 也经同一个 ViT 编码，与 observation patch 在 feature 维拼接。

### Goal vector

低维结构化目标复制到每个 patch，再在 feature 维拼接。

### Block-causal attention

- 同一帧内部：所有 patch 双向可见；
- 不同帧之间：只能看当前和过去。

原因：普通 token-level causal mask 会让同一张图中前面的 patch 看不到后面的 patch，破坏空间关系。

### Action readout

取每帧最后一个 patch hidden state 作为该时刻 action head 的输入。

## Goal image / goal vector 的作用

它们是 goal-conditioned policy 的正式输入，不是仅训练标签。

### Goal image

告诉 policy：

> 最终画面应该是什么样。

可能来自：

- demonstration 最终帧；
- 固定目标模板；
- 仿真器目标渲染；
- 用户提供的目标图。

### Goal vector

告诉 policy：

> 目标位置、接触点或状态参数是什么。

最明确的例子是 CAP 中的 3D contact point。

Patch Policy 本身不负责生成 goal。

## Goal 是否会天然提高成功率？

是的。和完全无 goal 的策略比，额外 goal 会消除任务歧义。

因此公平对比必须保持 goal 一致：

```text
Patch(o,g)
vs.
CLS(o,g)
```

而不是：

```text
Patch(o,g)
vs.
CLS(o)
```

## Dense patch 的直接对比对象

### WebSSL Patch vs WebSSL CLS / Avg Pool

保持：

- 同一个 WebSSL encoder；
- 同一个 action head；
- 同一个数据和 goal 设置。

只改变：

- 全部 patch；
- CLS；
- average pooling。

结果：

- BlockPush、Cube 这类强空间、多物体任务，patch 优势明显；
- LIBERO Goal 上优势不明显；
- Push-T 上提升有限。

### DINOv2 Patch vs DINOv2 CLS

真实机器人实验中：

- 同 DINOv2；
- 同 VQ-BeT；
- 同数据；
- 只改变 dense patch / CLS。

### Patch compression

从 256 patch 压缩到 64、16、4、1，性能下降，说明不是单纯 CLS token 训练不好，而是空间分辨率本身有价值。

## 评价

最可靠结论：

> 对强依赖局部位置、精细对齐和多物体关系的操作任务，dense patch representation 比单个 global feature 更有效。

不能推出：

> 小模型可全面替代开放词汇、复杂语言推理型 VLA。

---

# 11. 共同研究启发

## 1. 视觉侧与动作侧都不应过早压缩

Patch Policy 说明：

> 视觉不要过早压成 global token。

Masked Visual Actions 说明：

> action 不应只作为异质低维向量硬塞给视频 backbone。

可以形成统一观点：

```text
视觉侧：保留 dense spatial structure
动作侧：构造与视觉/几何空间对齐的 dense representation
```

## 2. “统一 action space”存在多个层级

- RynnBrain：统一槽位和 shape；
- LWM：统一成低维 visual waypoint；
- Masked Visual Actions：统一成机器人视觉轨迹；
- 可能的进一步方向：ray / geometry / motion field / raxel 等统一表征。

## 3. Latent 可解释性需要反事实验证

DWM 和 WorldScape 都把关键语义放进 latent：

- `zw`；
- reasoning latent；
- event memory。

仅靠最终成功率或 action invariance 不足以证明 latent 真的具有作者声称的物理/任务语义。

更强验证应包括：

- counterfactual rollout；
- probe；
- latent intervention；
- 显式解码或可视化；
- 双因素干预矩阵。

## 4. 在线控制增强与模型能力要区分

RTC、MPC、CEM、STL monitor 都能提升部署效果，但它们提升的并不一定是模型本身的认知或动作预测能力。

需要区分：

```text
模型能力提升
vs.
推理/控制系统增强
```

## 5. 失败与恢复是重要方向

Data Pyramid 和 STeP 都强调：

- 现有数据过度偏向成功 demonstration；
- 需要 failure onset、原因、恢复动作和结果；
- 可以考虑 training-free 或低成本 test-time correction，把“差一点成功”的轨迹变成成功。

---

# 12. 当前可继续深入的问题

1. 是否可以构造一种比完整 robot rendering 更轻、更通用的视觉 action representation？
2. 是否可以将 ray / raxel / camera geometry 与 action trajectory 融合成 video-native condition？
3. 如何做真正可验证的 world/action causal decomposition？
4. 如何在不牺牲纠错速度的情况下做 chunk continuity guidance？
5. 如何把 failure recovery 设计成 training-free / test-time adaptation？
6. dense patch token、3D geometry token、motion token 应该如何统一？
7. 在 WAM 中，能否同时用 action-conditioned future video 与显式 STL / geometry constraints 做规划？

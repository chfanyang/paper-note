# VLA / World Model Paper Reading Notes

这是一个用于 Obsidian 的论文阅读笔记仓库，主要围绕机器人操作、VLA、World Model、World Action Model、Streaming VLA、Memory 与 Long Video Understanding。

## 阅读路线

### 01 VLA Baselines

- [[01_VLA_Baselines/StarVLA|StarVLA]]：VLA 统一代码框架与 backbone-action head 抽象。
- [[01_VLA_Baselines/SimVLA|SimVLA]]：极简但强的 VLA baseline，强调训练 recipe 的重要性。
- [[01_VLA_Baselines/pi0.7|π0.7]]：rich-context-conditioned generalist robotic foundation model。

### 02 World Action Models

- [[02_World_Action_Models/Cosmos_Policy|Cosmos Policy]]：把 pretrained video diffusion model 微调成 policy、world model 和 value model。

### 03 Streaming VLA

待读论文：

- Thinking in Streaming Video / ThinkStream
- StreamVLA: Breaking the Reason-Act Cycle via Completion-State Gating

### 04 Robot Memory

待读论文：

- PAM: Adaptive Working Memory Recoding
- MemoAct
- CycleManip

### 05 Long Video Understanding

待读论文：

- MA-LMM
- M-LLM Based Video Frame Selection
- Re-thinking Temporal Search / T*
- Event-Anchored Frame Selection

### 06 Dense Video Captioning

待读论文：

- HiCM²
- Event-Equalized Dense Video Captioning
- HourHDVC

### 07 Grounding

待读论文：

- Point-VLA

## 推荐使用方式

每读一篇论文，统一按以下问题整理：

1. 它解决什么问题？
2. 它在 VLA / WAM / memory / streaming 体系中的位置是什么？
3. 它的核心结构是什么？
4. 它的训练目标和推理流程是什么？
5. 实验是否真正支撑它的 claim？
6. 它和我的研究有什么关系？
7. 它有什么可以借鉴或需要避免的地方？

## 研究主线索引

- [[VLA Baseline]]
- [[World Action Model]]
- [[Streaming VLA]]
- [[Robot Memory]]
- [[Long Video Understanding]]
- [[Dense Video Captioning]]
- [[Latent Action]]

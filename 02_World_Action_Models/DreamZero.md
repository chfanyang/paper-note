# DreamZero / World Action Models are Zero-shot Policies

tags: #WAM #World-Action-Model #Video-Diffusion #Robot-Manipulation #Cross-Embodiment #Generalization

## 1. Basic Info

- **Paper**: World Action Models are Zero-shot Policies
- **Method / System**: DreamZero
- **Year**: 2026
- **Category**: [[World Action Model]] / Video-model-based robot policy
- **Keywords**: World Action Model, WAM, video diffusion, action generation, inverse dynamics, zero-shot policy, cross-embodiment transfer, real-time closed-loop control
- **Related Notes**: [[Cosmos_Policy]], [[pi0.7]], [[LatBot]]

## 2. One-sentence Summary

DreamZero argues that robot foundation models should jointly predict **future world states and actions** rather than only mapping vision-language inputs to actions; by aligning future video generation with continuous action prediction, WAMs can obtain stronger generalization to unseen tasks, motions, environments, and embodiments than conventional VLM-based VLAs.

## 3. Core Motivation

The paper starts from a limitation of current VLA models:

- VLM-based VLAs inherit strong semantic priors from image-text pretraining.
- But static image-text pretraining is weak at physical dynamics, motion, geometry, and interaction priors.
- As a result, VLAs can generalize to new objects or language references, but struggle with **unseen physical skills** and **new environment dynamics**.

DreamZero's claim is that video diffusion models learn a different and more useful prior for robotics:

```text
VLM prior:
    what objects are, what language means

Video model prior:
    how the world evolves, how objects move, how interactions unfold
```

Therefore, instead of using a VLM as the main robot backbone, DreamZero builds a **World Action Model** that jointly models:

```text
p(future video, future action | observation history, language, proprioception)
```

## 4. Position in the Reading Map

DreamZero belongs to the same large family as [[Cosmos_Policy]], but its emphasis is different.

| Dimension | Cosmos Policy | DreamZero |
|---|---|---|
| Base model | Cosmos-Predict2-2B video diffusion | 14B autoregressive video diffusion transformer |
| Main outputs | action + future state + value | future video + continuous action |
| Key claim | video model can become policy/world/value model | WAM can be a zero-shot robot policy |
| Planning | best-of-N with predicted value | autoregressive closed-loop control |
| Generalization focus | strong policy and test-time planning | unseen motions/tasks/environments, cross-embodiment |
| System emphasis | single-stage fine-tuning of video model | scale, real-time control, cross-embodiment transfer |

Short distinction:

```text
Cosmos Policy:
    video model as policy + world model + value model

DreamZero:
    video-action joint generation as a generalist robot foundation model
```

## 5. Method Overview

### 5.1 WAM formulation

DreamZero defines a World Action Model as a model that conditions on current / historical observations and language, then predicts both future visual states and robot actions:

```text
input:
    observation history
    language instruction
    proprioceptive state

output:
    future video
    future continuous action chunk
```

The key is not merely adding video prediction as an auxiliary loss. The paper wants video and action to be **jointly generated**, so that action is tied to imagined world evolution.

### 5.2 Future video + inverse dynamics

A useful way to understand DreamZero is:

```text
DreamZero = future video prediction + inverse dynamics action prediction
```

But it is not a two-module pipeline. Instead, it uses a shared generative backbone so that video prediction and action prediction are aligned during training.

The intuition is:

```text
If the model imagines a plausible future state,
it should also know what action would bring the robot from the current state to that future.
```

### 5.3 Architecture

DreamZero uses a large autoregressive video diffusion transformer.

Simplified data flow:

```text
history images / current observation
language instruction
robot proprioception
        ↓
shared autoregressive diffusion transformer
        ↓
future video latents
future action chunk
```

Compared with Cosmos Policy, DreamZero more explicitly separates video and action output heads/decoders while maintaining a shared backbone for video-action alignment.

## 6. Why Autoregressive?

DreamZero emphasizes autoregressive video diffusion because robot control is naturally closed-loop.

The loop is approximately:

```text
1. predict future video + action chunk
2. execute action chunk on robot
3. receive real observation
4. replace imagined observation with real observation
5. update cache / context
6. predict next chunk
```

This has three advantages:

1. **Better closed-loop correction**: predicted futures do not have to accumulate indefinitely because real observations are inserted back.
2. **Efficiency**: AR structure allows KV/cache reuse.
3. **Temporal alignment**: action and video unfold together in a causal order.

This is different from a purely offline future-video generator.

## 7. Training Data View

DreamZero stresses that diverse, non-repetitive robot data can be more useful than many repeated demonstrations of the same tasks.

Traditional imitation learning often prefers:

```text
many clean demonstrations per task
```

DreamZero prefers a broader data regime:

```text
diverse robot trajectories
heterogeneous interaction data
non-repetitive behaviors
video-only demonstrations from humans or other robots
few-shot play data for new embodiments
```

The reason is that video prediction can learn from every transition, not only from repeated task-level demonstrations.

## 8. Main Experimental Claims

The main claims are:

1. DreamZero improves generalization to unseen tasks and environments compared with strong VLA baselines.
2. Joint video-action modeling gives stronger motion/skill generalization than static VLM-based policies.
3. Video-only data from humans or other robots can improve target robot performance.
4. A new embodiment can be adapted with very limited play data.
5. With algorithm/system/kernel optimization, a 14B video diffusion policy can run real-time closed-loop control around 7Hz.

Important note:

```text
"Zero-shot" here does not mean no robot data at all.
It means zero-shot generalization to unseen tasks / motions / environments after WAM training.
```

## 9. Strengths

### 9.1 Stronger physical prior than VLM-based VLA

The biggest conceptual strength is that DreamZero attacks the core weakness of VLM-based VLA: lack of dynamic physical priors.

For manipulation, knowing object semantics is not enough. The policy must understand:

- how objects move,
- how contact changes state,
- how the robot's motion affects the scene,
- what visual future corresponds to task progress.

Video-action joint modeling directly targets this.

### 9.2 Joint video-action alignment

DreamZero does not simply train a video predictor as a side task. It binds action and future visual state in one generative process.

This is important because:

```text
video prediction alone may look plausible but not be actionable;
action prediction alone may imitate data but not understand world evolution.
```

Joint modeling tries to connect both.

### 9.3 Cross-embodiment potential

The paper's video-only transfer result is especially important for robotics scaling. If a model can benefit from human videos or other robot videos without target-robot actions, then WAMs may scale beyond expensive teleoperation datasets.

### 9.4 Real-time engineering

A 14B video diffusion model is naturally too slow for robot control. DreamZero's system contribution is showing that with optimized denoising schedules, caching, quantization, and low-level kernels, such a model can become a real-time closed-loop policy.

## 10. Weaknesses / Limitations

### 10.1 Zero-shot wording needs caution

The model is not zero-shot from pure web video to robot control. It is trained on robot data and then evaluated on unseen tasks/environments/motions. The claim is still strong, but the term zero-shot should be interpreted carefully.

### 10.2 Very high system cost

DreamZero is not easy to reproduce in a normal academic lab:

- 14B video diffusion model,
- large-scale robot data,
- system-level parallelism,
- caching,
- quantization,
- CUDA kernel optimization,
- real-robot infrastructure.

### 10.3 Video plausibility is not equal to action success

Even if a predicted video looks plausible, the associated action may still be physically invalid, unstable, or difficult for the robot embodiment. DreamZero alleviates this by joint action decoding, but this remains a conceptual risk for WAMs.

### 10.4 Long-horizon symbolic planning is not the main focus

DreamZero improves skill/motion/environment generalization, but it is not primarily a hierarchical long-horizon planner. For tasks requiring explicit subtask decomposition, it may still need high-level planning or memory mechanisms.

## 11. Comparison with Related Methods

### 11.1 Compared with [[Cosmos_Policy]]

Cosmos Policy:

```text
current state + language
    → action + future state + value
    → direct execution or best-of-N planning
```

DreamZero:

```text
history observation + language + proprio
    → autoregressive future video + action
    → closed-loop robot execution
```

Cosmos Policy emphasizes a unified policy/world/value model and model-based planning. DreamZero emphasizes WAM as a generalist robot foundation model with stronger zero-shot and cross-embodiment generalization.

### 11.2 Compared with [[pi0.7]]

π0.7 is a rich-context VLA:

```text
observation + language + subtask + subgoal image + metadata → action
```

DreamZero is a world-action generative policy:

```text
observation history + language + proprio → future video + action
```

π0.7 uses a world model mainly to generate subgoal images as prompts. DreamZero makes the world model itself the core policy backbone.

### 11.3 Compared with latent action models

Latent action methods such as [[LatBot]] try to distill action-like representations from videos into VLA models. DreamZero directly trains a large model to jointly generate future video and actions.

A rough distinction:

```text
Latent action:
    learn compact action representation from video
    then transfer/distill to policy

DreamZero:
    directly learn joint world-action generation
```

## 12. Connection to My Research

DreamZero is highly relevant to my WAM / action-centric representation direction.

### 12.1 Supports the motivation of ACR-WAM

My motivation is that video prediction features are often redundant and not necessarily action-centric. DreamZero supports the idea that video and action must be aligned, but it does so at very large scale with joint generation.

My possible lightweight version:

```text
DreamZero:
    huge shared video-action generative model

ACR-WAM / Dream Expert:
    lightweight module that compresses video features into action-relevant latent tokens
```

### 12.2 Video features should be action-bound

A key lesson is that future visual prediction should not remain an isolated auxiliary task. It should be coupled to action prediction either through:

- shared backbone,
- joint denoising,
- action-conditioned future prediction,
- future-conditioned inverse dynamics,
- value or success prediction.

### 12.3 Data diversity matters

For my experiments, it may be useful to distinguish:

```text
repetitive expert demos
vs.
diverse rollout / failure / play data
```

WAM-style models may benefit more from diverse transitions than ordinary BC policies.

### 12.4 Long-horizon memory is still open

DreamZero is strong at motion and environment generalization, but it does not fully solve the memory / task-progress tracking problem. This leaves room for memory-based VLA or streaming planner work.

## 13. Useful Takeaways

- WAM should not be treated as merely adding a video prediction auxiliary loss.
- The important part is **world-action alignment**.
- Future video generation is useful only if it improves action selection or action representation.
- Cross-embodiment scaling may require exploiting video-only data.
- A practical WAM needs both algorithmic and systems-level acceleration.

## 14. Questions to Revisit

1. How exactly does DreamZero align video denoising and action denoising?
2. Are action latents generated through a separate decoder or injected into the video latent sequence?
3. What is the exact training mixture of robot data, human video, and cross-embodiment data?
4. How much performance comes from video pretraining vs. robot data diversity vs. system scale?
5. Can a small model reproduce part of the benefit via action-centric latent compression?

## 15. Short Conclusion

DreamZero's main contribution is to scale the WAM idea into a robot foundation model: by jointly predicting future video and continuous actions, it uses video dynamics priors to obtain stronger physical-skill generalization than conventional VLM-based VLAs. For my work, its most important lesson is that video features must be converted into action-relevant representations rather than used as generic visual predictions.

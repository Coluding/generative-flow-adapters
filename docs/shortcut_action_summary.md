# Summary: Shortcut Modeling (Primary) + Action Model

## 1) Thesis Core
We study how to turn a **frozen pretrained video generator** into an **efficient action-conditioned world model** by training only lightweight adapters.

Core formulation:
`f(x_t, t, c) = f_base(x_t, t) + Delta_phi(x_t, t, c)`

- `f_base`: frozen backbone (no full-model finetuning)
- `Delta_phi`: trainable adapter
- `c`: conditioning (mainly actions; optionally multimodal/context)

Main objective for thesis (priority order):
- **Shortcut modeling**: reduce effective denoising/rollout steps while preserving trajectory quality via consistency and shortcut-direction objectives.
- **Action modeling**: improve controllable dynamics under action inputs so shortcut rollouts stay behaviorally meaningful.

Shortcut research hypothesis:
- Learned shortcut objectives can move us to a better speed-quality frontier: fewer denoising/rollout steps at similar quality, or better quality at a fixed step budget.

## 2) Backbone Models and Architectures We Use
### Current primary implemented track
- **DynamiCrafter (diffusion-style video latent model)**
- Architecture style: **3D U-Net backbone** with residual blocks plus inserted spatial/temporal transformer blocks.
- Why important: natural fit for action control and hidden-state/control-style adapters.

- **Open-Sora (Flow matching video latent model)**
- **CosmosPredict-2.5 (Rectified Flow video model)**

### Unified framework support (already in repo)
- **Diffusion path**: predicts noise.
- **Flow path**: predicts velocity.
- Shared model interface and trainer logic allow same adapter framework across both.


## 3) Adapter Families Implemented
The repository already supports multiple adapter types under one composition wrapper:

- **Output adapters**
  - Direct correction at output level.
  - Includes lightweight affine style
  - No access to internal activations of base model

- **Hidden-state adapters**
  - Residual hidden modulation.
  - UniCon-inspired and related variants already represented (`unicon`, `replace_decoder`, `full_skip_controlnet`).
  - Very expensive --> not sure if desirable

- **LoRA adapters (low-rank weight updates)**
  - Injected into selected linear layers while backbone stays frozen.
  - Used for parameter-efficient action conditioning.

- **Hypernetwork adapters**
  - Dynamic generation of adapter weights conditioned on state/condition.
  - Includes **HyperAlign replication track** --> generate diffusion timestep specific LoRA weights


## 4) Shortcut Modeling Focus (Why It Is Central)
- In this thesis, action modeling is the control channel, while shortcut modeling is the main efficiency/scalability contribution.
- The core technical claim is not just controllability, but controllability **with fewer effective generation/rollout steps**.
- Therefore, all key experiments should be interpreted on a shortcut speed-quality frontier, not only on raw generation quality.

## 5) Shortcut + Action Modeling: Current Status Quo
Implemented and runnable now:

- Action conditioning encoders (`action` and structured `act`)
- Training objectives for diffusion/flow plus shortcut consistency terms:
  - local consistency
  - multistep self-consistency
  - shortcut-direction style flow setup
- Config-driven experiments for:
  - action-conditioned LoRA
  - HyperAlign action setup
  - flow shortcut training and hyper-shortcut variants
- End-to-end smoke tests/training tests to validate pipeline integration.

Shortcut-specific status:
- Shortcut objectives are already integrated into the trainer and config system.
- Step-size-aware conditioning and hyper-shortcut variants are already available.

So the infrastructure is no longer the bottleneck; the next bottleneck is experimental evidence on real data.

## 6) What We Need To Do Next (Concrete, Shortcut-First)
1. **Real-data benchmark setup**
- move from smoke data to a real action-conditioned video prediction dataset.
- fix train/val/test protocol and rollout horizon settings.

2. **Ablation matrix (core thesis experiments)**
- baseline frozen model (no adapter)
- shortcut-only adapter
- combined action+shortcut adapter
- action-only adapter (control reference)
- compare adapter families (output vs LoRA vs hypernetwork/HyperAlign).

3. **Metrics and evaluation**
- quality metrics (frame/video fidelity)
- temporal consistency
- action controllability / action-response alignment
- efficiency: quality vs number of denoising/rollout steps (primary metric curve)
- Pareto reporting: equal quality at fewer steps, or higher quality at fixed step budget

4. **Backbone strategy for thesis timeline**
- keep **DynamiCrafter as primary low-risk thesis track** for near-term results.
- optional second-track generalization study on transformer/flow backbone (OpenSora-style path) once primary results are stable.

5. **Thesis deliverables**
- reproducible configs + seeds
- final result tables and key ablations
- failure-case analysis and qualitative examples
- clear conclusion on whether shortcut modeling helps action-conditioned world modeling under parameter-efficient adaptation.

# Cosmos Predict2.5 Integration Report

## Executive summary

This report covers the local repository at `external_repos/cosmos-predict2.5` and evaluates whether its base model is a good integration target for this repository's adapter research.

Short answer:

- Yes, **if** we want a strong modern **latent video rectified-flow DiT** baseline for output adapters, low-rank adapters, hypernetworks, and DiT-native control.
- No, **if** the immediate goal is to extend the current **U-Net-centric UniCon / decoder replacement / full-skip ControlNet** line of work with minimal engineering.
- The only realistic target under the earlier `<=7B` preference is the **2B base checkpoint**. The 14B family is outside scope.

My recommendation:

1. Do **not** make Cosmos Predict2.5 the next backbone for the current U-Net adapter codepath.
2. Do treat it as a **second backbone family** that justifies adding explicit support for **transformer video backbones**.
3. Start with the **2B rectified-flow base** and support:
   - output adapters
   - LoRA / hyper-LoRA / HyperAlign-style weight modulation on transformer linears
   - lightweight hidden-state residual adapters attached to DiT blocks
4. Do **not** try to literally port the current UniCon / full-skip ControlNet assumptions to Cosmos. Build a DiT-native equivalent instead.

## What this repo actually contains

The repository is not just a thin inference wrapper around a released model. It is a fairly large NVIDIA codebase containing:

- `predict2`: the main Predict2.5 text/image/video generation stack
- `reason1`: the reasoning/text-embedding stack used for prompt conditioning
- `transfer2`: a control-conditioned branch for transfer / vid2vid tasks
- `predict2_multiview`, `camera`, `action`, `cosmos_policy`: downstream specializations
- `packages/cosmos-oss`: checkpoint registration, public model catalog, and release plumbing

For our purposes, the important path is:

- `external_repos/cosmos-predict2.5/cosmos_predict2/_src/predict2/`

The important architectural files are:

- `models/text2world_model_rectified_flow.py`
- `models/video2world_model_rectified_flow.py`
- `networks/minimal_v4_dit.py`
- `networks/minimal_v1_lvg_dit.py`
- `conditioner.py`
- `configs/video2world/...`
- `text_encoders/text_encoder.py`
- `tokenizers/cosmos.py`

## Which base model is the real released base

The public checkpoint registry in `packages/cosmos-oss/cosmos_oss/checkpoints_predict2.py` shows that the public base family includes a 2B and 14B line, plus multiview / robot / transfer variants.

For integration, the relevant public base is:

- `nvidia/Cosmos-Predict2.5-2B/base/pre-trained`

The important point is that the released base is wired to the **rectified-flow video2world stack**, not just a generic diffusion U-Net.

That maps to:

- config root: `cosmos_predict2/_src/predict2/configs/video2world/config.py`
- model wrapper: `Video2WorldModelRectifiedFlow`
- net: `MinimalV1LVGDiT` / `MiniTrainDIT`

So the true base is:

- latent video model
- text-conditioned
- optionally image/video-conditioned via frame conditioning
- rectified-flow trained
- transformer-based, not U-Net-based

## Base model architecture

### 1. Tokenizer / latent space

Cosmos Predict2.5 does **not** use a bespoke video latent model here. It uses the **Wan tokenizer / VAE** stack.

From `tokenizers/cosmos.py` and checkpoint registration:

- default tokenizer is `wan2pt1_tokenizer`
- checkpoint name resolves to `Wan2.1_VAE.pth`
- latent channel count is `16`

Implications:

- The denoiser operates in **latent video space**, not pixels.
- Our adapter output head would operate on latent velocity predictions with channel count 16.
- This is a good fit for our current frozen-backbone-plus-adapter abstraction.

### 2. Conditioning stack

The condition object starts as `Text2WorldCondition` and is extended to `Video2WorldCondition`.

Core fields include:

- `crossattn_emb`
- `padding_mask`
- `fps`
- `data_type`

Video conditioning adds:

- `gt_frames`
- `condition_video_input_mask_B_C_T_H_W`
- `num_conditional_frames_B`
- `use_video_condition`

Important design choice:

- Video conditioning is implemented by **masking and replacing the first latent frames**, not by a separate U-Net control encoder.
- For image/video-conditioned generation, the base model sees a latent frame prefix plus a conditioning mask.

This is structurally closer to prefix-conditioning than classic ControlNet.

### 3. Text encoder

The public high-quality base is configured around **Reason1 / Reason1.1** embeddings, not plain T5.

The main 2B rectified-flow experiment config sets:

- `text_encoder_class="reason1p1_7B"`
- `embedding_concat_strategy=FULL_CONCAT`

The text encoder code shows that this is based on **Qwen2.5-VL-7B-Instruct** and returns hidden states from all language layers. Those hidden states are mean-normalized and concatenated.

That produces a very large raw conditioning width. The 2B experiment therefore uses:

- `crossattn_proj_in_channels=100352`
- `crossattn_emb_channels=1024`
- `use_crossattn_projection=True`

So the conditioning path is:

1. huge Reason1 embedding tensor
2. learned projection down to 1024
3. cross-attention into the DiT blocks

Implications:

- The base denoiser is text-conditioned in a strong but expensive way.
- For our integration, we should not require online Reason1/Qwen inference at first.
- Better initial strategy: support **precomputed text embeddings** or a simplified text-conditioning path.

### 4. Rectified-flow wrapper

The released base is wrapped by `Text2WorldModelRectifiedFlow` or `Video2WorldModelRectifiedFlow`.

This wrapper is responsible for:

- tokenizer setup
- conditioner setup
- text encoder setup
- sampler setup via `FlowUniPCMultistepScheduler`
- rectified-flow training/inference logic via `RectifiedFlow`

The actual learnable denoiser is still:

- `self.net`

This matters because for our adapter framework the clean integration point is not the whole NVIDIA training system. It is:

- frozen tokenizer
- frozen conditioner interface
- frozen denoiser `net`
- thin wrapper that exposes `prediction = model(x_t, t, cond)`

That matches our repo's interface well.

### 5. Backbone: latent video DiT

The core backbone is `MiniTrainDIT` in `networks/minimal_v4_dit.py`.

The public 2B video-conditioned path uses `MinimalV1LVGDiT` in `minimal_v1_lvg_dit.py`, which is a thin subclass that:

- adds one extra input channel for the conditional-frame mask
- scales timesteps
- then delegates to `MiniTrainDIT`

The official 2B rectified-flow config uses approximately:

- `model_channels=2048`
- `num_heads=16`
- `num_blocks=28`
- `patch_spatial=2`
- `patch_temporal=1`
- `out_channels=16`
- `pos_emb_cls="rope3d"`
- `use_adaln_lora=True`
- `adaln_lora_dim=256`

This is a **flat transformer stack over spatio-temporal latent patches**.

There is:

- no encoder/decoder tower
- no multi-resolution U-Net pyramid
- no native skip-connection ladder in the U-Net sense

That single point determines most of the integration tradeoff.

### 6. Block structure

Each transformer block contains:

- self-attention
- cross-attention
- MLP
- AdaLN-style timestep conditioning with learned gates

The attention implementation uses explicit linears:

- `q_proj`
- `k_proj`
- `v_proj`
- `output_proj`

The MLP uses:

- `mlp.layer1`
- `mlp.layer2`

This is good news for adapters because these are exactly the module types we want for:

- LoRA
- hypernetwork-generated low-rank updates
- weight-space modulation

NVIDIA's own LoRA path already targets:

- `q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2`

That is a strong confirmation that this is the correct adapter surface.

### 7. AdaLN and timestep pathway

The model also has an explicit AdaLN modulation path, including an optional low-rank-style factorization via:

- `use_adaln_lora=True`
- `adaln_lora_dim=256`

This creates another promising adapter target family:

- timestep-conditioned modulation adapters
- hypernetwork-generated affine shifts/scales/gates
- small side networks that perturb the modulation pathway rather than the main attention weights

This is especially attractive if we want very parameter-efficient control without touching the full attention path.

### 8. Intermediate feature access

`MiniTrainDIT.forward(...)` supports `intermediate_feature_ids` and can return selected block features.

This is extremely useful for us because it means hidden-state adapters do not need to be implemented by fragile hooks only. We can build a cleaner DiT hidden-state API around block outputs.

That lowers the integration risk materially.

## What NVIDIA's own control branch tells us

The `transfer2` branch includes `ControlVideo2WorldModelRectifiedFlow` in:

- `cosmos_predict2/_src/transfer2/models/vid2vid_model_control_vace_rectified_flow.py`

This is important because it shows how NVIDIA itself adds stronger control to the base model.

What it does:

- encodes control hints into latent control inputs
- injects them through a specialized condition object
- uses control weights and masks in latent space

What it does **not** do:

- it does not turn the base into a classic Stable Diffusion-style U-Net ControlNet
- it does not rely on a standard encoder-decoder skip-copy control branch

Conclusion:

- Cosmos already has a **DiT-native control direction** in its own codebase.
- If we want a control-conditioned Cosmos adapter, we should follow that design logic rather than force a U-Net design onto it.

## Fit with our adapter families

## Output adapters

This is the easiest and cleanest fit.

Why:

- Our repo already treats the model as `f_base(x_t, t, c)` plus `Delta_phi(...)`.
- Cosmos predicts a latent-space rectified-flow velocity field.
- An output adapter can simply learn a residual correction on top of the frozen velocity prediction.

Recommended first experiment:

- freeze everything
- expose the base velocity prediction
- train a small latent output adapter conditioned on actions / multimodal inputs

This is low risk and immediately useful.

## LoRA / low-rank adapters

This is also an excellent fit.

Why:

- The base uses explicit linear projections in attention and MLPs.
- NVIDIA already supports LoRA on those exact modules.
- Our repo already has reusable low-rank injection machinery.

Recommended target set for a first Cosmos low-rank adapter:

- `q_proj`
- `k_proj`
- `v_proj`
- `output_proj`
- `mlp.layer1`
- `mlp.layer2`

Possible second-stage targets:

- timestep embedding linears
- cross-attention projection layer
- AdaLN modulation linears

This is probably the strongest integration story in the entire repo.

## Hypernetwork adapters

This is also a strong fit, but only if we make the hypernetwork DiT-aware rather than reusing U-Net assumptions.

Good hypernetwork targets:

- attention projections per block
- MLP projections per block
- AdaLN modulation weights per block

Good hypernetwork input signals:

- pooled conditioning embedding
- timestep / sigma / flow time
- optional pooled latent state summary
- optional selected block features from `intermediate_feature_ids`

The clean research opportunity here is:

- compare static LoRA vs stepwise LoRA vs hypernetwork-generated low-rank updates on a strong 2B rectified-flow video DiT

That would be a meaningful extension of this repository.

## Hidden-state residual adapters

This is viable, but it needs a new architecture family in our repo.

What maps well:

- per-block residual adapters after self-attention
- per-block residual adapters after cross-attention
- per-block residual adapters after MLP
- zero-init residual side branches over selected transformer blocks

What does not map well:

- decoder-only replacement in the U-Net sense
- full skip-copy control from encoder to decoder

So hidden-state adaptation is feasible, but it should be framed as:

- DiT block adapters
- transformer residual control
- modulation-path adapters

not as UniCon-for-U-Net.

## UniCon / replace-decoder / full-skip ControlNet

This is the weakest fit.

Reason:

- Our current implementations in `adapters/hidden_states/unicon.py` assume U-Net structure:
  - `input_blocks`
  - `middle_block`
  - `output_blocks`
  - encoder/decoder skip tensors
- Cosmos is a flat transformer stack.

So the following current adapters are **not directly portable**:

- `unicon`
- `replace_decoder`
- `full_skip_controlnet`

Could we build Cosmos equivalents? Yes, but they would be conceptually different:

- replicate selected transformer blocks in a side branch
- add zero-init residual connections into main block states
- inject extra condition tokens or latent hint tokens into cross-attention
- perturb AdaLN modulations instead of U-Net decoder activations

That is a valid research direction, but it is not a drop-in extension of the current UniCon code.

## Integration difficulty

## What is easy

- Treating the base as a frozen latent velocity field
- Reusing our `flow` model type and flow-matching objective
- Adding a new provider wrapper around the pretrained denoiser
- Attaching LoRA/hypernetwork modules to transformer linears
- Adding output adapters

## What is hard

- Reusing NVIDIA's full training stack
- Depending on their FSDP / Megatron / transformer-engine environment
- Supporting the full official inference pipeline end to end
- Loading the online Reason1 text encoder in a lightweight development setup
- Reusing our current U-Net hidden-state adapters without redesign

## Main dependency risks

The repo has heavy assumptions around:

- `transformer_engine`
- Megatron distributed state
- FSDP and DTensor codepaths
- NVIDIA's `imaginaire` infrastructure
- custom checkpoint handling

This means the correct integration strategy is **not** to absorb the whole repo into our training loop.

The correct strategy is a **thin wrapper** around the smallest viable inference-time base path.

## Recommended integration strategy

## Phase 1: thin frozen base wrapper

Add a new provider, for example:

- `provider: cosmos_predict2_5`

The wrapper should:

- instantiate the 2B rectified-flow base
- expose `forward(x_t, t, cond)` returning a latent velocity prediction
- hide NVIDIA-specific condition packing behind a small local adapter layer

The wrapper should not try to import or mirror the whole training system.

## Phase 2: minimal conditioning surface

Start with a narrow condition schema:

- text embedding tensor
- optional video-conditioning frames / mask
- optional adapter embedding for actions or multimodal control

Important practical recommendation:

- support **precomputed Reason1 embeddings** first
- do not require online Qwen/Reason1 execution for initial experiments

That keeps the integration within reach.

## Phase 3: first adapter experiments

First experiments should be:

1. output residual adapter on latent velocity
2. static LoRA on attention and MLP projections
3. hypernetwork-generated low-rank updates on the same projections

These three are the highest-value / lowest-risk set.

## Phase 4: DiT hidden-state adapters

After the base wrapper is stable, add a new hidden-state family for transformer backbones, for example:

- `architecture: dit_residual`
- `architecture: dit_control_branch`
- `architecture: dit_modulation_adapter`

This family should use:

- selected block outputs
- zero-init residual connectors
- optional cross-attention context augmentation
- optional AdaLN perturbation

That is the right Cosmos-native equivalent of our current U-Net hidden-state work.

## Concrete opportunities for our adapter research

## 1. Strong flow-matching benchmark

This repo currently wants to support both diffusion and flow-matching backbones behind one abstraction.

Cosmos is a strong test case because it is:

- modern
- open-source
- code-available
- genuinely flow/rectified-flow based
- sized reasonably at 2B
- text- and image/video-conditionable

So it is one of the better backbones for validating that our framework is not diffusion-only.

## 2. Better DiT adapter story than many released video repos

Many open video DiT repos are harder to adapt because they either:

- hide the model behind a large pipeline abstraction
- do not expose intermediate features cleanly
- do not show concrete LoRA targets

Cosmos does expose:

- explicit transformer blocks
- explicit attention/MLP linears
- optional intermediate features
- official LoRA support

That makes it unusually good for systematic adapter research.

## 3. Adapter design space around conditioning projection

The huge Reason1 embeddings are projected before cross-attention. That creates an interesting adapter location:

- adapt the cross-attention projection rather than the full denoiser

Possible variants:

- output-side correction after projection
- low-rank updates on projection weights
- hypernetwork-generated projection deltas conditioned on action or control signals
- extra learned control tokens concatenated after projection

This is a clean intervention point that does not exist in the same form in many smaller models.

## 4. AdaLN modulation as a control surface

The AdaLN pathway is not just incidental. It is a meaningful control interface.

Possible research directions:

- learn action-conditioned shifts/scales/gates
- hypernetwork-generate modulation deltas per timestep
- compare weight-space LoRA vs modulation-space control

That could become one of the more interesting contributions in this codebase because it is very natural for DiTs and not tied to U-Net skip geometry.

## Why it may not be worth integrating right now

It is not worth integrating immediately if the next milestone is one of these:

- finish a paper-faithful UniCon implementation on video U-Nets
- benchmark ControlNet-like skip-branch variants quickly
- stay within the current DynamiCrafter-centric codepath with minimal refactor

In those cases Cosmos will slow us down because the architecture mismatch is real.

The model is worth integrating if the next milestone is:

- add a serious modern flow-based video backbone to the framework
- show that the adapter abstractions generalize beyond U-Nets
- study LoRA / hypernetwork / output-adapter behavior on a strong video DiT

## Final recommendation

### Recommendation by research goal

If the goal is:

- **U-Net hidden-state / UniCon / ControlNet replication**
  - do not prioritize Cosmos next
- **modern flow-matching video backbone for adapter research**
  - integrate Cosmos 2B
- **LoRA / hypernetwork / modulation adapter research on video transformers**
  - integrate Cosmos 2B
- **fastest path to first action-conditioned results**
  - stay with DynamiCrafter first

### My overall judgement

Cosmos Predict2.5 is **worth integrating**, but not as a drop-in continuation of the current U-Net adapter line.

It is worth integrating as:

- our first serious **rectified-flow latent video DiT** backbone
- the backbone on which we develop **DiT-native adapter families**
- a benchmark for output adapters, LoRA, hypernetworks, and modulation adapters

It is **not** the right first target for:

- current UniCon code
- decoder replacement
- full-skip ControlNet variants

## Proposed implementation plan for this repo

1. Add `models/base/cosmos_predict2_5.py` with a thin frozen wrapper around the 2B rectified-flow denoiser.
2. Extend `models/base/factory.py` with a `cosmos_predict2_5` provider.
3. Keep the first condition API narrow: precomputed text embeddings plus optional frame-conditioning inputs.
4. Reuse existing output adapter experiments first.
5. Add a Cosmos-specific low-rank target inventory aligned with NVIDIA's own module names.
6. Add a new DiT hidden-state adapter family instead of trying to reuse the U-Net hidden-state adapters.
7. Only after that consider DiT-native control branches inspired by `transfer2`.

## Bottom line

- **Integrate** Cosmos Predict2.5 2B if we want a strong modern flow-based video transformer backbone.
- **Do not** integrate it as a shortcut for our current UniCon/ControlNet implementations.
- The best near-term value is in:
  - output adapters
  - LoRA / hypernetworks on transformer linears
  - AdaLN / modulation adapters
  - DiT-native hidden-state control

# OpenSora vs DynamiCrafter Architecture Report

## Scope

This report compares:

1. The **OpenSora backbone path** used in this repo (`provider: opensora`).
2. The **DynamiCrafter backbone path** used in this repo (`provider: dynamicrafter`).
3. How both are wired into adapter training.

Primary references are code locations in this repository and vendored/external model code.

---

## Executive Summary

- **OpenSora** is a **transformer-first, sequence-based flow model** (MMDiT with dual-stream then single-stream blocks), operating on packed video latent tokens and trained with continuous-time velocity matching.
  - Wrapper/config entry: `src/generative_flow_adapters/backbones/opensora/model.py:32-401`
  - Upstream architecture: `external_repos/Open-Sora/opensora/models/mmdit/model.py:39-268`, `.../layers.py:138-402`

- **DynamiCrafter** is a **3D U-Net-style latent diffusion model** with residual blocks plus spatial/temporal transformer blocks inserted at multiple resolutions, trained on discrete diffusion timesteps.
  - Wrapper entry: `src/generative_flow_adapters/models/base/dynamicrafter.py:14-119`
  - Core UNet: `src/external_deps/lvdm/modules/networks/openaimodel3d.py:310-793`
  - Spatial/temporal attention modules: `src/external_deps/lvdm/modules/attention.py:285-470`

- For adapters, both are exposed through the same `BaseGenerativeModel` interface and `AdaptedModel` composition path:
  - `src/generative_flow_adapters/models/adapted_model.py:15-89`
  - `src/generative_flow_adapters/training/builders.py:22-44`

---

## 1) OpenSora Architecture (Detailed)

## 1.1 Wrapper-level contract

- `OpenSoraModelWrapper` sets `model_type="flow"` and `prediction_type="velocity"` at init:
  - `src/generative_flow_adapters/backbones/opensora/model.py:90`
- Forward signature is standard framework API `(x_t, t, cond)`:
  - `src/generative_flow_adapters/backbones/opensora/model.py:270-275`
- Input expected as latent video tensor `(B, C, T, H, W)`:
  - `.../model.py:279-303`

## 1.2 Tokenization and positional IDs

- Video latent packing:
  - `pack_latents`: `src/generative_flow_adapters/backbones/opensora/common.py:19-46`
- Unpacking output back to video layout:
  - `unpack_latents`: `.../common.py:49-81`
- 3D image token IDs `(t, h, w)`:
  - `create_image_ids`: `.../common.py:84-118`
- Text IDs:
  - `create_text_ids`: `.../common.py:121-144`

Wrapper flow:
- packs `x_t` -> creates `img_ids` -> prepares conditioning -> calls MMDiT -> unpacks output:
  - `src/generative_flow_adapters/backbones/opensora/model.py:302-342`

## 1.3 Conditioning channels

`_prepare_conditioning` supports:

- `txt` (T5 embeddings) and optional `prompt` encoding fallback:
  - `.../model.py:366-379`, `403-423`
- `y_vec` (CLIP pooled embedding), optional prompt encoding fallback:
  - `.../model.py:383-390`, `425-445`
- `cond` for I2V (packed if present):
  - `.../model.py:391-395`
- `guidance` scalar/tensor:
  - `.../model.py:396-401`

## 1.4 Upstream MMDiT core

OpenSora upstream `MMDiTModel` defines:

- config with `depth`, `depth_single_blocks`, `axes_dim`, `cond_embed`, `guidance_embed`, `fused_qkv`, etc.:
  - `external_repos/Open-Sora/opensora/models/mmdit/model.py:39-60`
- input projections:
  - `img_in`, `txt_in`, `time_in`, `vector_in`, optional `guidance_in`, optional `cond_in`:
  - `.../model.py:98-114`
- **DoubleStream blocks** (text/image streams processed jointly in attention):
  - stack definition: `.../model.py:115-126`
  - block + processor internals: `.../layers.py:195-307`
- **SingleStream blocks** (merged token stream):
  - stack definition: `.../model.py:128-138`
  - block + processor internals: `.../layers.py:309-389`
- final projection layer:
  - `.../layers.py:391-402`
- main forward path:
  - `prepare_block_inputs`: `.../model.py:154-203`
  - block traversal: `.../model.py:208-233`

## 1.5 Attention mechanics

- `SelfAttention` supports fused or split QKV paths:
  - `external_repos/Open-Sora/opensora/models/mmdit/layers.py:138-169`
- Uses QK normalization (`QKNorm`) before attention:
  - `.../layers.py:126-136`, `151-163`
- Double-stream processor explicitly concatenates text/image QKV for joint attention:
  - `.../layers.py:238-245`

---

## 2) DynamiCrafter Architecture (Detailed)

## 2.1 Wrapper-level contract

- `DynamicCrafterUNetWrapper` is a thin adapter around vendored `UNetModel`:
  - `src/generative_flow_adapters/models/base/dynamicrafter.py:14-16`
- `from_config` loads YAML UNet params and schedule params:
  - `.../dynamicrafter.py:31-57`, `122-149`
- Forward takes `(x_t, t, cond)` and maps cond fields into UNet kwargs (`context`, `act`, `fs`, `concat`):
  - `.../dynamicrafter.py:78-115`

## 2.2 UNet macro-architecture

DynamiCrafter `UNetModel`:

- hierarchical U-Net with input blocks, middle block, output blocks:
  - `src/external_deps/lvdm/modules/networks/openaimodel3d.py:454-616`, `618-690`
- down/up sampling and residual blocks:
  - `.../openaimodel3d.py:52-126` (down/up helpers)
  - `.../openaimodel3d.py:128-220` (ResBlock)
- timestep embedding + optional action/fps conditioning:
  - `.../openaimodel3d.py:439-453`, `416-438`, `755-763`

## 2.3 Spatial + temporal transformer insertion

- `SpatialTransformer` over flattened spatial tokens:
  - `src/external_deps/lvdm/modules/attention.py:285-361`
- `TemporalTransformer` over time axis:
  - `.../attention.py:363-470`
- Both are inserted at selected resolutions in input/mid/output paths:
  - `src/external_deps/lvdm/modules/networks/openaimodel3d.py:492-528`, `569-599`, `636-672`

## 2.4 Runtime forward behavior

- UNet forward:
  - `src/external_deps/lvdm/modules/networks/openaimodel3d.py:712-793`
- Notable logic:
  - repeats text context over frames / handles mixed text+image context:
    - `.../openaimodel3d.py:719-728`
  - action conditioning and dropout mask:
    - `.../openaimodel3d.py:729-750`
  - optional adapter feature injection hooks (`features_adapter`):
    - `.../openaimodel3d.py:771-777`

---

## 3) Training Objective and Time Parameterization

## 3.1 OpenSora flow objective

In this repo, flow objective and shifted continuous timesteps are in:

- `src/generative_flow_adapters/losses/flow_matching.py:14-167`
  - timesteps sampled in `[0,1]` via sigmoid of Gaussian: `:77-80`
  - schedule shift:
    - linear spatial scaling: `:96-100`
    - temporal `sqrt(num_frames)` scaling: `:100-102`
    - shifted time formula `t' = alpha t / (1 + (alpha-1)t)`: `:104-106`
  - velocity target: `(1 - sigma_min) * noise - x_start`: `:161-167`

OpenSora upstream training uses the same core pattern:

- shift alpha from spatial token count plus temporal scaling:
  - `external_repos/Open-Sora/scripts/diffusion/train.py:385-390`
- velocity target:
  - `.../train.py:443`, `462-466`

## 3.2 DynamiCrafter diffusion objective

- Discrete diffusion schedule and noisy sampling:
  - `src/generative_flow_adapters/losses/diffusion.py:18-66`
- Integer timesteps with beta schedule:
  - `.../diffusion.py:50-52`

## 3.3 Trainer branching

- `Trainer.training_step` dispatches by `model_type`:
  - `src/generative_flow_adapters/training/trainer.py:53-110`
- `model_type == "diffusion"` uses `DiffusionTrainingObjective`.
- flow branch now samples shifted continuous timesteps by default (unless explicitly overridden):
  - `.../trainer.py:85-107`

---

## 4) Side-by-Side Architecture Comparison

## 4.1 Backbone family

- OpenSora: transformer-only MMDiT sequence model
  - `external_repos/Open-Sora/opensora/models/mmdit/model.py:69-140`
- DynamiCrafter: convolutional/residual U-Net with transformer blocks inside
  - `src/external_deps/lvdm/modules/networks/openaimodel3d.py:310-710`

## 4.2 Data layout

- OpenSora backbone internally runs token sequence `(B, L, D)` after patch packing:
  - packing in wrapper: `src/generative_flow_adapters/backbones/opensora/model.py:302-304`
- DynamiCrafter core runs feature maps and frequently reshapes to `(b*t, c, h, w)`:
  - `src/external_deps/lvdm/modules/networks/openaimodel3d.py:752`

## 4.3 Conditioning style

- OpenSora:
  - explicit text token stream + global vector stream + optional i2v tensor stream
  - wrapper conditioning assembly: `src/generative_flow_adapters/backbones/opensora/model.py:344-401`
  - upstream fusion path in `prepare_block_inputs`: `external_repos/Open-Sora/opensora/models/mmdit/model.py:154-203`

- DynamiCrafter:
  - cross-attention context + timestep/action/fps embedding in residual path
  - forward conditioning handling: `src/external_deps/lvdm/modules/networks/openaimodel3d.py:729-763`
  - wrapper cond mapping: `src/generative_flow_adapters/models/base/dynamicrafter.py:85-115`

## 4.4 Time/noise parameterization

- OpenSora path in this repo: continuous flow time in `[0,1]`, velocity target.
  - `src/generative_flow_adapters/losses/flow_matching.py:53-106`, `161-167`
- DynamiCrafter path in this repo: discrete diffusion timesteps with beta schedule.
  - `src/generative_flow_adapters/losses/diffusion.py:18-82`

## 4.5 Scale and typical defaults

- OpenSora canonical dimensions in wrapper config object:
  - `src/generative_flow_adapters/backbones/opensora/model.py:37-45`
- DynamiCrafter 512 config example:
  - UNet params: `external_repos/avid/latent_diffusion/configs/train/dynamicrafter_512.yaml:38-69`

---

## 5) Adapter Integration Implications

## 5.1 Common integration layer

- Both backbones are wrapped and then composed via `AdaptedModel`:
  - `src/generative_flow_adapters/training/builders.py:22-44`
  - `src/generative_flow_adapters/models/adapted_model.py:55-89`

This means adapter code can stay mostly backbone-agnostic at interface level, but feature geometry and conditioning semantics differ significantly underneath.

## 5.2 Why OpenSora vs DynamiCrafter adapters feel different

- OpenSora adapters usually target transformer projections (`q/k/v/proj`, or fused equivalents):
  - recommended targets: `src/generative_flow_adapters/backbones/opensora/model.py:465-497`
- DynamiCrafter adapters may couple to U-Net block outputs and/or attention/residual maps with strong multiscale structure:
  - UNet block hierarchy: `src/external_deps/lvdm/modules/networks/openaimodel3d.py:454-690`

## 5.3 Practical migration notes

When porting a DynamiCrafter-style adapter idea to OpenSora:

1. Replace multiscale map assumptions with sequence-token assumptions.
2. Revisit conditioning payload shape (OpenSora expects text token + vec + optional i2v cond tensors).
3. Keep objective consistent (flow velocity + shifted continuous timesteps for OpenSora).

---

## 6) Configuration Anchors in This Repo

- OpenSora example config:
  - `configs/opensora_output_adapter.yaml:12-70`
- DynamiCrafter example config:
  - `configs/diffusion_output_dynamicrafter.yaml:3-44`

These two configs are the quickest concrete reference for how each backbone is currently exercised in this codebase.


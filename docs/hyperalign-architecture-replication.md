# HyperAlign Architecture Replication

## Goal

This document describes the changes made to turn the existing HyperAlign adapter from a paper-inspired approximation into a paper-faithful architectural replication for the video backbone used in this repository.

Scope:

- replicate the HyperAlign architecture from the paper and appendix
- adapt it to the DynamiCrafter video U-Net used here
- keep the existing training framework intact

Out of scope:

- the paper's reward-maximization training objective
- the paper's preference regularization loss
- end-to-end reproduction of the paper's training recipe and metrics

This is therefore an architectural replication, not yet a full method replication.

## Files Changed

- [src/generative_flow_adapters/adapters/hypernetworks/hyperalign.py](/Users/lukasbierling/Documents/thesis-uva/code/generative-flow-adapters/src/generative_flow_adapters/adapters/hypernetworks/hyperalign.py)
- [src/generative_flow_adapters/adapters/low_rank/common.py](/Users/lukasbierling/Documents/thesis-uva/code/generative-flow-adapters/src/generative_flow_adapters/adapters/low_rank/common.py)
- [src/generative_flow_adapters/adapters/dynamicrafter_conditioning.py](/Users/lukasbierling/Documents/thesis-uva/code/generative-flow-adapters/src/generative_flow_adapters/adapters/dynamicrafter_conditioning.py)
- [src/generative_flow_adapters/adapters/factory.py](/Users/lukasbierling/Documents/thesis-uva/code/generative-flow-adapters/src/generative_flow_adapters/adapters/factory.py)
- [src/generative_flow_adapters/adapters/output/dynamicrafter.py](/Users/lukasbierling/Documents/thesis-uva/code/generative-flow-adapters/src/generative_flow_adapters/adapters/output/dynamicrafter.py)
- [configs/diffusion_hyperalign_action.yaml](/Users/lukasbierling/Documents/thesis-uva/code/generative-flow-adapters/configs/diffusion_hyperalign_action.yaml)
- [README.md](/Users/lukasbierling/Documents/thesis-uva/code/generative-flow-adapters/README.md)
- [tests/test_hyperalign_architecture.py](/Users/lukasbierling/Documents/thesis-uva/code/generative-flow-adapters/tests/test_hyperalign_architecture.py)

## Paper To Code Mapping

### 1. Attention-only LoRA targets

Paper:

- HyperAlign generates LoRA for attention projection layers only
- appendix target set: `to_q`, `to_k`, `to_v`, `to_out.0`

Implementation:

- introduced `PAPER_HYPERALIGN_TARGET_MODULES` in [common.py](/Users/lukasbierling/Documents/thesis-uva/code/generative-flow-adapters/src/generative_flow_adapters/adapters/low_rank/common.py)
- HyperAlign now uses exact suffix matching instead of the generic substring matching used by the baseline LoRA path
- `hyperalign` construction in [factory.py](/Users/lukasbierling/Documents/thesis-uva/code/generative-flow-adapters/src/generative_flow_adapters/adapters/factory.py) now defaults to the paper target set and rejects non-paper target inventories

Why:

- the previous implementation inherited the generic adapter target selection, which was too loose and not paper-faithful

### 2. Auxiliary-factorized LoRA generation

Paper:

- each module's LoRA is decomposed as:
  - `A_i = A_aux_i @ A_hyper_i`
  - `B_i = B_hyper_i @ B_aux_i`
- the hypernetwork predicts fixed-shape `A_hyper_i` and `B_hyper_i`
- module-specific auxiliary matrices absorb dimension differences across layers
- `B_aux` is zero-initialized so the effective LoRA starts at zero

Implementation:

- extended `WrappedLoRALinear` in [common.py](/Users/lukasbierling/Documents/thesis-uva/code/generative-flow-adapters/src/generative_flow_adapters/adapters/low_rank/common.py) with optional HyperAlign auxiliary matrices
- added:
  - `down_aux` with shape `[d_in, a]`
  - `up_aux` with shape `[b, d_out]`
  - `set_dynamic_hyper_factors(...)`
- kept the baseline static LoRA path unchanged for non-HyperAlign adapters
- zero-initialized `up_aux`, which is the implementation-side equivalent of the paper's zero-initialized `B_aux`

Why:

- the old HyperAlign path predicted full per-layer LoRA tensors directly
- that was simpler, but it was not what the paper describes

### 3. Shared decoder output head

Paper:

- the transformer decoder outputs one token per adapted module
- a final linear layer maps each decoded token to a shared dimension `d_w = r(a + b)`
- this output is reshaped into `A_hyper_i` and `B_hyper_i`

Implementation:

- removed the old per-module `_LoRAParameterHead`
- added a shared `self._factor_head` in [hyperalign.py](/Users/lukasbierling/Documents/thesis-uva/code/generative-flow-adapters/src/generative_flow_adapters/adapters/hypernetworks/hyperalign.py)
- each decoder token now produces a fixed-width factor vector
- `_split_hyper_factors(...)` reshapes it into:
  - `hyper_down`: `[batch, num_modules, a, r]`
  - `hyper_up`: `[batch, num_modules, r, b]`

Why:

- the paper uses a shared decoder output space, not one bespoke prediction head per adapted layer

### 4. Perception encoder from the pretrained U-Net down path

Paper:

- the perception encoder is built from the pretrained U-Net downsampling blocks
- it consumes the current latent `x_t`, timestep `t`, and conditioning `c`

Implementation:

- HyperAlign now always builds memory tokens from `module.input_blocks`
- it uses `_prepare_unet_runtime(...)` from the DynamiCrafter path to match backbone conditioning semantics
- the video latent is first converted to DynamiCrafter's internal layout:
  - from `[batch, channels, frames, height, width]`
  - to `[(batch * frames), channels, height, width]`
- each block output is spatially pooled and projected to decoder width

Why:

- the previous implementation had a fallback summary path
- for a paper-faithful replication, HyperAlign should rely on the pretrained U-Net encoder path, not a generic pooled summary MLP

### 5. Video-specific temporal memory tokens

Paper:

- the paper is written for image diffusion models
- it does not need to discuss preserving frame structure inside the hypernetwork encoder

Video adaptation used here:

- we do not collapse the frame dimension immediately
- for each input block:
  - spatial dimensions are mean-pooled
  - frame dimension is preserved
  - projected tokens are concatenated across blocks
- resulting decoder memory shape:
  - `[batch, num_blocks * frames, hidden_dim]`

Why:

- the previous implementation pooled across frames as well as space
- that erased most temporal structure, which is a bad fit for a video U-Net
- this is the main video-specific adaptation required to keep the paper's architecture meaningful in this repository

### 6. Zero query tokens and positional encodings

Paper:

- the decoder consumes zero tokens with positional encodings

Implementation:

- the adapter now uses:
  - zero query tokens as buffers
  - sinusoidal positional encodings over module tokens
- encoder memory tokens also receive sinusoidal positional encodings

Why:

- the previous implementation used a learned query parameter tensor
- that was close in spirit, but not an exact architectural match

### 7. HyperAlign-S / HyperAlign-I / HyperAlign-P

Paper:

- `S`: regenerate LoRA every denoising step
- `I`: generate LoRA once at the beginning and reuse it
- `P`: regenerate LoRA at a small number of stage boundaries

Implementation:

- added `update_mode` support in [hyperalign.py](/Users/lukasbierling/Documents/thesis-uva/code/generative-flow-adapters/src/generative_flow_adapters/adapters/hypernetworks/hyperalign.py)
- supported aliases:
  - `stepwise`, `s`, `hyperalign-s`
  - `initial`, `i`, `hyperalign-i`
  - `piecewise`, `p`, `hyperalign-p`
- added `piecewise_progress_markers`
- caching logic tracks:
  - last timestep
  - batch signature
  - cached hyper factors
  - cached stage index

How new trajectory detection works:

- in normal sampling, timesteps descend
- when a later call arrives with a larger timestep than the previous call, the adapter treats that as a new denoising trajectory and refreshes the cache

Why:

- the old implementation only supported the `S` behavior

### 8. Per-frame expansion of dynamic LoRA

Problem discovered during verification:

- HyperAlign predicts one set of LoRA factors per sample
- video attention layers often operate on `batch * frames`

Implementation:

- generalized `_apply_batched_low_rank(...)` in [common.py](/Users/lukasbierling/Documents/thesis-uva/code/generative-flow-adapters/src/generative_flow_adapters/adapters/low_rank/common.py)
- dynamic factors are now allowed to divide the active batch dimension evenly
- if the effective batch is `batch * frames`, sample-level factors are repeated across frames

Why:

- this is required for the HyperAlign factors to line up with DynamiCrafter's per-frame internal layout

### 9. Shared conditioning-agnostic embedding path with DynamiCrafter

Problem:

- the DynamiCrafter output adapter already supported a generic `cond["embedding"]` path
- HyperAlign was still building its encoder memory with `adapter=None`
- that meant HyperAlign ignored the conditioning-agnostic embedding logic used by the main DynamiCrafter adapter path

Implementation:

- added [dynamicrafter_conditioning.py](/Users/lukasbierling/Documents/thesis-uva/code/generative-flow-adapters/src/generative_flow_adapters/adapters/dynamicrafter_conditioning.py) as a shared helper module
- moved the following logic into the shared helper:
  - optional conversion of `step_size` into `fs`
  - optional step-level embedding
  - combination of base `embedding` and step-level embedding
- updated [dynamicrafter.py](/Users/lukasbierling/Documents/thesis-uva/code/generative-flow-adapters/src/generative_flow_adapters/adapters/output/dynamicrafter.py) to use the shared helper
- updated [hyperalign.py](/Users/lukasbierling/Documents/thesis-uva/code/generative-flow-adapters/src/generative_flow_adapters/adapters/hypernetworks/hyperalign.py) to use the same helper before calling `_prepare_unet_runtime(...)`
- HyperAlign now:
  - stores `cond_dim`, `cond_hidden_dim`, and `use_adapter_conditioning`
  - initializes adapter-conditioning projection/fusion modules with `_prepare_adapter_conditioning(...)`
  - passes `adapter=self` into `_prepare_unet_runtime(...)`

Result:

- HyperAlign now accepts the same generic DynamiCrafter-style conditioning payload as the output adapter
- if `cond["embedding"]` is present, it is fused into the timestep embedding path
- optional step-level conditioning is handled consistently across both adapters
- the conditioning logic is no longer duplicated

## Config Changes

The canonical HyperAlign config is now [configs/diffusion_hyperalign_action.yaml](/Users/lukasbierling/Documents/thesis-uva/code/generative-flow-adapters/configs/diffusion_hyperalign_action.yaml).

Important changes:

- switched from `provider: dummy` to `provider: dynamicrafter`
- removed generic `adapter.target_modules`
- added paper-relevant settings under `adapter.extra`:
  - `aux_down_dim`
  - `aux_up_dim`
  - `num_decoder_layers`
  - `num_decoder_heads`
  - `update_mode`
  - `piecewise_progress_markers`

The current default paper-style values are:

- `rank = 4`
- `aux_down_dim = 16`
- `aux_up_dim = 16`
- `hidden_dim = 128`, which matches `rank * (aux_down_dim + aux_up_dim)`
- `num_decoder_layers = 4`
- `num_decoder_heads = 8`

## Validation

Added standard-library tests in [tests/test_hyperalign_architecture.py](/Users/lukasbierling/Documents/thesis-uva/code/generative-flow-adapters/tests/test_hyperalign_architecture.py).

Verified behaviors:

- exact paper target inventory is selected
- `B_aux` equivalent is zero-initialized
- decoder output has the fixed shared shape required by the paper
- reshaped hyper factors have the expected `A_hyper` / `B_hyper` dimensions
- `initial` mode reuses factors until a new trajectory starts
- `piecewise` mode refreshes only when the configured stage changes
- HyperAlign accepts the same generic DynamiCrafter `embedding` and optional step-level payload shapes as the output adapter

Command used:

```bash
PYTHONPATH=src python -m unittest tests.test_hyperalign_architecture -v
```

## Remaining Gaps Relative To The Full Paper

These are still not implemented:

- reward objective from the paper
- preference-regularization objective from the paper
- paper training loop based on trajectory optimization
- exact paper evaluation protocol

There are also a few practical adaptations that are faithful in spirit but necessarily implementation-specific:

- video-aware memory tokenization instead of image-only encoder flattening
- trajectory cache reset inferred from timestep ordering because the existing framework does not expose explicit sampling-session boundaries

## Bottom Line

After these changes, the HyperAlign adapter in this repository now matches the paper architecture at the important structural points:

- attention-only LoRA targets
- auxiliary-factorized LoRA generation
- pretrained U-Net perception encoder
- transformer decoder with zero query tokens
- fixed shared decoder output head
- `S/I/P` update modes

The remaining missing part is the paper's training method, not the adapter architecture itself.

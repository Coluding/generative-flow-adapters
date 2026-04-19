# Generative Flow Adapters

Research scaffold for adapting frozen pretrained generative models into action-conditioned world models through plug-and-play adapters.

## Core idea

The repository centers on a single composition rule:

```python
f(x_t, t, c) = f_base(x_t, t) + Delta_phi(x_t, t, c)
```

`f_base` stays frozen. `Delta_phi` is the trainable adapter. The resulting model preserves the same forward interface across diffusion and flow matching backbones:

```python
prediction = model(x_t, t, cond)
```

## Design goals

- Treat adapters as first-class modules.
- Keep diffusion and flow matching under one `BaseGenerativeModel` interface.
- Build around wrapping existing backbones instead of reimplementing them.
- Make configuration the primary control surface.
- Leave clear extension points for shortcut conditioning, multimodal inputs, planning, and RL.

## Repository layout

```text
src/generative_flow_adapters/
  adapters/         Adapter interfaces and implementations grouped by family
  conditioning/     Condition encoders for actions, goals, and multimodal signals
  losses/           Shared objectives and consistency losses
  models/           Frozen base wrappers and adapted composition
  training/         Thin trainer and config-driven builders
src/external_deps/  Vendored third-party code with clear ownership boundary
configs/            Example experiment configurations
examples/           Small entrypoints for construction and smoke testing
```

## Current scope

This commit provides the foundation layer:

- abstract base model and adapter interfaces
- `AdaptedModel` wrapper for additive composition
- adapter variants for output-level, residual hidden-state style, hypernetwork, and LoRA-style weight injection
- condition encoders for tensor and multimodal inputs
- shared diffusion and flow losses plus shortcut consistency losses
- config dataclasses and factory functions
- soft integration hooks for Hugging Face `diffusers`
- dummy backbones for local smoke tests

## Quick start

Install in editable mode:

```bash
pip install -e .
```

Optional integration dependencies:

```bash
pip install -e .[diffusers]
pip install -e .[dynamicrafter]
```

Build a model from config:

```bash
python examples/build_model.py --config configs/diffusion_lora_action.yaml
```

DynamiCrafter starting point (vendored AVID architecture):

```bash
python examples/build_model.py --config configs/diffusion_output_dynamicrafter.yaml
```

HyperAlign paper-replication starting point:

```bash
python examples/build_model.py --config configs/diffusion_hyperalign_action.yaml
```

UniCon-style Figure 3(d) hidden-state starting point:

```bash
python examples/build_model.py --config configs/diffusion_hidden_unicon_decoder.yaml
```

Run a short synthetic training smoke test:

```bash
python examples/training_test.py --config configs/flow_output_shortcut.yaml --steps 3
```

## Configuration shape

```yaml
model:
  type: diffusion
  provider: dummy
  feature_dim: 32
adapter:
  type: lora
  rank: 8
conditioning:
  type: action
  input_dim: 8
  output_dim: 32
training:
  loss: diffusion
```

`adapter.type: output` accepts `adapter.extra.architecture`:
- `affine` (default lightweight baseline)
- `dynamicrafter` (AVID/DynamiCrafter 3D UNet adapter; requires `adapter.extra.unet_config_path`)

`adapter.type: hidden_state` accepts `adapter.extra.architecture`:
- `residual` (default lightweight hidden-state baseline)
- `unicon` (paper-aligned Figure 3(d) decoder-part-focused UniCon for U-Net backbones)
- `replace_decoder` (paper Figure 3(e))
- `full_skip_controlnet` (paper Figure 3(c))

`adapter.type: hyper` accepts `adapter.extra.architecture`:
- `hyper_lora_simple` (lightweight baseline generating LoRA weights from pooled model state)
- `hyperalign` (paper-aligned HyperAlign architecture for DynamiCrafter video U-Nets: attention-only targets, auxiliary-factorized LoRA, perception encoder, transformer decoder, and `S/I/P` update modes)

HyperAlign implementation notes are documented in [docs/hyperalign-architecture-replication.md](/Users/lukasbierling/Documents/thesis-uva/code/generative-flow-adapters/docs/hyperalign-architecture-replication.md).

Output adapters are composed in [adapted_model.py](/Users/lukasbierling/Documents/thesis-uva/code/generative-flow-adapters/src/generative_flow_adapters/models/adapted_model.py), not inside the adapter itself. Supported `adapter.composition` modes are:
- `add`
- `replace`
- `avid_mask_mix`

## Extension points

- Add a new adapter in `src/generative_flow_adapters/adapters/` and register it in `adapters/factory.py`.
- Add a new conditioning encoder in `src/generative_flow_adapters/conditioning/encoders.py`.
- Add a new backbone wrapper in `src/generative_flow_adapters/models/base/`.
- Add a new objective in `src/generative_flow_adapters/losses/` and register it in `losses/registry.py`.

## Notes

- The LoRA path is implemented as a reusable injection mechanism for `nn.Linear` layers. It computes the adapter correction as the difference between the frozen base pass and a temporarily enabled low-rank pass.
- The `diffusers` integration is intentionally thin. The repository should wrap pretrained models and route tensor I/O through a shared interface rather than fork core model code.

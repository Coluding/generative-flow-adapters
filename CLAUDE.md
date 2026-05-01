# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands

```bash
# Install in editable mode
pip install -e .

# Install with optional dependencies
pip install -e .[diffusers]
pip install -e .[dynamicrafter]
pip install -e .[dev]

# Run all tests
pytest

# Run a single test file
pytest tests/test_hyperalign_architecture.py

# Run a specific test
pytest tests/test_hyperalign_architecture.py::TestHyperAlignAdapter::test_forward_shape

# Lint
ruff check src/

# Build a model from config
python examples/build_model.py --config configs/diffusion_lora_action.yaml

# Run synthetic training smoke test
python examples/training_test.py --config configs/flow_output_shortcut.yaml --steps 3
```

## Architecture Overview

This repository implements adapter-first composition for adapting frozen generative models into action-conditioned world models.

### Core Composition Rule

All models follow additive composition:
```python
prediction = base_model(x_t, t) + adapter(x_t, t, cond, base_output)
```

The base model stays frozen. Only the adapter and condition encoder are trained.

### Key Abstractions

**BaseGenerativeModel** (`models/base/interfaces.py`): Common interface for all backbones
- `forward(x_t, t, cond=None) -> prediction`
- `model_type`: "diffusion" or "flow" (determines loss semantics)
- `prediction_type`: "noise" or "velocity"

**Adapter** (`adapters/base.py`): All adapters implement
- `forward(x_t, t, cond, base_output=None) -> delta`
- `attach_base_model(base_model)`: called by AdaptedModel to give adapter access to backbone

**AdaptedModel** (`models/adapted_model.py`): Owns the composition logic
- Handles condition encoding, condition dropout, and output composition modes (add, replace, mask_mix)

### Adapter Families

Located in `src/generative_flow_adapters/adapters/`:

| Type | Factory Key | Description |
|------|-------------|-------------|
| Output | `output` | Direct residual on final output (affine, dynamicrafter, shortcut_direction) |
| Hidden State | `hidden_state` | Residual modulation at hidden states (residual, unicon, replace_decoder, full_skip_controlnet) |
| Hypernetwork | `hyper` | Generates context-dependent weights (hyperalign, hyper_lora_simple) |
| LoRA | `lora` | Low-rank weight injection into nn.Linear layers |

All adapters are registered and built through `adapters/factory.py:build_adapter()`.

### Configuration System

YAML configs map to dataclasses in `config.py`:
- `ExperimentConfig` contains `ModelConfig`, `AdapterConfig`, `ConditioningConfig`, `TrainingConfig`
- Load with `load_config(path)`, build experiment with `training/builders.py:build_experiment(config)`
- Unknown fields go into `extra` dict for architecture-specific parameters

### Loss Registry

`losses/registry.py:LossRegistry` selects loss by model type:
- `diffusion` → `diffusion_loss`
- `flow` / `flow_matching` → `flow_matching_loss`

Shortcut consistency losses are separate: `shortcut_direction`, `local_consistency`, `multistep_self_consistency`

### Backbone Providers

New backbones go in `models/base/`. Current providers:
- `dummy`: Lightweight MLP for testing
- `diffusers`: Hugging Face diffusers integration
- `dynamicrafter`: Video U-Net from DynamiCrafter/AVID

### Vendored Code

`src/external_deps/` contains vendored third-party code:
- `lvdm/`: Latent video diffusion modules from DynamiCrafter
- `avid_utils/`: AVID evaluation utilities

`backbones/dynamicrafter/` contains adapted DynamiCrafter model code.

## Extension Points

- New adapter: add in `adapters/`, register in `adapters/factory.py`
- New condition encoder: add in `conditioning/encoders.py`
- New backbone: add wrapper in `models/base/`, register in `models/base/factory.py`
- New loss: add in `losses/`, register in `losses/registry.py`
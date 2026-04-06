# Foundation Architecture

## Unifying abstraction

All generative backbones implement:

```python
BaseGenerativeModel.forward(x_t, t, cond=None) -> prediction
```

The model type determines only the semantics of the prediction:

- `diffusion`: predicted noise
- `flow`: predicted velocity

This keeps training and evaluation code independent from the backbone family.

## Adapter contract

All adapters implement:

```python
Adapter.forward(x_t, t, cond, base_output=None) -> delta
```

Adapters are first-class modules. The repository treats them as additive corrections on top of a frozen base model.

## Composition rule

`AdaptedModel` owns the composition:

```python
base = base_model(x_t, t, cond=None or encoded_cond)
delta = adapter(x_t, t, encoded_cond, base_output=base)
prediction = base + delta
```

This keeps the external interface stable while making adapter internals swappable.

## Conditioning path

Conditioning is encoded separately from the backbone:

- `action` and `goal`: MLP encoder
- `multimodal`: concatenation plus projection
- optional `horizon`: appended for shortcut conditioning

This separation is deliberate. The base model can stay strictly frozen while conditioning logic evolves independently.

## Implemented adapter families

- `OutputAdapter`: direct residual prediction on the final output tensor
- `ResidualConditioningAdapter`: hidden-state style residual modulation
- `HyperNetworkAdapter`: generates context-dependent weights for a small correction map
- `LoRAAdapter`: injects low-rank updates into selected `nn.Linear` layers and computes `delta` as the difference between frozen and temporarily enabled passes

## Training objectives

The base loss is selected by `LossRegistry`:

- `diffusion -> diffusion_loss`
- `flow -> flow_matching_loss`

Shortcut extensions are separate consistency penalties:

- `local_consistency_loss`
- `multistep_self_consistency_loss`

This avoids entangling the main prediction objective with rollout-specific experiments.

## Why this layout

- new backbones go under `models/base/`
- new adapters go under `adapters/`
- new modalities go under `conditioning/`
- new objectives go under `losses/`

The repository should grow by adding modules beside existing ones, not by rewriting the composition layer.

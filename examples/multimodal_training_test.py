"""Real-backbone smoke run for the multimodal compositional adapter.

Builds the *real* DynamiCrafter base (frozen prior + first-stage VAE) plus the
multi-stream model (video output adapter + per-modality heads + compositional
fusion), wires the video VAE codec to the base's first-stage encoder, and runs a
few ``MultiModalTrainer`` steps on a synthetic clip batch.

No dataset / checkpoint required: ``allow_missing_checkpoint`` in the config lets
the UNet build with random weights, and the batch is synthesised here. Swap the
synthetic batch for a real MetaWorld loader (+ a real checkpoint) to turn this
into an actual training run.

    python examples/multimodal_training_test.py \
        --config configs/multimodal_dynamicrafter.yaml --steps 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from generative_flow_adapters.multimodal.builders import (
    build_codecs,
    build_multimodal_experiment,
)
from generative_flow_adapters.multimodal.config import load_multimodal_config
from generative_flow_adapters.multimodal.preprocessor import MultiModalBatchPreprocessor
from generative_flow_adapters.multimodal.trainer import MultiModalTrainer


def _apply_dynamicrafter_safety(config) -> None:
    """Mirror examples/train_with_metaworld.py: tolerate a missing checkpoint /
    dummy concat condition so the base builds without the real weights."""
    model = config.base.model
    if model.provider.lower() != "dynamicrafter":
        return
    model.extra.setdefault("allow_dummy_concat_condition", True)
    ckpt = model.pretrained_model_name_or_path
    if ckpt and not Path(ckpt).exists():
        model.extra.setdefault("allow_missing_checkpoint", True)


def _synthetic_batch(config, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """One synthetic clip: pixel video + per-modality streams + action cond.

    Video is pixel space ``(B, 3, T, H, W)`` in [-1, 1] (the VAE codec encodes it
    to the latent the base denoises). Modality streams arrive in their raw shape;
    the codecs encode them into diffusion targets.
    """
    model = config.base.model
    t_len = int(model.extra.get("temporal_length", 16))
    latent_h = int(model.extra.get("latent_height", 16))
    latent_w = int(model.extra.get("latent_width", 16))
    # VAE downsamples 8x; pixel resolution = 8 * latent resolution.
    pix_h, pix_w = latent_h * 8, latent_w * 8
    action_dim = int(config.base.conditioning.conditions[0].input_dim)
    cond_extra = config.base.conditioning.extra
    context_tokens = int(cond_extra.get("context_tokens", 77))
    context_dim = int(cond_extra.get("context_dim", 1024))
    fs_value = int(getattr(config.base.data, "fs_value", 1) or 1)

    batch: dict[str, torch.Tensor] = {
        "video": torch.rand(batch_size, 3, t_len, pix_h, pix_w, device=device) * 2 - 1,
        "act": torch.randn(batch_size, action_dim, device=device),
        # Cross-attention context + frame-stride the DynamiCrafter base reads
        # directly off the cond dict. The real video preprocessor builds these
        # from the text/image encoder; synthesised here for the smoke run.
        "context": torch.randn(batch_size, context_tokens, context_dim, device=device),
        "fs": torch.full((batch_size,), fs_value, dtype=torch.long, device=device),
    }
    for spec in config.adapter_modalities:
        if spec.codec.lower() == "resize":
            # raw map larger than the codec target so the resize actually fires
            c = spec.feature_shape[0]
            batch[spec.name] = torch.randn(batch_size, c, 32, 32, device=device)
        else:
            batch[spec.name] = torch.randn(batch_size, *spec.feature_shape, device=device)
    return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    config = load_multimodal_config(args.config)
    _apply_dynamicrafter_safety(config)

    # Build base + multi-stream model + optimizer. VAE codecs are deferred here
    # (vae=None) and rebuilt below once we can reach the base's first-stage model.
    components = build_multimodal_experiment(config)
    model = components.model.to(device)

    vae = getattr(model.base_model, "first_stage_model", None)
    if vae is None and any(s.codec.lower() == "vae" for s in config.output_modalities):
        raise RuntimeError(
            "Config has a 'vae' codec but the base has no first_stage_model; "
            "set model.extra.load_first_stage_model: true."
        )
    codecs = build_codecs(config, vae=vae)

    condition_keys = tuple(c.key for c in config.base.conditioning.conditions) or ("act",)
    preprocessor = MultiModalBatchPreprocessor(
        config.output_modalities,
        codecs,
        video_preprocessor=None,
        condition_keys=condition_keys,
    )

    trainer = MultiModalTrainer(
        model,
        components.optimizer,
        config.base.training,
        config.output_modalities,
    )

    print(f"experiment={config.base.name}")
    print(f"provider={config.base.model.provider} device={device}")
    print(f"streams={[s.name for s in config.output_modalities]}")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable_params={n_trainable:,}")

    raw = _synthetic_batch(config, args.batch_size, device)
    for step in range(1, args.steps + 1):
        batch = preprocessor(raw, train=True)
        # `context`/`fs` bypass the preprocessor's float-cast (fs must stay int);
        # inject them straight into the cond dict the base reads.
        batch["cond"]["context"] = raw["context"]
        batch["cond"]["fs"] = raw["fs"]
        metrics = trainer.training_step(batch)
        parts = " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
        print(f"step={step}/{args.steps} {parts}")


if __name__ == "__main__":
    main()

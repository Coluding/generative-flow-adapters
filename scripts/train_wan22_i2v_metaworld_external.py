"""Train the AVID action adapter on a frozen Wan2.2-TI2V-5B base — **new
plug-and-play building approach** (``BaseVideoModel`` + ``AdaptedModel``).

Same experiment as ``scripts/train_wan22_i2v_metaworld.py`` (same
``configs/diffusion_wan22_avid_i2v_metaworld.yaml``, same AVID adapter, same
diffusion-forcing preprocessor and loss). The ONLY difference is how the frozen
base is built:

- old: provider ``wan2.2`` -> the vendored ``Wan22DiTWrapper`` (a hand-copied
  fork of the Wan code, with our own forward/sampling wiring);
- new: provider ``wan2.2_external`` -> ``WanTI2VVideoModel``, which wraps the
  upstream ``wan.WanTI2V`` from ``external_repos/wan22`` and exposes the strict
  ``BaseVideoModel`` interface (``encode``/``decode``/``denoise``/``generate``).
  The adapter composes at the ``denoise`` seam via ``AdaptedModel`` exactly as
  before — training is untouched; only the base object changed.

The base's own Wan2.2-VAE is reused for the pixel->latent encode (no second VAE
load). Generation-based eval (video panels, FID/FVD) is OFF by default here,
since that path still uses the legacy reimplemented sampler we are retiring;
held-out **loss** eval is unaffected. Pass ``--enable-eval-gen`` to re-enable.

Requires a CUDA device (the upstream WanTI2V pins ``cuda``).

Smoke run:

    python scripts/train_wan22_i2v_metaworld_external.py \
        --config configs/diffusion_wan22_avid_i2v_metaworld.yaml \
        --hdf5 ds/metaworld_corner2.hdf5 \
        --ckpt-dir ckpts/Wan2.2-TI2V-5B \
        --steps 5 --batch-size 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from generative_flow_adapters.config import load_config
from generative_flow_adapters.data import (
    Wan22DiffusionForcingPreprocessor,
    WanBatchPreprocessConfig,
    build_metaworld_clip_dataset,
)
from generative_flow_adapters.training import build_experiment
from generative_flow_adapters.training.trainer import Trainer

# Wan2.2-VAE spatial stride (vae_stride = (4, 16, 16)).
_WAN22_VAE_SPATIAL_STRIDE = 16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/diffusion_wan22_avid_i2v_metaworld.yaml")
    parser.add_argument("--hdf5", default="ds/metaworld_corner2.hdf5", help="Path to MetaWorld HDF5 file")
    parser.add_argument(
        "--ckpt-dir",
        default="ckpts/Wan2.2-TI2V-5B",
        help="Wan2.2-TI2V-5B checkpoint DIR (DiT safetensors + Wan2.2_VAE.pth + uncond_context.pt).",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--max-area",
        type=int,
        default=None,
        help="Pixel-area budget for the Wan-native aspect-preserving resize (best_output_size). "
        "Overrides config model.extra.max_area. Native Wan = 901120 (704*1280); lower it for less VRAM.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sampling", choices=["random", "exhaustive"], default="random")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--enable-eval-gen",
        action="store_true",
        help="Keep the config's generation-based eval (video panels + FID/FVD). Off by default "
        "because that path still uses the legacy sampler; held-out loss eval always runs.",
    )
    parser.add_argument("--eval-hdf5", default=None, help="Separate held-out MetaWorld HDF5 for loss eval.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Fraction of --hdf5 held out for loss eval via a random window split (0 disables). Overrides config.",
    )
    parser.add_argument("--eval-every", type=int, default=None, help="Held-out loss eval cadence (steps). Overrides config.")
    parser.add_argument("--eval-batches", type=int, default=None, help="Batches averaged per loss-eval cycle. Overrides config.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This script needs a CUDA device: the upstream WanTI2V base pins cuda.")
    device = "cuda"
    config = load_config(args.config)

    # --- NEW building approach: point the base provider at the external repo ---
    ckpt_dir = Path(args.ckpt_dir)
    if not (ckpt_dir / "Wan2.2_VAE.pth").exists():
        raise FileNotFoundError(f"Wan2.2_VAE.pth not found in {ckpt_dir}; pass --ckpt-dir with the full checkpoint.")
    if not list(ckpt_dir.glob("diffusion_pytorch_model*.safetensors")):
        raise FileNotFoundError(
            f"No Wan2.2 DiT safetensors in {ckpt_dir}. The external WanTI2V loads real weights via "
            "from_pretrained(ckpt_dir); a random base is not supported here."
        )
    config.model.provider = "wan2.2_external"        # -> WanTI2VVideoModel (BaseVideoModel)
    config.model.pretrained_model_name_or_path = str(ckpt_dir)   # WanTI2V takes the DIR
    config.model.extra["offload_model"] = False       # keep the DiT resident on GPU for training

    # Eval-cadence overrides (CLI wins over YAML). Loss eval only runs when a
    # cadence is set *and* a held-out loader exists.
    if args.eval_every is not None:
        config.training.eval_every_n_steps = args.eval_every
    if args.eval_batches is not None:
        config.training.eval_num_batches = args.eval_batches
    eval_hdf5 = args.eval_hdf5 or config.data.eval_hdf5
    val_fraction = args.val_fraction if args.val_fraction is not None else config.data.val_fraction
    want_eval = bool(config.training.eval_every_n_steps) and (eval_hdf5 is not None or val_fraction > 0.0)

    # Generation-based eval uses the legacy sampler we are retiring — off unless
    # explicitly requested. Held-out loss eval (above) is unaffected.
    if not args.enable_eval_gen:
        config.training.inference_every_n_steps = 0
        config.training.quality_metrics = []
        config.training.quality_dist_metrics = []

    # Latent geometry from the config drives both the VAE resize and the model.
    temporal_length = int(config.model.extra.get("temporal_length", 17))
    latent_height = int(config.model.extra.get("latent_height", 16))
    latent_width = int(config.model.extra.get("latent_width", 16))
    target_height = latent_height * _WAN22_VAE_SPATIAL_STRIDE
    target_width = latent_width * _WAN22_VAE_SPATIAL_STRIDE
    # Wan-native aspect-preserving resize budget (CLI wins over config). When set,
    # frames are sized via best_output_size to the base's native regime and the
    # fixed target_height/width above are ignored. Alignment grid = patch(2)*stride(16).
    max_area = args.max_area if args.max_area is not None else config.model.extra.get("max_area")
    max_area = int(max_area) if max_area is not None else None
    align = 2 * _WAN22_VAE_SPATIAL_STRIDE  # = 32

    # build_experiment now constructs AdaptedModel(WanTI2VVideoModel, AVID adapter).
    experiment = build_experiment(config)
    model = experiment.model.to(device)
    trainer = Trainer(
        experiment.model,
        experiment.optimizer,
        experiment.loss_fn,
        config.training,
        wandb_logger=getattr(experiment, "wandb_logger", None),
        checkpoint_manager=getattr(experiment, "checkpoint_manager", None),
    )

    # Reuse the base's own Wan2.2-VAE for the pixel->latent encode and eval decode
    # (no second VAE load).
    from generative_flow_adapters.models.base.wan import make_wan_decode_fn

    vae = model.base_model.wan.vae
    if trainer.wandb_logger is not None:
        trainer.wandb_logger.set_decode_fn(make_wan_decode_fn(vae))

    condition_keys = tuple(spec.key for spec in config.conditioning.conditions if spec.key != "step_level")
    timestep_scale = float(config.training.extra.get("flow_timestep_scale", 1000.0))
    cond_frames = int(config.training.extra.get("cond_frames", 1))
    cond_frames_dist = config.training.extra.get("cond_frames_dist")
    preprocessor = Wan22DiffusionForcingPreprocessor(
        vae=vae,
        config=WanBatchPreprocessConfig(
            target_height=target_height, target_width=target_width, timestep_scale=timestep_scale,
            max_area=max_area, align_h=align, align_w=align,
        ),
        condition_keys=condition_keys or ("act",),
        cond_frames=cond_frames,
        cond_frames_dist=cond_frames_dist,
    )

    translator, dataset = build_metaworld_clip_dataset(
        config.data,
        default_window_width=temporal_length,
        hdf5=args.hdf5,
        frame_stride=args.frame_stride,
        sampling=args.sampling,
    )

    eval_dataset = None
    train_dataset = dataset
    if want_eval and eval_hdf5 is not None:
        _, eval_dataset = build_metaworld_clip_dataset(
            config.data,
            default_window_width=temporal_length,
            hdf5=eval_hdf5,
            frame_stride=args.frame_stride,
            sampling=args.sampling,
        )
    elif want_eval:
        val_len = max(args.batch_size, int(len(dataset) * val_fraction))
        val_len = min(val_len, len(dataset) - args.batch_size)
        if val_len >= args.batch_size:
            generator = torch.Generator().manual_seed(args.seed)
            train_dataset, eval_dataset = torch.utils.data.random_split(
                dataset, [len(dataset) - val_len, val_len], generator=generator
            )
        else:
            print("  warning: dataset too small for the requested val split; eval disabled.")

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(dataset.sampling == "exhaustive"),
        num_workers=args.num_workers,
        drop_last=True,
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=True,
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"experiment={config.name}  (NEW BaseVideoModel building approach)")
    print(f"device={device}  base={type(model.base_model).__name__} (external wan.WanTI2V)")
    print(f"adapter={config.adapter.type}/{config.adapter.extra.get('backbone')}  cond_frames={cond_frames}")
    if max_area is not None:
        from generative_flow_adapters.data.wan_batch_preprocessor import best_output_size

        vid = dataset[0]["video"]  # [T, H, W, C]
        src_h, src_w = int(vid.shape[1]), int(vid.shape[2])
        ow, oh = best_output_size(src_w, src_h, align, align, max_area)
        lat_f = 1 + (temporal_length - 1) // 4
        tok = (oh // align) * (ow // align)  # patch tokens per frame (align == patch*stride)
        print(f"resize: source {src_w}x{src_h} -> Wan-native {ow}x{oh} px (max_area={max_area}, aligned to {align})")
        print(f"latent geometry: {temporal_length}f -> {lat_f}x{oh // _WAN22_VAE_SPATIAL_STRIDE}x{ow // _WAN22_VAE_SPATIAL_STRIDE}"
              f" (48-ch) | seq_len={lat_f * tok} tokens/clip")
    else:
        print(f"latent geometry: {temporal_length}f -> ({target_height}x{target_width} px, fixed) -> 48-ch Wan2.2 latent")
    print(f"params trainable={trainable:,} total={total:,} ({100.0 * trainable / max(total, 1):.2f}%)")
    print(f"dataset_size={len(dataset)} | steps={args.steps} batch_size={args.batch_size}")
    print(f"eval: {'loss@every=' + str(config.training.eval_every_n_steps) if eval_loader is not None else 'disabled'}"
          f"  gen_eval={'on' if args.enable_eval_gen else 'off'}")

    trainer.train(
        loader=loader,
        max_steps=args.steps,
        preprocessor=preprocessor,
        log_every=args.log_every,
        eval_loader=eval_loader,
    )


if __name__ == "__main__":
    main()

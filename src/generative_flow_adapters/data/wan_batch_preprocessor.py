"""Batch preprocessor for the Wan2.1 flow-matching backbone.

Turns a MetaWorld clip batch (raw pixels + per-frame actions) into the
trainer-expected flow-matching batch. Unlike the diffusion path (where the
trainer builds ``x_t`` from ``target`` via ``q_sample``), the flow branch reads
``x_t`` and ``target`` straight from the batch — so this preprocessor builds the
consistent rectified-flow triple itself:

    z0   = WanVAE.encode(video)            # clean 16-ch latent
    noise ~ N(0, I)
    x_t  = (1 - t) * z0 + t * noise        # linear interpolation
    target (velocity) = noise - z0

Run with ``training.extra.use_batch_timesteps_for_flow: true`` so the trainer
uses the ``t`` paired with these tensors instead of resampling its own.

Conditioning is kept minimal and Wan-native: the frozen base sees only an
(optional) text ``context`` — None here, so it uses a null context — while the
trainable adapter is conditioned on the action via the ``action`` key.
"""

from __future__ import annotations

import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from generative_flow_adapters.data.latent_cache import LatentCache, latent_key

from generative_flow_adapters.data.batch_preprocessor import (
    _center_crop_to,
    resize_stretch,
    resize_with_pad,
)


def _lanczos_cover_crop_uint8(video: Tensor, out_h: int, out_w: int) -> Tensor:
    """PIL **LANCZOS** cover-resize + center-crop on raw uint8 frames, byte-for-byte
    matching the upstream Wan i2v conditioning-frame preprocessing
    (``img.resize(..., Image.LANCZOS)`` then center ``crop``).

    ``video``: ``[B, T, H, W, 3]`` uint8 (RGB) -> ``[B, T, out_h, out_w, 3]`` uint8.
    Done in uint8/PIL space *before* normalization (upstream resizes then
    ``to_tensor``), so the whole pipeline — filter, quantization, and
    normalization — matches what the frozen base sees at inference."""
    b, t, ih, iw, c = video.shape
    scale = max(out_w / iw, out_h / ih)
    rw, rh = round(iw * scale), round(ih * scale)  # upstream: round(iw*scale), round(ih*scale)
    x1 = (rw - out_w) // 2                          # upstream: (img.width  - ow)//2
    y1 = (rh - out_h) // 2                          # upstream: (img.height - oh)//2
    arr = video.detach().cpu().numpy()
    out = np.empty((b, t, out_h, out_w, c), dtype=np.uint8)
    for bi in range(b):
        for ti in range(t):
            img = Image.fromarray(arr[bi, ti])                 # HWC uint8 RGB
            img = img.resize((rw, rh), Image.LANCZOS)          # PIL takes (width, height)
            img = img.crop((x1, y1, x1 + out_w, y1 + out_h))   # center crop to (out_w, out_h)
            out[bi, ti] = np.asarray(img)
    return torch.from_numpy(out)


def best_output_size(w: int, h: int, dw: int, dh: int, expected_area: int) -> tuple[int, int]:
    """Largest ``(width, height)`` that fits within ``expected_area`` pixels, is
    divisible by ``(dw, dh)``, and best preserves the ``w/h`` aspect ratio.

    First-party port of Wan2.2's ``wan/utils/utils.best_output_size`` (Apache-2.0)
    — the resolution policy the upstream i2v pipeline uses. Rounding both sides to
    the ``(dw, dh)`` grid independently would drift the aspect ratio, so it tries
    snapping width-first and height-first and keeps whichever stays closest to the
    source ratio. ``dw``/``dh`` are ``patch_size · vae_stride`` (32 for Wan2.2
    TI2V), so the result tiles into whole latent/patch tokens with no padding.
    """
    ratio = w / h
    ow = (expected_area * ratio) ** 0.5
    oh = expected_area / ow
    # width-first
    ow1 = int(ow // dw * dw)
    oh1 = int(expected_area / max(ow1, 1) // dh * dh)
    ratio1 = ow1 / oh1
    # height-first
    oh2 = int(oh // dh * dh)
    ow2 = int(expected_area / max(oh2, 1) // dw * dw)
    ratio2 = ow2 / oh2
    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2, ratio2 / ratio):
        return ow1, oh1
    return ow2, oh2


class PromptContextProvider:
    """Maps a clip's ``task_name`` to its precomputed T5 text embedding so the
    frozen base can be conditioned on a task prompt during training **without
    loading T5**.

    Loads the table produced by ``precompute_prompt_contexts.py``. Two modes:

    - **Pool** (``table["pool"]`` present): a flat, task-independent list of prompt
      embeddings. Every clip samples one at random during training (element 0 at
      eval). No ``task_name`` needed. This is the simple "a set of prompts, sampled
      per run" setup — pool takes precedence over ``positive``.
    - **Task** (``table["positive"]`` = ``{task_name: emb, "__default__": ...}``):
      per-task prompt (or a per-task list of paraphrases). Looks up the clip's
      ``task_name``; unknown tasks fall back to ``__default__``.

    Either way returns a *list* of per-sample ``[Lᵢ, C]`` tensors (prompts differ in
    length; the Wan DiT pads each) — the ``context`` form ``denoise`` forwards.
    """

    def __init__(self, contexts_path: str) -> None:
        table = torch.load(contexts_path, map_location="cpu", weights_only=False)
        # Pool mode: a flat, task-independent set of prompt embeddings. When present
        # it takes precedence over `positive` — every clip samples from the same pool
        # (no task_name needed). Element 0 is the deterministic eval choice.
        self._pool = table.get("pool")
        self.pool_mode = isinstance(self._pool, (list, tuple)) and len(self._pool) > 0
        positive = table.get("positive")
        self._positive = dict(positive) if isinstance(positive, Mapping) else {}
        self._default = self._positive.get("__default__")
        if not self.pool_mode and not self._positive:
            raise ValueError(
                f"{contexts_path}: expected a 'pool' list or a non-empty 'positive' "
                "mapping; re-run precompute_prompt_contexts.py."
            )
        self.text_len = table.get("text_len")
        self.contexts_path = contexts_path

    def contexts_for(self, batch: Any, device: torch.device, sample: bool = True) -> list[Tensor]:
        """One ``[Lᵢ, C]`` embedding per clip. ``batch`` is a list of task_names
        (task mode) or an int batch size (pool mode). A random draw when ``sample``
        (training augmentation), the first/deterministic element at eval."""
        if self.pool_mode:  # task-independent: sample the shared pool per clip
            n = batch if isinstance(batch, int) else len(batch)
            pool = self._pool
            pick = lambda: pool[random.randrange(len(pool))] if sample and len(pool) > 1 else pool[0]
            return [pick().to(device) for _ in range(n)]
        task_names = batch
        if not isinstance(task_names, (list, tuple)):
            raise TypeError(
                "PromptContextProvider expects batch['task_name'] to be a list of "
                f"strings (got {type(task_names).__name__}); is the dataset emitting task_name?"
            )
        out: list[Tensor] = []
        for name in task_names:
            emb = self._positive.get(str(name), self._default)
            if emb is None:
                raise KeyError(
                    f"No prompt embedding for task {name!r} and no '__default__' in "
                    f"{self.contexts_path}. Add it to the prompts file and re-run precompute."
                )
            if isinstance(emb, (list, tuple)):  # a set of paraphrases -> pick one
                emb = emb[random.randrange(len(emb))] if sample and len(emb) > 1 else emb[0]
            out.append(emb.to(device))
        return out


@dataclass(slots=True)
class WanBatchPreprocessConfig:
    target_height: int | None = 256
    target_width: int | None = 256
    resize_mode: str = "stretch"  # "stretch" | "pad"
    # Aspect-preserving alternative to a fixed target_h/w: when `max_area` is set,
    # frames are resized to the largest (w, h) that fits `max_area` pixels, is
    # divisible by (align_w, align_h), and best matches the source aspect ratio
    # (Wan `best_output_size`), then center-cropped. This matches how the upstream
    # Wan i2v pipeline sizes frames — the base's native, patch/VAE-aligned regime —
    # instead of a fixed 256² that runs the frozen base off-distribution.
    max_area: int | None = None
    align_h: int = 32  # patch_size[1] * vae_stride[1] (Wan2.2 TI2V: 2 * 16)
    align_w: int = 32  # patch_size[2] * vae_stride[2]
    action_key: str = "action"
    action_aggregation: str = "sum"  # "sum" | "mean" | "last" over the clip's frames
    # Per-frame action tokens for the cross-attention conditioning path (in
    # addition to the aggregated `action_key`). Emitted as cond[action_seq_key]
    # with shape [B, L, A]. `action_seq_len`: None -> pass the raw per-frame
    # actions through unchanged; an int -> bin the frames into that many
    # contiguous tokens, summing the (delta) actions within each bin (the
    # aggregated `action` is just the 1-bin case).
    action_seq_key: str = "action_seq"
    action_seq_len: int | None = None
    # Per-frame (AVID-style) action conditioning. When True, the adapter's action
    # encoder is fed the per-frame binned sequence [B, L, A] under `action_key`
    # (instead of the aggregated [B, A]), so each latent frame is conditioned on
    # its own action — matching the original AVID action head — rather than one
    # summed vector broadcast across the clip. Requires `action_seq_len` = the
    # latent frame count so [B, L, A] aligns with the UNet's per-frame layout.
    # False (default) keeps the aggregated AdaLN-broadcast behaviour.
    action_per_frame: bool = False
    # Optional text conditioning for the frozen base: path to the precomputed
    # prompt-context table (from precompute_prompt_contexts.py). When set, each
    # clip's `task_name` is mapped to its cached T5 embedding and passed to the
    # base as cond["context"] — no T5 at train time. None -> frame-only (the base
    # uses its cached unconditional context, current behaviour).
    prompt_contexts_path: str | None = None
    task_key: str = "task_name"  # batch key holding the per-clip task name
    # Disk cache of VAE-encoded latents (see data/latent_cache.py). The VAE encode
    # is the per-step bottleneck and is pure recomputation (frozen VAE, deterministic
    # pixels), so `z0` is cached per clip. None -> always encode. `write_latent_cache`
    # controls filling it on a miss (training fills lazily; precompute forces it).
    latent_cache_dir: str | None = None
    write_latent_cache: bool = True
    sigma_min: float = 1e-5
    # The model is fed t = sigma * timestep_scale. Wan's pretrained convention is
    # t in [0, num_train_timesteps] = [0, 1000], so the interpolation coordinate
    # sigma in [0, 1] is scaled up before the DiT sees it. Feeding raw sigma
    # (scale=1) puts the frozen base off-distribution — see the fix ticket.
    timestep_scale: float = 1000.0
    # Training-time timestep shift (SD3/Wan convention):
    #     sigma' = s*sigma / (1 + (s-1)*sigma)
    # s > 1 concentrates the sampled noise levels toward sigma=1 (high noise),
    # where the future frames are least determined by x_t — i.e. where action /
    # dynamics information actually carries loss signal. This matches both Wan's
    # own pretraining regime (shift 5.0) and the ACWM-DiT training recipe
    # (shift 5.0 + mid/high-noise weighting); our previous adapter runs trained
    # on unshifted U(0,1), spending most supervision at low/mid sigma where the
    # frozen base is already near-optimal (see the 2026-07-21 per-sigma sweep:
    # flat base-parity at every sigma). Applied by the Wan2.2 diffusion-forcing
    # preprocessor during TRAINING only — eval batches stay U(0,1) so eval
    # losses remain comparable across runs. None or 1.0 -> no shift.
    sigma_shift: float | None = None


class WanBatchPreprocessor:
    """Encode MetaWorld clips to Wan latents and build a flow-matching batch."""

    def __init__(self, vae: Any, config: WanBatchPreprocessConfig, condition_keys: tuple[str, ...] = ()) -> None:
        self.vae = vae
        self.config = config
        # Dataset-emitted structured conditions (e.g. ("act",)); the first is
        # routed to the adapter's MLP action encoder under `action_key`.
        self.condition_keys = tuple(condition_keys)
        # Optional text conditioning for the frozen base (task prompt -> cached
        # T5 embedding). None -> frame-only (base uses its unconditional context).
        self.prompt_contexts: PromptContextProvider | None = (
            PromptContextProvider(config.prompt_contexts_path)
            if config.prompt_contexts_path
            else None
        )
        self.latent_cache: LatentCache | None = (
            LatentCache(config.latent_cache_dir) if config.latent_cache_dir else None
        )

    @property
    def device(self) -> torch.device:
        return self.vae.device

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def __call__(self, batch: Mapping[str, Any], train: bool = True) -> dict[str, Any]:
        raw_video = batch["video"]
        batch_size = int(raw_video.shape[0])

        # Wan-VAE encodes a list of [C, T, H, W] clips -> list of [48, f, h, w].
        # Cached per clip when a latent cache is configured. `_encode_z0` is
        # cache-first: on a full hit the raw frames are never resized/uploaded
        # (the resized latent already exists), skipping both the VAE and the
        # LANCZOS resize — the per-step cost collapses to a disk read.
        z0 = self._encode_z0(raw_video, batch, batch_size)

        noise = torch.randn_like(z0)
        # sigma in [0,1] is the interpolation coordinate; t fed to the model is
        # sigma * timestep_scale (Wan native = [0,1000]).
        sigma = torch.rand(batch_size, device=z0.device, dtype=z0.dtype).clamp_min(self.config.sigma_min)
        sigma_b = sigma.view(batch_size, *([1] * (z0.dim() - 1)))
        x_t = (1.0 - sigma_b) * z0 + sigma_b * noise
        target = noise - z0  # rectified-flow velocity (sigma-independent)
        t = sigma * self.config.timestep_scale

        cond = self._build_condition(batch, batch_size, train)
        # `x0` (clean latent) is kept alongside the velocity `target` so the
        # eval video logger can decode a correct ground-truth panel — decoding
        # `target` (= noise - z0) would render noise, not the source clip.
        return {"x_t": x_t, "t": t, "target": target, "x0": z0, "cond": cond}

    def _encode_z0(self, raw_video: Tensor, batch: Mapping[str, Any], batch_size: int) -> Tensor:
        """VAE-encode the batch's clips to ``z0`` ([B,48,f,h,w]), cache-first.

        ``raw_video`` is the *un-resized* dataset video ([B,T,H,W,3] uint8). Cache
        keys are built from the raw frame dims via ``_output_hw`` (the resized
        geometry is deterministic, so no resize is needed to look a clip up). Only
        the **misses** are then resized, uploaded and VAE-encoded — on a full hit
        the pixels are never touched. Misses are written back when
        ``write_latent_cache``."""
        if self.latent_cache is None:
            video = self._normalize_video(raw_video).to(device=self.device, dtype=torch.float32)
            self.last_encoded = batch_size
            return torch.stack(self.vae.encode([video[i] for i in range(batch_size)]), dim=0).float()

        # Predict the resized T×H×W without resizing (the fast path). The dataset
        # never emits non-uint8 video; if it did, fall back to resizing up front.
        standard = raw_video.dim() == 5 and raw_video.shape[-1] == 3 and raw_video.dtype == torch.uint8
        normalized: Tensor | None = None
        if standard:
            frames = int(raw_video.shape[1])
            out_h, out_w = self._output_hw(int(raw_video.shape[2]), int(raw_video.shape[3]))
        else:
            normalized = self._normalize_video(raw_video).to(device=self.device, dtype=torch.float32)
            frames, out_h, out_w = int(normalized.shape[2]), int(normalized.shape[3]), int(normalized.shape[4])

        keys = self._latent_keys(batch, batch_size, frames, out_h, out_w)
        z_list: list[Tensor | None] = [None] * batch_size
        miss_idx: list[int] = []
        for i, k in enumerate(keys):
            cached = self.latent_cache.get(k) if k is not None else None
            if cached is not None:
                z_list[i] = cached.to(self.device).float()
            else:
                miss_idx.append(i)
        self.last_encoded = len(miss_idx)  # how many this call had to run the VAE on (misses)
        if os.environ.get("GFA_DEBUG_CACHE") and keys:
            k0 = keys[0]
            present = (k0 in self.latent_cache) if k0 is not None else "NO-KEY(missing batch fields)"
            print(f"[cache] miss {len(miss_idx)}/{batch_size} | key0={k0!r} present={present}", flush=True)
        if miss_idx:
            # Resize + normalize + upload ONLY the missing clips: all of them on a
            # cold cache, none on a warm one (this whole block is skipped).
            if normalized is None:
                miss_video = self._normalize_video(raw_video[miss_idx]).to(device=self.device, dtype=torch.float32)
                encoded = self.vae.encode([miss_video[j] for j in range(len(miss_idx))])
            else:
                encoded = self.vae.encode([normalized[i] for i in miss_idx])
            for j, i in enumerate(miss_idx):
                z = encoded[j].float()
                z_list[i] = z
                if keys[i] is not None and self.config.write_latent_cache:
                    self.latent_cache.put(keys[i], z)
        return torch.stack([z for z in z_list if z is not None], dim=0)

    def _output_hw(self, src_h: int, src_w: int) -> tuple[int, int]:
        """Resized (H, W) a clip of source ``(src_h, src_w)`` would get from
        ``_normalize_video`` — computed WITHOUT resizing, so cache keys can be
        built from the raw frames. Mirrors the branches in ``_normalize_video``."""
        cfg = self.config
        if cfg.max_area is not None:
            out_w, out_h = best_output_size(src_w, src_h, cfg.align_w, cfg.align_h, cfg.max_area)
            return int(out_h), int(out_w)
        if cfg.target_height is None or cfg.target_width is None:
            return int(src_h), int(src_w)  # no resize
        return int(cfg.target_height), int(cfg.target_width)  # fixed target (pad/stretch)

    def _latent_keys(self, batch: Mapping[str, Any], batch_size: int, t: int, h: int, w: int) -> list[str | None]:
        """Per-sample cache keys from the clip's stable identity + its resized T×H×W.
        Returns ``None`` for a sample if the dataset didn't emit the identity fields
        (then it is always encoded, never cached)."""
        env, ep = batch.get("env_name"), batch.get("episode_idx")
        st, fs = batch.get("start_idx"), batch.get("frame_stride")
        if any(v is None for v in (env, ep, st, fs)):
            return [None] * batch_size

        def _item(seq: Any, i: int) -> Any:
            v = seq[i]
            return v.item() if isinstance(v, Tensor) else v

        keys: list[str | None] = []
        for i in range(batch_size):
            try:
                keys.append(latent_key(_item(env, i), _item(ep, i), _item(st, i), _item(fs, i), t, h, w))
            except Exception:
                keys.append(None)
        return keys

    @torch.no_grad()
    def precompute(self, batch: Mapping[str, Any]) -> tuple[int, int]:
        """Encode a batch's clips and write them to the latent cache (no flow-matching
        triple built). Used by the offline precompute pass. Returns
        ``(batch_size, num_newly_encoded)`` — the second is 0 when all were cache hits."""
        if self.latent_cache is None:
            raise RuntimeError("precompute() needs latent_cache_dir set on the preprocessor config.")
        raw_video = batch["video"]
        batch_size = int(raw_video.shape[0])
        self._encode_z0(raw_video, batch, batch_size)
        return batch_size, int(getattr(self, "last_encoded", 0))

    def _build_condition(self, batch: Mapping[str, Any], batch_size: int, train: bool = True) -> dict[str, Any]:
        cond: dict[str, Any] = {}
        keys = self.condition_keys or ("act",)
        for i, key in enumerate(keys):
            value = batch.get(key)
            if not isinstance(value, Tensor):
                continue
            agg = self._aggregate_action(value).to(device=self.device, dtype=torch.float32)
            # The first structured condition feeds the adapter's action encoder.
            out_key = self.config.action_key if i == 0 else key
            cond[out_key] = agg
            if i == 0:
                # Per-frame action tokens, binned to the latent temporal grid. Also
                # used by the cross-attention path via `action_seq_key`.
                seq = self._action_sequence(value).to(device=self.device, dtype=torch.float32)
                cond[self.config.action_seq_key] = seq
                # AVID-style per-frame conditioning: override the aggregated `agg`
                # so the encoder sees [B, L, A] and each latent frame is conditioned
                # on its own action (vs one summed vector broadcast across the clip).
                if self.config.action_per_frame:
                    cond[out_key] = seq
        # Text conditioning for the frozen base: task_name -> cached T5 embedding,
        # passed as cond["context"] (a list of per-sample [Lᵢ, C] tensors). The
        # adapter's condition encoder ignores this key (reads only its own).
        if self.prompt_contexts is not None:
            if self.prompt_contexts.pool_mode:
                # Task-independent: every clip samples the shared prompt pool.
                cond["context"] = self.prompt_contexts.contexts_for(batch_size, self.device, sample=train)
            else:
                task_names = batch.get(self.config.task_key)
                if task_names is None:
                    raise KeyError(
                        f"prompt_contexts_path is set but batch has no '{self.config.task_key}'. "
                        "Ensure the dataset emits task_name (MetaWorldTranslator does)."
                    )
                cond["context"] = self.prompt_contexts.contexts_for(task_names, self.device, sample=train)
        return cond

    def _aggregate_action(self, action: Tensor) -> Tensor:
        # action: [B, T, A] per-frame -> [B, A]. MetaWorld stores delta-actions,
        # so SUM matches the strided-window aggregation used elsewhere.
        if action.dim() == 2:  # already [B, A]
            return action
        if action.dim() != 3:
            raise ValueError(f"Expected action [B,T,A] or [B,A], got {tuple(action.shape)}")
        mode = self.config.action_aggregation
        if mode == "sum":
            return action.sum(dim=1)
        if mode == "mean":
            return action.mean(dim=1)
        if mode == "last":
            return action[:, -1]
        raise ValueError(f"Unknown action_aggregation: {mode!r}")

    def _action_sequence(self, action: Tensor) -> Tensor:
        # action: [B, T, A] per-frame -> [B, L, A] tokens for the cross-attention
        # conditioning path. Raw delta-actions (no normalization), matching the
        # aggregated path. `action_seq_len` None -> passthrough (L = T); an int
        # -> bin the T frames into that many contiguous tokens, summing deltas
        # within each bin (consistent with the "sum" aggregation).
        if action.dim() == 2:  # already [B, A] -> a single token
            return action.unsqueeze(1)
        if action.dim() != 3:
            raise ValueError(f"Expected action [B,T,A] or [B,A], got {tuple(action.shape)}")
        n = self.config.action_seq_len
        t = int(action.shape[1])
        if n is None or n <= 0 or n >= t:
            return action  # per-frame passthrough
        b, _, a = action.shape
        edges = torch.linspace(0, t, n + 1).round().long().tolist()
        bins = [
            action[:, edges[j] : edges[j + 1]].sum(dim=1) if edges[j + 1] > edges[j] else action.new_zeros(b, a)
            for j in range(n)
        ]
        return torch.stack(bins, dim=1)  # [B, n, A]

    def _normalize_video(self, video: Any) -> Tensor:
        if not isinstance(video, Tensor):
            raise TypeError(f"Expected tensor 'video', got {type(video).__name__}.")
        cfg = self.config

        # Wan-native path (max_area set): resize the RAW uint8 frames with PIL
        # LANCZOS + center-crop *before* normalizing — exactly the upstream i2v
        # preprocessing — so the base's conditioning-latent distribution is
        # identical at train and inference. (Falls through to the tensor/bicubic
        # branch below for non-uint8 input, which the dataset never emits.)
        if cfg.max_area is not None and video.dtype == torch.uint8:
            if video.dim() != 5 or video.shape[-1] != 3:
                raise ValueError(f"Expected uint8 video [B,T,H,W,3], got {tuple(video.shape)}")
            src_h, src_w = int(video.shape[2]), int(video.shape[3])
            out_w, out_h = best_output_size(src_w, src_h, cfg.align_w, cfg.align_h, cfg.max_area)
            video = _lanczos_cover_crop_uint8(video, out_h, out_w)  # [B,T,out_h,out_w,3] uint8
            normalized = video.to(dtype=torch.float32) / 127.5 - 1.0  # == to_tensor().sub(.5).div(.5)
            return normalized.permute(0, 4, 1, 2, 3).contiguous()  # [B,T,H,W,C]->[B,C,T,H,W]

        if video.dtype == torch.uint8:
            normalized = video.to(dtype=torch.float32) / 127.5 - 1.0
            normalized = normalized.permute(0, 4, 1, 2, 3).contiguous()  # [B,T,H,W,C]->[B,C,T,H,W]
        elif video.dim() == 5 and video.is_floating_point():
            normalized = video
        else:
            raise ValueError(f"Unsupported video tensor: shape={tuple(video.shape)} dtype={video.dtype}")

        if cfg.max_area is None and (cfg.target_height is None or cfg.target_width is None):
            return normalized

        batch, channels, frames = normalized.shape[:3]
        # Reshape to [B*T, C, H, W] for the 2D resize helpers.
        flat = normalized.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, *normalized.shape[3:])
        if cfg.max_area is not None:
            # Float fallback (non-uint8 input): tensor bicubic cover-crop.
            _, _, src_h, src_w = flat.shape
            out_w, out_h = best_output_size(src_w, src_h, cfg.align_w, cfg.align_h, cfg.max_area)
            resized = _center_crop_to(flat, out_h, out_w)
        elif cfg.resize_mode == "pad":
            out_h, out_w = cfg.target_height, cfg.target_width
            resized = resize_with_pad(flat, out_h, out_w)
        else:
            out_h, out_w = cfg.target_height, cfg.target_width
            resized = resize_stretch(flat, out_h, out_w)
        return resized.reshape(batch, frames, channels, out_h, out_w).permute(0, 2, 1, 3, 4)

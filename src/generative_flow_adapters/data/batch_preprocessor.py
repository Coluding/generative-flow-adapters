"""First-party port of ``LatentVisualDiffusion.get_batch_input``.

Takes a raw clip batch produced by ``TranslatedClipDataset`` (uint8 video,
per-frame actions, captions, etc.) and turns it into the dict shape the
trainer expects::

    {
        "target": (B, 4, T, h, w) latent,
        "cond": {
            "context":          (B, L, C) text embeddings (zero by default),
            "concat":           (B, 4, T, h, w) first-frame latent replicated,
            "act":              (B, T, A) actions, untouched,
            "fs":               (B,) long, frame stride / fps,
            "dropout_actions":  bool,
        },
    }

Mirrors the DynamiCrafter behavior:
- per-frame VAE encode with ``scale_factor = 0.18215``
- concat-channel conditioning = first-frame latent replicated across T
- classifier-free guidance dropout on text-only / image-only / both at
  ``uncond_prob`` each (15% total coverage at the default 5% setting)

"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from generative_flow_adapters.data.latent_encoder import VideoAutoencoderKL

CaptionEncoder = Callable[[list[str]], Tensor]


@dataclass(slots=True)
class BatchPreprocessConfig:
    target_height: int | None = None
    target_width: int | None = None
    resize_mode: str = "stretch"
    uncond_prob: float = 0.05
    cond_frame_index: int = 0
    rand_cond_frame: bool = False
    context_tokens: int = 77
    context_dim: int = 1024
    dropout_actions: bool = False
    include_concat: bool = True
    include_video: bool = False


class DynamiCrafterBatchPreprocessor:
    """Pixel-video clip batch -> latent batch with DynamiCrafter conditioning."""

    def __init__(
        self,
        vae: VideoAutoencoderKL,
        config: BatchPreprocessConfig | None = None,
        caption_encoder: CaptionEncoder | None = None,
        image_encoder: Callable[[Tensor], Tensor] | None = None,
        image_resampler: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self.vae = vae
        self.config = config or BatchPreprocessConfig()
        self.caption_encoder = caption_encoder
        # DynamiCrafter's cross-attention expects context = [text(77); image(T*16)].
        # When both `image_encoder` (OpenCLIP vision) and `image_resampler` (the
        # Resampler whose weights live in `image_proj_model.*` of the checkpoint)
        # are provided, we build the image branch and concatenate it. When
        # either is None, the preprocessor falls back to text-only context and
        # the UNet runs in its degraded "no image cross-attention" mode — the
        # rollout still produces something via the concat-channel image
        # conditioning, but fine detail will be blurry.
        self.image_encoder = image_encoder
        self.image_resampler = image_resampler

    @property
    def device(self) -> torch.device:
        return next(self.vae.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.vae.parameters()).dtype

    def __call__(self, batch: Mapping[str, Any], train: bool = True) -> dict[str, Any]:
        config = self.config
        device = self.device
        video = self._normalize_video(batch["video"]).to(device=device, dtype=self.dtype)
        z = self.vae.encode_video(video)
        batch_size = z.shape[0]
        frames = z.shape[2]

        prompt_mask, image_keep = self._cfg_masks(batch_size=batch_size, device=device, train=train)

        captions = batch.get("caption", [""] * batch_size)
        if isinstance(captions, str):
            captions = [captions] * batch_size
        context = self._encode_captions(captions=list(captions), batch_size=batch_size)
        null_context = self._encode_captions(captions=[""] * batch_size, batch_size=batch_size)
        # prompt_mask has shape (B, 1, 1); broadcasts directly against (B, L, C).
        context = torch.where(prompt_mask, null_context, context)

        # The conditioning frame index is shared by the concat branch and the
        # image cross-attention branch so both look at the same frame.
        cond_idx = int(config.cond_frame_index)
        if config.rand_cond_frame and train:
            cond_idx = int(torch.randint(0, frames, (1,)).item())

        # Image cross-attention branch: CLIP-image embed the conditioning
        # frame, run the Resampler to get T*16 image tokens, append to text.
        image_emb = self._encode_image_branch(
            video=video,
            cond_frame_index=cond_idx,
            image_keep=image_keep,
            frames=frames,
        )
        if image_emb is not None:
            context = torch.cat([context, image_emb], dim=1)

        cond: dict[str, Any] = {
            "context": context,
            "dropout_actions": bool(config.dropout_actions),
        }

        if config.include_concat:
            frame_latent = z[:, :, cond_idx, :, :].unsqueeze(2)
            # image_keep has shape (B, 1, 1, 1, 1); zero out dropped conditioning
            frame_latent = frame_latent * image_keep
            cond["concat"] = frame_latent.repeat(1, 1, frames, 1, 1)

        actions = batch.get("act")
        if isinstance(actions, Tensor):
            cond["act"] = actions.to(device=device, dtype=self.dtype)

        fs = self._extract_fs(batch=batch, batch_size=batch_size, device=device)
        if fs is not None:
            cond["fs"] = fs

        out: dict[str, Any] = {"target": z, "cond": cond}
        if config.include_video:
            out["video"] = video
        return out

    @torch.no_grad()
    def _encode_image_branch(
        self,
        *,
        video: Tensor,            # (B, 3, T, H, W) in [-1, 1]
        cond_frame_index: int,
        image_keep: Tensor,        # (B, 1, 1, 1, 1)
        frames: int,
    ) -> Tensor | None:
        """Run OpenCLIP vision + Resampler on the conditioning frame; return
        ``(B, T*16, 1024)`` image tokens for the UNet's hybrid cross-attention,
        or ``None`` when either module wasn't provided at construction.

        Mirrors AVID's training-time path: apply the CFG image-drop mask in
        *pixel* space before CLIP, so a dropped sample produces the same
        "all-zero pixels → CLIP → Resampler" embedding the base model saw
        during training (cleaner than zeroing the embedding directly).
        """
        if self.image_encoder is None or self.image_resampler is None:
            return None
        cond_pixels = video[:, :, cond_frame_index]                   # (B, 3, H, W) in [-1, 1]
        cond_pixels = cond_pixels * image_keep.view(-1, 1, 1, 1)      # broadcast-compatible mask
        tokens = self.image_encoder(cond_pixels)                      # (B, 257, 1280)
        image_emb = self.image_resampler(tokens)                      # (B, video_length*16, 1024)
        return image_emb.to(device=self.device, dtype=self.dtype)

    def _normalize_video(self, video: Any) -> Tensor:
        if not isinstance(video, Tensor):
            raise TypeError(f"Expected tensor 'video', got {type(video).__name__}.")
        if video.dtype == torch.uint8:
            # (B, T, H, W, C) -> (B, C, T, H, W), normalize to [-1, 1]
            normalized = video.to(dtype=torch.float32) / 127.5 - 1.0
            normalized = normalized.permute(0, 4, 1, 2, 3).contiguous()
        elif video.dim() == 5 and video.is_floating_point():
            normalized = video
        else:
            raise ValueError(f"Unsupported video tensor: shape={tuple(video.shape)} dtype={video.dtype}")

        config = self.config
        if config.target_height is None or config.target_width is None:
            return normalized
        batch, channels, frames, _, _ = normalized.shape
        flat = rearrange(normalized, "b c t h w -> (b t) c h w")
        if config.resize_mode == "stretch":
            resized = resize_stretch(flat, config.target_height, config.target_width)
        elif config.resize_mode == "pad":
            resized = resize_with_pad(flat, config.target_height, config.target_width)
        elif config.resize_mode == "crop":
            resized = _center_crop_to(flat, config.target_height, config.target_width)
        else:
            raise ValueError(
                f"Unknown resize_mode {config.resize_mode!r}; expected 'stretch', 'pad', or 'crop'."
            )
        return rearrange(resized, "(b t) c h w -> b c t h w", b=batch, t=frames)

    def _cfg_masks(self, *, batch_size: int, device: torch.device, train: bool) -> tuple[Tensor, Tensor]:
        config = self.config
        if not train or config.uncond_prob <= 0.0:
            prompt_mask = torch.zeros(batch_size, 1, 1, device=device, dtype=torch.bool)
            image_keep = torch.ones(batch_size, 1, 1, 1, 1, device=device, dtype=self.dtype)
            return prompt_mask, image_keep

        random_num = torch.rand(batch_size, device=device)
        uncond = float(config.uncond_prob)
        prompt_mask = (random_num < 2 * uncond).view(batch_size, 1, 1)
        image_drop = ((random_num >= uncond) & (random_num < 3 * uncond)).view(batch_size, 1, 1, 1, 1)
        image_keep = (1.0 - image_drop.to(self.dtype)).to(device=device)
        return prompt_mask, image_keep

    def _encode_captions(self, *, captions: list[str], batch_size: int) -> Tensor:
        if self.caption_encoder is None:
            return torch.zeros(
                batch_size,
                self.config.context_tokens,
                self.config.context_dim,
                device=self.device,
                dtype=self.dtype,
            )
        encoded = self.caption_encoder(captions)
        return encoded.to(device=self.device, dtype=self.dtype)

    def _extract_fs(self, *, batch: Mapping[str, Any], batch_size: int, device: torch.device) -> Tensor | None:
        fs = batch.get("frame_stride", batch.get("fps"))
        if fs is None:
            return None
        if isinstance(fs, Tensor):
            return fs.to(device=device, dtype=torch.long)
        if isinstance(fs, list):
            return torch.tensor(fs, device=device, dtype=torch.long)
        if isinstance(fs, (int, float)):
            return torch.full((batch_size,), int(fs), device=device, dtype=torch.long)
        return None


#   "stretch": direct bicubic resize (may distort aspect; default).
#   "pad":     resize keeping aspect, then letterbox-pad with -1.0.
#   "crop":    resize-to-cover, then center-crop.

def resize_stretch(frames: Tensor, target_h: int, target_w: int) -> Tensor:
    """Bicubic + antialias resize to (target_h, target_w), no aspect preservation.

    Inputs are expected in ``[-1, 1]`` (VAE convention). Bicubic + antialias
    is the same recipe the OpenCLIP image embedder uses for its 224x224
    preprocess; the clamp tames bicubic overshoot so pixels stay in the range
    the VAE was trained on.

    This is the right choice when content geometry matters less than filling
    the trained latent resolution — e.g. mapping square robotics frames onto
    DynamiCrafter's 320x512 training resolution. The horizontal stretch is
    consistently applied at training and inference, so the adapter learns the
    stretched distribution end-to-end.
    """
    return F.interpolate(
        frames,
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(-1.0, 1.0)


def resize_with_pad(frames: Tensor, target_h: int, target_w: int) -> Tensor:
    """Bicubic + antialias resize preserving aspect ratio, then pad to target.

    Inputs are expected in ``[-1, 1]`` (VAE convention). Bicubic + antialias
    gives sharper results than bilinear on both up- and down-scaling, and the
    clamp tames bicubic overshoot so pixels stay in the range the VAE was
    trained on.

    Pad fill is ``-1.0`` (pure black in [-1, 1] space) so the letterbox
    matches the appearance of real widescreen video with black bars — i.e. in
    the base model's training distribution. With ``0.0`` we'd be feeding it
    *mid-gray* bars, which the model has rarely seen.
    """
    _, _, ih, iw = frames.shape
    scale = min(target_h / ih, target_w / iw)
    new_h = max(int(round(ih * scale)), 1)
    new_w = max(int(round(iw * scale)), 1)
    resized = F.interpolate(
        frames,
        size=(new_h, new_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(-1.0, 1.0)
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    return F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=-1.0)


def _center_crop_to(frames: Tensor, target_h: int, target_w: int) -> Tensor:
    """Resize so the shorter side matches, then center-crop the longer side.

    Same interpolation policy as :func:`resize_with_pad`.
    """
    _, _, ih, iw = frames.shape
    aspect = iw / ih
    target_aspect = target_w / target_h
    if aspect < target_aspect:
        new_w = target_w
        new_h = max(int(round(target_w / aspect)), target_h)
    else:
        new_h = target_h
        new_w = max(int(round(target_h * aspect)), target_w)
    resized = F.interpolate(
        frames,
        size=(new_h, new_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(-1.0, 1.0)
    top = (new_h - target_h) // 2
    left = (new_w - target_w) // 2
    return resized[:, :, top : top + target_h, left : left + target_w]

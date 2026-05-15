"""OpenCLIP text and image embedders for DynamiCrafter conditioning.

Self-contained ``nn.Module`` wrappers around the ``open_clip`` library. We
read out the same intermediate tensors DynamiCrafter was trained against
(penultimate-layer text states, pre-projection image tokens) without pulling
in the vendored ``external_deps.lvdm`` tree.

The two encoders match the contract of
``FrozenOpenCLIPEmbedder`` / ``FrozenOpenCLIPImageEmbedderV2`` from
DynamiCrafter so existing checkpoints / Resampler weights stay compatible.
"""
from __future__ import annotations

import gc
from typing import Literal

import torch
from torch import Tensor, nn


DYNAMICRAFTER_OPEN_CLIP_ARCH = "ViT-H-14"
DYNAMICRAFTER_OPEN_CLIP_VERSION = "laion2b_s32b_b79k"
DYNAMICRAFTER_OPEN_CLIP_LAYER = "penultimate"
DYNAMICRAFTER_CONTEXT_LENGTH = 77

# CLIP image normalization (OpenAI / OpenCLIP standard preprocessing).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _require_open_clip():
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "OpenCLIP encoding requires `open_clip_torch`. "
            "Install with `pip install open_clip_torch`."
        ) from exc
    return open_clip


class OpenCLIPTextEmbedder(nn.Module):
    """Frozen OpenCLIP text tower returning last / penultimate hidden states.

    Tokenizes via the OpenCLIP tokenizer (context length 77), runs the text
    transformer up to the chosen depth, and returns the post-``ln_final``
    states ``[B, 77, ctx_dim]``. DynamiCrafter was trained on
    ``layer='penultimate'``.
    """

    def __init__(
        self,
        arch: str = DYNAMICRAFTER_OPEN_CLIP_ARCH,
        version: str = DYNAMICRAFTER_OPEN_CLIP_VERSION,
        layer: Literal["last", "penultimate"] = DYNAMICRAFTER_OPEN_CLIP_LAYER,
    ) -> None:
        super().__init__()
        open_clip = _require_open_clip()

        if layer not in ("last", "penultimate"):
            raise ValueError(f"layer must be 'last' or 'penultimate', got {layer!r}.")
        # layer_skip counts back from the end: 0 → last block, 1 → penultimate.
        self._layer_skip = 0 if layer == "last" else 1

        model, _, _ = open_clip.create_model_and_transforms(
            arch, pretrained=version, device=torch.device("cpu")
        )
        del model.visual  # text-only encoder.
        self.model = model
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, texts: list[str]) -> Tensor:
        open_clip = _require_open_clip()
        device = next(self.parameters()).device

        tokens = open_clip.tokenize(list(texts)).to(device)
        x = self.model.token_embedding(tokens)           # [B, 77, D]
        x = x + self.model.positional_embedding

        # open_clip >= 2.32 ships the text transformer with batch_first=True
        # (NLD); older releases expected LND. Permute only when needed.
        transformer = self.model.transformer
        batch_first = bool(getattr(transformer, "batch_first", False))
        if not batch_first:
            x = x.permute(1, 0, 2)
        stop_at = len(transformer.resblocks) - self._layer_skip
        for index, block in enumerate(transformer.resblocks):
            if index == stop_at:
                break
            x = block(x, attn_mask=self.model.attn_mask)
        if not batch_first:
            x = x.permute(1, 0, 2)
        return self.model.ln_final(x)


class OpenCLIPImageEmbedder(nn.Module):
    """Frozen OpenCLIP vision tower returning the pre-projection token sequence.

    Resizes input pixels to 224x224 with antialiased bicubic, rescales
    ``[-1, 1] → [0, 1]``, applies CLIP mean/std normalization, then runs the
    visual transformer and returns the 257-token sequence (CLS + 16x16 patches)
    at width 1280 for ViT-H/14. We skip ``ln_post``, pooling, and the
    projection head — the downstream DynamiCrafter Resampler consumes the
    raw 1280-dim tokens.
    """

    def __init__(
        self,
        arch: str = DYNAMICRAFTER_OPEN_CLIP_ARCH,
        version: str = DYNAMICRAFTER_OPEN_CLIP_VERSION,
    ) -> None:
        super().__init__()
        open_clip = _require_open_clip()

        model, _, _ = open_clip.create_model_and_transforms(
            arch, pretrained=version, device=torch.device("cpu")
        )
        if hasattr(model, "transformer"):
            del model.transformer  # vision-only encoder.
        self.visual = model.visual
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

        self.register_buffer("mean", torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_CLIP_STD).view(1, 3, 1, 1), persistent=False)

    def _preprocess(self, image: Tensor) -> Tensor:
        x = torch.nn.functional.interpolate(
            image.float(),
            size=(224, 224),
            mode="bicubic",
            align_corners=True,
            antialias=True,
        )
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std

    @torch.no_grad()
    def forward(self, image: Tensor) -> Tensor:
        if image.dim() != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected image with shape [B, 3, H, W]; got {tuple(image.shape)}.")
        device = next(self.parameters()).device
        x = self._preprocess(image.to(device))

        # Patch embedding — mirrors open_clip.VisionTransformer for the
        # non-`input_patchnorm` path, which is what ViT-H/14 uses.
        x = self.visual.conv1(x)                                       # [B, W, gh, gw]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)     # [B, gh*gw, W]
        cls = self.visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)                                  # [B, 1+gh*gw, W]
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)

        transformer = self.visual.transformer
        batch_first = bool(getattr(transformer, "batch_first", False))
        if not batch_first:
            x = x.permute(1, 0, 2)
        x = transformer(x)
        if not batch_first:
            x = x.permute(1, 0, 2)
        return x


def encode_with_openclip(
    texts: list[str],
    *,
    arch: str = DYNAMICRAFTER_OPEN_CLIP_ARCH,
    version: str = DYNAMICRAFTER_OPEN_CLIP_VERSION,
    layer: str = DYNAMICRAFTER_OPEN_CLIP_LAYER,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    del_after_use: bool = True,
) -> Tensor:
    """One-shot helper that builds :class:`OpenCLIPTextEmbedder`, encodes
    ``texts``, and frees the model. Returns ``[B, 77, ctx_dim]``.

    Use the class directly when you want to keep the encoder around across
    calls instead of paying the model-load cost every time.
    """
    target_device = torch.device(device)

    encoder = OpenCLIPTextEmbedder(arch=arch, version=version, layer=layer).to(target_device)
    try:
        embedding = encoder(list(texts)).detach()
        if dtype is not None:
            embedding = embedding.to(dtype=dtype)
        embedding = embedding.clone()
    finally:
        if del_after_use:
            del encoder
            _cleanup()
    return embedding


def encode_image_with_openclip(
    image: Tensor,
    *,
    arch: str = DYNAMICRAFTER_OPEN_CLIP_ARCH,
    version: str = DYNAMICRAFTER_OPEN_CLIP_VERSION,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    del_after_use: bool = True,
) -> Tensor:
    """One-shot helper that builds :class:`OpenCLIPImageEmbedder`, encodes
    ``image`` (expected in ``[-1, 1]``), and frees the model. Returns
    ``[B, 257, 1280]`` for ViT-H/14.
    """
    target_device = torch.device(device)
    encoder = OpenCLIPImageEmbedder(arch=arch, version=version).to(target_device)
    try:
        tokens = encoder(image.to(target_device)).detach()
        if dtype is not None:
            tokens = tokens.to(dtype=dtype)
        tokens = tokens.clone()
    finally:
        if del_after_use:
            del encoder
            _cleanup()
    return tokens


def build_dynamicrafter_resampler_from_checkpoint(
    checkpoint_path: str,
    *,
    dim: int = 1024,
    depth: int = 4,
    dim_head: int = 64,
    heads: int = 12,
    num_queries: int = 16,
    embedding_dim: int = 1280,
    output_dim: int = 1024,
    ff_mult: int = 4,
    video_length: int = 16,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Build the DynamiCrafter Resampler and load the ``image_proj_model.*``
    weights from the checkpoint. Defaults match ``dynamicrafter_512.yaml``.
    """
    from generative_flow_adapters.data._resampler import Resampler

    resampler = Resampler(
        dim=dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        num_queries=num_queries,
        embedding_dim=embedding_dim,
        output_dim=output_dim,
        ff_mult=ff_mult,
        video_length=video_length,
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    full_state = state.get("state_dict", state)
    prefixes = ("model.image_proj_model.", "image_proj_model.")
    weights: dict[str, Tensor] = {}
    for prefix in prefixes:
        candidates = {k[len(prefix):]: v for k, v in full_state.items() if k.startswith(prefix)}
        if candidates:
            weights = candidates
            break
    if not weights:
        raise RuntimeError(
            f"No `image_proj_model.*` keys in {checkpoint_path}; cannot rebuild the Resampler."
        )
    resampler.load_state_dict(weights, strict=True)
    for parameter in resampler.parameters():
        parameter.requires_grad = False
    return resampler.to(device).eval()

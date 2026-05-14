"""Precompute the OpenCLIP null-prompt embedding once and discard the model.

DynamiCrafter's frozen U-Net was pretrained against a specific OpenCLIP text
embedding (ViT-H-14 / laion2b_s32b_b79k, penultimate layer). When training
adapters without language supervision we still need to feed that "null prompt"
tensor into cross-attention — feeding zeros is out of distribution for the
frozen weights and matches what AVID does when ``use_language=False``: CLIP
still runs on ``""``.

Loading CLIP-H every forward pass is wasteful when every caption is the same
empty string. ``precompute_null_text_embedding`` runs the encoder exactly
once, returns the resulting tensor, and drops the encoder so it does not sit
in GPU memory during training. ``CachedNullCaptionEncoder`` is the matching
``CaptionEncoder`` that broadcasts the cached tensor to whatever batch size
the preprocessor asks for.
"""

from __future__ import annotations

import gc

import torch
from torch import Tensor


DYNAMICRAFTER_OPEN_CLIP_ARCH = "ViT-H-14"
DYNAMICRAFTER_OPEN_CLIP_VERSION = "laion2b_s32b_b79k"
DYNAMICRAFTER_OPEN_CLIP_LAYER = "penultimate"
DYNAMICRAFTER_CONTEXT_LENGTH = 77


def encode_with_openclip(
    texts: list[str],
    *,
    arch: str = DYNAMICRAFTER_OPEN_CLIP_ARCH,
    version: str = DYNAMICRAFTER_OPEN_CLIP_VERSION,
    layer: str = DYNAMICRAFTER_OPEN_CLIP_LAYER,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Encode a list of strings through OpenCLIP, return ``[B, 77, context_dim]``.

    Mirrors :class:`external_deps.lvdm.modules.encoders.condition.FrozenOpenCLIPEmbedder`
    but talks to ``open_clip`` directly so this module doesn't pull in
    ``kornia`` / ``transformers`` (which the vendored file imports for unused
    image-encoder classes).

    ``layer`` selects which OpenCLIP transformer layer to read out:
    ``"last"`` returns the final block, ``"penultimate"`` returns the second
    to last — DynamiCrafter was trained on ``penultimate``.

    The caller is responsible for freeing the model; we do it inside
    :func:`precompute_null_text_embedding`.
    """
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "OpenCLIP encoding requires `open_clip_torch`. "
            "Install with `pip install open_clip_torch`."
        ) from exc

    if layer not in {"last", "penultimate"}:
        raise ValueError(f"layer must be 'last' or 'penultimate', got {layer!r}.")
    layer_idx = 0 if layer == "last" else 1

    target_device = torch.device(device)
    model, _, _ = open_clip.create_model_and_transforms(
        arch, pretrained=version, device=torch.device("cpu")
    )
    del model.visual
    model = model.to(target_device).eval()
    for p in model.parameters():
        p.requires_grad = False

    # open_clip 3.x defaults the text transformer to batch_first=True; 2.x
    # used [L, B, D]. Detect and avoid an unnecessary permute either way.
    batch_first = bool(getattr(model.transformer, "batch_first", False))

    try:
        tokens = open_clip.tokenize(list(texts)).to(target_device)
        with torch.no_grad():
            x = model.token_embedding(tokens)
            x = x + model.positional_embedding
            if not batch_first:
                x = x.permute(1, 0, 2)
            stop_at = len(model.transformer.resblocks) - layer_idx
            for i, block in enumerate(model.transformer.resblocks):
                if i == stop_at:
                    break
                x = block(x, attn_mask=model.attn_mask)
            if not batch_first:
                x = x.permute(1, 0, 2)
            embedding = model.ln_final(x)
        embedding = embedding.detach().to(device=target_device)
        if dtype is not None:
            embedding = embedding.to(dtype=dtype)
        embedding = embedding.clone()
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return embedding


def precompute_null_text_embedding(
    *,
    arch: str = DYNAMICRAFTER_OPEN_CLIP_ARCH,
    version: str = DYNAMICRAFTER_OPEN_CLIP_VERSION,
    layer: str = DYNAMICRAFTER_OPEN_CLIP_LAYER,
    max_length: int = DYNAMICRAFTER_CONTEXT_LENGTH,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Load OpenCLIP, encode the empty string once, free the model.

    Returns a ``[1, max_length, context_dim]`` tensor on ``device``.
    """
    del max_length  # OpenCLIP context length is fixed at 77 by the tokenizer.
    return encode_with_openclip(
        [""], arch=arch, version=version, layer=layer, device=device, dtype=dtype
    )


class CachedNullCaptionEncoder:
    """``CaptionEncoder`` that returns the cached null embedding for every input.

    Plug an instance into :class:`DynamiCrafterBatchPreprocessor` to feed the
    pretrained null-prompt tensor through cross-attention without rerunning
    CLIP per batch.
    """

    def __init__(self, null_embedding: Tensor) -> None:
        if null_embedding.dim() != 3 or null_embedding.shape[0] != 1:
            raise ValueError(
                "null_embedding must have shape [1, tokens, dim], "
                f"got {tuple(null_embedding.shape)}."
            )
        self.null_embedding = null_embedding

    def __call__(self, captions: list[str]) -> Tensor:
        return self.null_embedding.expand(len(captions), -1, -1)

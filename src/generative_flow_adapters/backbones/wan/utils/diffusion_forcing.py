"""Diffusion-forcing helpers vendored from Wan2.2 (`wan/utils/utils.py`,
Apache-2.0).

Wan2.2 TI2V conditions on an observation frame *without* CLIP or channel
concat: it clamps the conditioning frame's clean latent into a masked region
and gives those tokens timestep 0, while the rest of the clip is noised at the
sampled timestep (per-token / "diffusion forcing"). ``masks_like`` builds the
``(keep, noise)`` mask pair that drives both the latent clamp and the per-token
timestep; ``best_output_size`` picks a VAE/patch-aligned output size.
"""

from __future__ import annotations

import torch


def masks_like(tensor, zero=False, generator=None, p=0.2):
    assert isinstance(tensor, list)
    out1 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    out2 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    if zero:
        if generator is not None:
            for u, v in zip(out1, out2):
                random_num = torch.rand(
                    1, generator=generator, device=generator.device).item()
                if random_num < p:
                    u[:, 0] = torch.normal(
                        mean=-3.5,
                        std=0.5,
                        size=(1,),
                        device=u.device,
                        generator=generator).expand_as(u[:, 0]).exp()
                    v[:, 0] = torch.zeros_like(v[:, 0])
                else:
                    u[:, 0] = u[:, 0]
                    v[:, 0] = v[:, 0]
        else:
            for u, v in zip(out1, out2):
                u[:, 0] = torch.zeros_like(u[:, 0])
                v[:, 0] = torch.zeros_like(v[:, 0])

    return out1, out2


def best_output_size(w, h, dw, dh, expected_area):
    # float output size
    ratio = w / h
    ow = (expected_area * ratio)**0.5
    oh = expected_area / ow

    # process width first
    ow1 = int(ow // dw * dw)
    oh1 = int(expected_area / ow1 // dh * dh)
    assert ow1 % dw == 0 and oh1 % dh == 0 and ow1 * oh1 <= expected_area
    ratio1 = ow1 / oh1

    # process height first
    oh2 = int(oh // dh * dh)
    ow2 = int(expected_area / oh2 // dw * dw)
    assert oh2 % dh == 0 and ow2 % dw == 0 and ow2 * oh2 <= expected_area
    ratio2 = ow2 / oh2

    # compare ratios
    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2,
                                                  ratio2 / ratio):
        return ow1, oh1
    else:
        return ow2, oh2

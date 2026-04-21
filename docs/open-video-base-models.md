# Open Video Base Models For Adapter-Based Research

Research date: 2026-04-20

## Goal

This report surveys open-source video generation base models that are realistic candidates for this repository's adaptation setup:

- frozen base model plus trainable adapter
- adapter families including output adapters, hypernetworks, LoRA-style updates, and UniCon/ControlNet-style conditioning
- text conditioning, image/video conditioning, or both
- practical model size ceiling of at most 7B parameters

The main question is not just "which open video model is strongest?" but "which open video model is the best frozen backbone for adapter research?"

## Executive recommendation

If we want one primary modern base model to invest in, the best choice is:

1. `Wan2.1-T2V-1.3B`

If we want one modern DiT/flow model and one control-friendly U-Net family in parallel, the best pair is:

1. `Wan2.1-T2V-1.3B`
2. `VideoCrafter2` or `DynamiCrafter`

If we specifically want a flow-matching-heavy research track, the best stack is:

1. `Wan2.1-T2V-1.3B`
2. `Open-Sora 1.2/1.3`
3. `Pyramid Flow miniFLUX`

If we specifically want the easiest path for UniCon/ControlNet-style hidden-state and skip-path adaptation, the best base is:

1. `VideoCrafter2` or `DynamiCrafter`

If we want the strongest open diffusion-transformer ecosystem under 7B, the best family is:

1. `CogVideoX-2B`
2. `CogVideoX-5B`

## What I considered

Included models satisfy all or most of the following:

- public code is available
- official checkpoints or model cards are available
- model scale is at or below 7B, or clearly below 7B by architecture/checkpoint size
- the base can be conditioned on text or image/video
- the architecture is plausible for frozen-backbone adapter research

Excluded from the final shortlist:

- `HunyuanVideo` because the public base is above the size ceiling
- `Mochi` because it is above the size ceiling
- `Open-Sora 2.0` because it is 11B
- `Waver 1.0` because the released model is 12B

## Shortlist

| Model | Family | Size | Native conditioning | Why it matters here | Main caveat |
| --- | --- | --- | --- | --- | --- |
| `Wan2.1-T2V-1.3B` | Flow matching DiT | 1.3B | Text | Best small modern base; strong quality/efficiency; official code and weights; adapter-friendly transformer blocks | Small model only covers T2V directly; strongest I2V is 14B |
| `Wan2.1-VACE-1.3B` | Flow matching DiT | 1.3B | Text + image/video/mask/reference | Best small multi-condition Wan variant if editing/reference conditioning matters from day 1 | More task-specific than plain T2V base |
| `CogVideoX-2B` | Diffusion transformer | 2B | Text | Best practical open baseline for secondary development; strong ecosystem; open fine-tuning tooling | 480p/6s base quality lower than newest flow-matching models |
| `CogVideoX-5B` | Diffusion transformer | 5B | Text; official 5B-I2V variants exist | Stronger quality than 2B while staying under the size limit | 5B model license is more restrictive than 2B |
| `Open-Sora 1.2/1.3` | Rectified flow / flow matching style video DiT | 1.0B to 1.1B | Text, plus image-to-video and extension support in the family | Best open research-first flow codebase; excellent for training/fine-tuning experiments | More research-oriented and less turnkey than Wan/CogVideoX |
| `Pyramid Flow miniFLUX` | Flow matching pyramid DiT | exact count not published; appears safely below 7B from checkpoint size | Text and text-conditioned I2V | Best explicitly flow-matching video-first architecture after Wan/Open-Sora | Less standard ecosystem than Wan/CogVideoX/Open-Sora |
| `VideoCrafter2` | Latent video diffusion U-Net | not explicitly published; clearly SD/U-Net scale and far below 7B | Text and image | Best classic latent video diffusion family for ControlNet/UniCon-style work | Research/non-commercial license |
| `DynamiCrafter` | Latent video diffusion U-Net | inherited from VideoCrafter1 scale; well below 7B | Image + text dynamics prompt | Best immediate fit with this repo because DynamiCrafter support already exists locally | Not a general text-to-video base |
| `Allegro` / `Allegro-TI2V` | DiT video diffusion | 2.8B DiT + 175M VAE | Text, and text+image for TI2V | Strong mid-size open T2V/TI2V choice with released fine-tuning code | Newer ecosystem, less battle-tested than CogVideoX |
| `Vchitect-2.0-2B` | Diffusion transformer | 2B | Text | Strong 2B T2V alternative with open code/checkpoint | Weights are gated behind Hugging Face access terms |
| `LTX-Video 2B` | Latent diffusion / DiT-style I2V base | 2B line | Prompt + image/video conditions | Excellent for multi-keyframe and video-conditioned control experiments | Official positioning is mostly I2V/video-conditioned, not pure T2V |
| `Stable Video Diffusion` | Latent video diffusion | well below 7B by architecture | Image | Strong lightweight image-to-video baseline with very simple latent stack | Not text-native |
| `Latte-1` | Latent diffusion transformer | exact total not emphasized; transformer checkpoint indicates a mid-size model under the 7B ceiling | Text | Clean academic DiT baseline with training code and diffusers support | Older performance tier |

## Best choices by research objective

### 1. Best single primary base: `Wan2.1-T2V-1.3B`

Why it is the best overall choice:

- It is small enough to be practical.
- It is modern enough to still matter.
- It is explicitly based on flow matching within a diffusion-transformer setup.
- It uses a strong video VAE and multilingual T5 conditioning.
- It already has a family around it, not just one checkpoint.
- The official stack already supports Diffusers and a broader model suite including `VACE-1.3B`.

Why it is a good fit for adapter research:

- Output adapters are trivial because the base exposes a standard prediction interface.
- Hypernetwork and LoRA-style experiments fit naturally because the backbone is transformer-heavy.
- Conditional modulation can be attached at cross-attention, time-conditioning MLPs, or residual block inputs.
- A Wan-based project would let us evaluate both a simple base (`T2V-1.3B`) and a more condition-rich sibling (`VACE-1.3B`) inside one family.

Important nuance:

- Classic U-Net-style ControlNet is not the natural form here.
- For Wan, the better analogue is a DiT-native control branch: extra condition tokens, a zero-initialized residual side branch, or transformer-block modulation.
- So Wan is the best long-term foundation model, but not the easiest first ControlNet replica.

## 2. Best small flow-matching research base: `Open-Sora 1.2/1.3`

Why it stands out:

- It is one of the cleanest open research codebases for video rectified flow / flow-style training.
- The official reports are unusually explicit about the 3D-VAE, timestep sampling, conditioning, and adaptation path.
- The released STDiT weights are around 1B, which is ideal for adapter experiments.

Why it is valuable here:

- It maps very well to the repository abstraction that unifies diffusion and flow through `BaseGenerativeModel.forward(x_t, t, cond)`.
- It is better than most alternatives if we want to study how adapter mechanisms transfer from diffusion to flow matching with minimal conceptual mismatch.
- It provides an open path for text-to-video, image-to-video, and video extension.

Why it is not my top single recommendation:

- It is more research-first than productized.
- Wan is currently the more compelling small practical base.

## 3. Best diffusion-transformer ecosystem pick: `CogVideoX-2B` and `CogVideoX-5B`

Why this family matters:

- Official repo, model cards, fine-tuning support, diffusers integration, and a separate `CogKit` toolkit all exist.
- The 2B model is explicitly positioned for "secondary development."
- The family spans budget and stronger-quality checkpoints without leaving the size budget.

Why it is strong for adapters:

- Transformer linear layers are straightforward LoRA/hypernetwork targets.
- Text/video fusion is already handled via expert-transformer and cross-modal blocks.
- It is a strong benchmark family if we want our adapters to generalize beyond one backbone.

Why it is not my first choice over Wan:

- The 2B/5B family is diffusion, not flow matching.
- It is still a very strong baseline family, but Wan is the cleaner answer to the user's explicit request to include flow matching models and modern small bases.

## 4. Best for UniCon/ControlNet-style adaptation: `VideoCrafter2` and `DynamiCrafter`

Why these are special:

- They are classic latent video diffusion U-Net families.
- Hidden-state replacement, decoder-part adaptation, skip-branch control, and ControlNet-style side networks are much more natural on these models than on pure DiTs.
- This repository already contains DynamiCrafter-related integration and a HyperAlign replication built around that family.

Why I would keep them even if Wan is the main base:

- They are the easiest place to validate the hidden-state and full-skip control ideas already present in this repo.
- They reduce implementation risk for UniCon-like experiments.
- They are still useful even if they are not the strongest SOTA generators anymore.

Practical recommendation:

- If immediate experiments matter, start on `DynamiCrafter`.
- If the goal is a more general open T2V U-Net base, use `VideoCrafter2`.

## 5. Best explicitly flow-matching video-first alternative: `Pyramid Flow miniFLUX`

Why it is important:

- It is explicitly a video generation method based on flow matching.
- It supports text-to-video and text-conditioned image-to-video.
- It includes training code for its VAE and DiT fine-tuning.

Why it is attractive here:

- It is not just "a diffusion model that can be adapted for flow-style research"; flow matching is central to the method.
- The pyramid structure is interesting for adapter placement because conditioning can be injected at multiple resolution/noise stages.

Main downside:

- It has a smaller community and a less standardized downstream ecosystem than Wan, CogVideoX, or Open-Sora.

## 6. Best multi-condition video-conditioned base under 7B: `LTX-Video 2B`

Why it is useful:

- Official support exists for image-to-video, multi-keyframe conditioning, video extension, video-to-video, and mixed image/video conditions.
- The 2B line stays in the practical size range.
- The model family has already released ICLoRA-style condition-specific variants, which is highly relevant to adapter research.

Why I am not ranking it above Wan/CogVideoX/Open-Sora:

- The official positioning is centered on image/video-conditioned generation rather than being the cleanest general T2V base.
- If the project's emphasis becomes reference-conditioned video generation, it should move much higher.

## 7. Best additional mid-size open alternatives: `Allegro` and `Vchitect-2.0-2B`

`Allegro`:

- Very good if we want a modern open T2V/TI2V DiT around 3B.
- Training and fine-tuning code are released.
- It is a strong candidate if Wan or CogVideoX do not fit some implementation detail.

`Vchitect-2.0-2B`:

- Strong 2B T2V transformer alternative.
- Open code and official checkpoint exist.
- Worth keeping as a backup benchmark, especially if we want another 2B DiT outside CogVideoX.

Neither beats Wan as the main modern under-7B pick.

## 8. Good lightweight or legacy baselines: `Stable Video Diffusion`, `Latte-1`

These are not my top picks, but they are useful:

- `Stable Video Diffusion` is a clean image-to-video latent diffusion baseline with a simple ecosystem and good adapter friendliness.
- `Latte-1` is a clean academic latent diffusion transformer baseline with released training code and diffusers support.

They are better viewed as baselines than as the main research target.

## Architecture fit for our adapter families

### Output adapters

All shortlisted models are compatible.

This is the easiest family to support across:

- U-Net video diffusion models
- DiT video diffusion models
- flow-matching video transformers

If we want the broadest benchmark matrix, output adapters let us compare nearly everything in this report.

### Hypernetwork adapters

Best fits:

- `Wan2.1`
- `CogVideoX`
- `Open-Sora`
- `Allegro`
- `Vchitect-2.0`
- `LTX-Video`

Reason:

- Transformer-heavy models expose many `nn.Linear`-like projection points where dynamic low-rank or generated residual weights are natural.

Still viable, but less clean:

- `VideoCrafter2`
- `DynamiCrafter`
- `Stable Video Diffusion`

Reason:

- Hypernetworks can work on U-Nets, but targeting is more heterogeneous and tends to become more implementation-specific.

### UniCon / hidden-state replacement / ControlNet-style control

Best fits:

- `VideoCrafter2`
- `DynamiCrafter`
- `Stable Video Diffusion`

Reason:

- U-Net encoder-decoder structure and skip paths make these models the most natural hosts for hidden-state adapters, decoder replacement, or full-skip control branches.

Possible, but should be redesigned for DiTs:

- `Wan2.1`
- `CogVideoX`
- `Open-Sora`
- `Pyramid Flow`
- `Allegro`
- `Vchitect-2.0`
- `LTX-Video`

Reason:

- For these models, I would not literally port U-Net ControlNet.
- I would implement a DiT-native control analogue:
- extra conditioning token streams
- zero-initialized residual side blocks
- attention biasing or modulation
- per-block feature injection

That is still valid adapter research, but it is not the same architecture as classic ControlNet.

## Parameter-size notes

Models with explicit official size statements:

- `Wan2.1-T2V-1.3B`
- `Wan2.1-VACE-1.3B`
- `CogVideoX-2B`
- `CogVideoX-5B`
- `Open-Sora 1.2/1.3` around `1.0B` to `1.1B`
- `Allegro` with `2.8B` DiT and `175M` VAE
- `Vchitect-2.0-2B`
- `LTX-Video` 2B line

Models where the repo/model card does not foreground a single total parameter number, but the architecture is still clearly under the 7B ceiling:

- `VideoCrafter2`
- `DynamiCrafter`
- `Stable Video Diffusion`
- `Latte-1`
- `Pyramid Flow miniFLUX`

For `Pyramid Flow miniFLUX`, the exact count is not prominently stated in the official repo/model card. The 768p diffusion-transformer checkpoint size strongly suggests it remains safely below 7B, but this is an inference from checkpoint size rather than a published parameter count.

## Licensing and usability caveats

This matters for actual research use:

- `Wan2.1`: Apache-2.0 and broadly usable.
- `CogVideoX-2B`: Apache-2.0.
- `CogVideoX-5B`: open but under a custom model license.
- `Open-Sora`: Apache-2.0.
- `Pyramid Flow miniFLUX`: Apache-2.0 model card, MIT code repo.
- `Allegro`: Apache-2.0.
- `Vchitect-2.0`: Apache-2.0 code, but weights are gated on Hugging Face.
- `VideoCrafter2`: research/non-commercial.
- `DynamiCrafter`: research/non-commercial.
- `Stable Video Diffusion`: Stability community license, not a simple Apache-style open license.

If license flexibility matters, `Wan2.1`, `Open-Sora`, `CogVideoX-2B`, `Allegro`, and `Pyramid Flow` are safer starting points than `VideoCrafter2`, `DynamiCrafter`, or `Stable Video Diffusion`.

## Representation autoencoder angle

You asked to also consider newer models that use representation autoencoders.

What I found:

- I did not find a strong, clearly open, code-plus-checkpoint video base under 7B that is explicitly positioned as a representation-autoencoder video model in the same way recent image RAE papers are.
- The closest practical open candidates are modern latent-video systems with stronger learned video compression backbones:
- `Wan2.1` with `Wan-VAE`
- `Open-Sora 1.2/1.3` with 3D-VAE
- `CogVideoX` with a 3D causal VAE
- `Pyramid Flow` with a causal video VAE
- `LTX-Video` with a strong compressed latent stack

So if the real goal is "newer latent representation bottlenecks rather than old pixel-space or weak-autoencoder baselines," these models already cover that direction well.

## Final recommendation

### Recommended portfolio

If we want a serious research portfolio rather than one model:

1. `Wan2.1-T2V-1.3B` as the main modern small foundation model.
2. `VideoCrafter2` or `DynamiCrafter` as the main UniCon/ControlNet-friendly U-Net family.
3. `CogVideoX-2B` as the strongest open diffusion-transformer baseline for secondary development.
4. `Open-Sora 1.2/1.3` as the cleanest open research flow/rectified-flow codebase.

### If forced to pick only one

Pick `Wan2.1-T2V-1.3B`.

Why:

- It satisfies the size constraint cleanly.
- It is modern.
- It includes flow matching.
- It is practically usable.
- It has official code and weights.
- It lives in a larger family that already extends toward richer conditioning.

### If forced to pick one for easiest integration with this repository right now

Pick `DynamiCrafter`.

Why:

- The repository already has a DynamiCrafter path.
- The current hidden-state and HyperAlign work already assumes a U-Net-style video backbone.
- It is the lowest-risk path for fast UniCon/ControlNet-style experiments.

That means the practical answer and the strategic answer differ:

- immediate implementation answer: `DynamiCrafter`
- best long-term foundation-model answer: `Wan2.1-T2V-1.3B`

## Suggested next step for this repo

The most sensible near-term plan is a two-track benchmark:

1. Keep `DynamiCrafter` as the U-Net control benchmark for hidden-state, UniCon, and skip-control experiments.
2. Add `Wan2.1-T2V-1.3B` as the first modern flow-matching DiT benchmark for output adapters, hypernetworks, and DiT-native control.

That gives this repository both:

- a control-friendly legacy architecture
- a modern under-7B flow-matching foundation model

## Sources

- Wan2.1 GitHub: https://github.com/Wan-Video/Wan2.1
- Wan2.1 T2V-1.3B model card: https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers
- Wan2.1 VACE-1.3B model card: https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B
- CogVideo GitHub: https://github.com/zai-org/CogVideo
- CogVideoX-2B model card: https://huggingface.co/zai-org/CogVideoX-2b
- CogVideoX-5B model card: https://huggingface.co/zai-org/CogVideoX-5b
- CogKit: https://github.com/THUDM/CogKit
- Open-Sora GitHub: https://github.com/hpcaitech/Open-Sora
- Open-Sora 1.2 report: https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_03.md
- OpenSora STDiT-v3: https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3
- HPC-AI model list: https://huggingface.co/hpcai-tech/models
- Pyramid Flow GitHub: https://github.com/jy0205/Pyramid-Flow
- Pyramid Flow miniFLUX model card: https://huggingface.co/rain1011/pyramid-flow-miniflux
- VideoCrafter GitHub: https://github.com/AILab-CVC/VideoCrafter
- DynamiCrafter GitHub: https://github.com/Doubiiu/DynamiCrafter
- DynamiCrafter model card: https://huggingface.co/Doubiiu/DynamiCrafter
- Allegro GitHub: https://github.com/rhymes-ai/Allegro
- Vchitect-2.0 GitHub: https://github.com/Vchitect/Vchitect-2.0
- Vchitect-2.0-2B model card: https://huggingface.co/Vchitect/Vchitect-2.0-2B
- LTX-Video model card: https://huggingface.co/Lightricks/LTX-Video
- Stable Video Diffusion model card: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid
- Latte GitHub: https://github.com/Vchitect/Latte
- Latte-1 model card: https://huggingface.co/maxin-cn/Latte-1

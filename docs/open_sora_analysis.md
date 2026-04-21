# Open-Sora Repository Analysis for Adapter Research

## Executive Summary

Open-Sora is a comprehensive open-source video generation framework by HPC-AI Tech, implementing state-of-the-art text-to-video (T2V) and image-to-video (I2V) generation. The latest version (v2.0) features an **11B parameter Flux-based MMDiT architecture** trained on ColossalAI infrastructure.

**Key Points for Adapter Integration:**
- Architecture: MMDiT (Multimodal Diffusion Transformer) based on Flux
- Model Type: **Flow Matching** (velocity prediction)
- No U-Net: Uses a dual-stream transformer architecture
- Built-in LoRA support via PEFT library
- Processor pattern enables custom attention implementations

---

## 1. Repository Structure

```
Open-Sora/
├── configs/                    # Hierarchical Python configs
│   ├── diffusion/
│   │   ├── train/             # Training configs (stage1.py, stage2.py, etc.)
│   │   └── inference/         # Inference configs (256px.py, 768px.py)
│   └── vae/                   # VAE training/inference configs
├── opensora/                   # Main Python package
│   ├── models/
│   │   ├── mmdit/             # Core model architecture
│   │   │   ├── model.py       # MMDiTModel class
│   │   │   ├── layers.py      # Attention, blocks, normalization
│   │   │   ├── math.py        # RoPE, attention functions
│   │   │   └── distributed.py # Distributed training variants
│   │   ├── hunyuan_vae/       # 3D VAE for video compression
│   │   ├── dc_ae/             # Deep Compression AutoEncoder
│   │   └── text/              # T5 & CLIP text encoders
│   ├── acceleration/          # ColossalAI integration
│   ├── datasets/              # Data loading & bucketing
│   └── utils/                 # Training, checkpointing, sampling
├── scripts/
│   ├── diffusion/
│   │   ├── train.py           # Main training script
│   │   └── inference.py       # Inference script
│   └── vae/                   # VAE training scripts
└── gradio/                    # Web interface
```

---

## 2. Model Architecture

### 2.1 MMDiTModel (Flux-based)

**Location:** `opensora/models/mmdit/model.py`

```python
@dataclass
class MMDiTConfig:
    in_channels: int = 64           # VAE latent channels
    vec_in_dim: int = 768           # CLIP embedding dimension
    context_in_dim: int = 4096      # T5 text embedding dimension
    hidden_size: int = 3072         # Transformer hidden dimension
    mlp_ratio: float = 4.0
    num_heads: int = 24
    depth: int = 19                 # DoubleStreamBlocks
    depth_single_blocks: int = 38   # SingleStreamBlocks
    axes_dim: list[int] = [16, 56, 56]  # RoPE positional dims
    patch_size: int = 2
    guidance_embed: bool = True     # CFG embedding
    cond_embed: bool = True         # I2V condition embedding
```

**Total parameters:** ~11B

### 2.2 Architecture Flow

```
Input: (B, C=64, T, H, W) from VAE latents
       ↓
   pack() → (B, T*H*W/4, C*4)  # Patchify
       ↓
┌──────────────────────────────────────┐
│        EMBEDDING LAYERS              │
│  img_in: Linear(64 → 3072)           │
│  txt_in: Linear(4096 → 3072)         │
│  time_in: MLP(256 → 3072)            │
│  vector_in: MLP(768 → 3072)          │
│  cond_in: Linear(68 → 3072)          │
│  pe_embedder: RoPE positional        │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│     DOUBLE STREAM BLOCKS (×19)       │
│  Parallel img/txt processing         │
│  Joint attention via concatenation   │
│  ┌────────────┐  ┌────────────┐      │
│  │ IMG STREAM │  │ TXT STREAM │      │
│  │ img_mod    │  │ txt_mod    │      │
│  │ img_norm1  │  │ txt_norm1  │      │
│  │ img_attn   │  │ txt_attn   │      │
│  │ img_norm2  │  │ txt_norm2  │      │
│  │ img_mlp    │  │ txt_mlp    │      │
│  └────────────┘  └────────────┘      │
└──────────────────────────────────────┘
       ↓
   torch.cat([txt, img])
       ↓
┌──────────────────────────────────────┐
│     SINGLE STREAM BLOCKS (×38)       │
│  Merged txt+img processing           │
│  modulation → norm → attn → mlp      │
└──────────────────────────────────────┘
       ↓
   Extract img portion
       ↓
┌──────────────────────────────────────┐
│         LAST LAYER                   │
│  AdaLN modulation + linear output    │
└──────────────────────────────────────┘
       ↓
Output: (B, T*H*W/4, patch_size² * C)
       ↓
   unpack() → (B, C, T, H, W)
```

### 2.3 Key Components for Adapter Injection

**Location:** `opensora/models/mmdit/layers.py`

#### DoubleStreamBlock (lines 256-307)
```python
class DoubleStreamBlock(nn.Module):
    # Image stream
    self.img_mod = Modulation(hidden_size, double=True)  # AdaLN
    self.img_norm1 = nn.LayerNorm(hidden_size)
    self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads)
    self.img_norm2 = nn.LayerNorm(hidden_size)
    self.img_mlp = [Linear → GELU → Linear]

    # Text stream
    self.txt_mod = Modulation(hidden_size, double=True)
    self.txt_attn = SelfAttention(...)
    self.txt_mlp = [...]

    # IMPORTANT: Uses processor pattern
    self.processor = DoubleStreamBlockProcessor()

    def set_processor(self, processor) -> None:
        self.processor = processor
```

#### SingleStreamBlock (lines 337-389)
```python
class SingleStreamBlock(nn.Module):
    self.modulation = Modulation(hidden_size, double=False)
    self.pre_norm = nn.LayerNorm(hidden_size)
    self.linear1 = nn.Linear(hidden_size, hidden_size*3 + mlp_hidden)  # Fused QKV+MLP
    self.linear2 = nn.Linear(hidden_size + mlp_hidden, hidden_size)
    self.norm = QKNorm(head_dim)  # RMSNorm for Q/K

    # IMPORTANT: Uses processor pattern
    self.processor = SingleStreamBlockProcessor()
```

#### SelfAttention (lines 138-169)
```python
class SelfAttention(nn.Module):
    # Fused QKV option
    self.qkv = nn.Linear(dim, dim * 3)  # OR separate q_proj, k_proj, v_proj
    self.norm = QKNorm(head_dim)
    self.proj = nn.Linear(dim, dim)  # Output projection
```

---

## 3. Training Pipeline

### 3.1 Loss Function

**Velocity Prediction (Flow Matching):**

```python
# scripts/diffusion/train.py lines 441-466
sigma_min = 1e-5
v_t = (1 - sigma_min) * x_1 - x_0  # Target velocity
loss = F.mse_loss(model_pred.float(), v_t.float(), reduction="mean")
```

**Time Shift (from SD3):**
```python
def time_shift(alpha, t):
    return alpha * t / (1 + (alpha - 1) * t)

# Alpha depends on resolution and frame count
shift_alpha = get_res_lin_function()((H * W) // 4)
shift_alpha *= math.sqrt(num_frames)  # Temporal scaling
```

### 3.2 Distributed Training

Uses ColossalAI with:
- **ZeRO Stage 2** for optimizer state sharding
- **Sequence Parallelism** for long sequences
- **Tensor Parallelism** for model parallel
- **HybridAdam** optimizer

### 3.3 Data Pipeline

- Variable batch sizes via bucket config
- Supports multiple resolutions (256px, 768px, 1024px)
- Variable frame counts (1-129 frames)
- Video preprocessing with aspect ratio handling

---

## 4. Accessing Pretrained Weights

### 4.1 Download Commands

```bash
# From HuggingFace
huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir ./ckpts

# From ModelScope (China)
modelscope download hpcai-tech/Open-Sora-v2 --local_dir ./ckpts
```

### 4.2 Required Checkpoints

| Model | Path | Size |
|-------|------|------|
| MMDiT (v2.0) | `./ckpts/Open_Sora_v2.safetensors` | ~22GB |
| Hunyuan VAE | `./ckpts/hunyuan_vae.safetensors` | ~300M |
| T5-XXL | `./ckpts/google/t5-v1_1-xxl/` | ~11GB |
| CLIP ViT-L | `./ckpts/openai/clip-vit-large-patch14/` | ~428M |

### 4.3 Loading Checkpoints

```python
from opensora.registry import build_module, MODELS

# Build model from config
model = build_module(cfg.model, MODELS, device_map="cuda", torch_dtype=torch.bfloat16)

# Or use the Flux factory function
from opensora.models.mmdit.model import Flux

model = Flux(
    from_pretrained="./ckpts/Open_Sora_v2.safetensors",
    in_channels=64,
    hidden_size=3072,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    # ... other params
)
```

---

## 5. Integration with Adapter Framework

### 5.1 Critical Differences from Current Framework

| Aspect | DynamicCrafter (Current) | Open-Sora |
|--------|--------------------------|-----------|
| Architecture | 3D U-Net | MMDiT Transformer |
| Model Type | Diffusion | Flow Matching |
| Prediction | Noise (ε) | Velocity (v) |
| Blocks | input_blocks, output_blocks | double_blocks, single_blocks |
| Attention | CrossAttention | Joint Self-Attention |
| Conditioning | cross-attention | Concatenation + Modulation |

### 5.2 Required Interface Wrapper

```python
# Proposed: src/generative_flow_adapters/models/base/opensora.py

from generative_flow_adapters.models.base.interfaces import BaseGenerativeModel

class OpenSoraModelWrapper(BaseGenerativeModel):
    """Wraps Open-Sora's MMDiT for the adapter framework."""

    def __init__(self, model: MMDiTModel, vae, t5_encoder, clip_encoder):
        super().__init__(model_type="flow", prediction_type="velocity")
        self.model = model
        self.vae = vae
        self.t5_encoder = t5_encoder
        self.clip_encoder = clip_encoder

    def forward(self, x_t: Tensor, t: Tensor, cond: object | None = None) -> Tensor:
        """
        Args:
            x_t: Noisy latents (B, C, T, H, W) - packed internally
            t: Timesteps (B,) - expects [0, 1] range for flow matching
            cond: Dict with keys:
                - prompt: List[str] or pre-encoded txt, txt_ids
                - img_ids: Position IDs for image tokens
                - y_vec: CLIP embeddings
                - cond: Conditional image latents (for I2V)
                - guidance: CFG scale
        """
        # Pack latents
        img = pack(x_t, patch_size=self.model.patch_size)
        img_ids = create_img_ids(x_t.shape)

        # Get text embeddings from cond
        txt, txt_ids = self._encode_text(cond)
        y_vec = self._encode_clip(cond)

        output = self.model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=t,
            y_vec=y_vec,
            cond=cond.get("cond"),
            guidance=cond.get("guidance"),
        )

        return unpack(output, ...)
```

### 5.3 Adapter Injection Points

#### Option A: Output Adapter (Simplest)
Apply adapter to model output before unpacking:

```python
class OpenSoraOutputAdapter(Adapter):
    def forward(self, x_t, t, cond, base_output=None):
        if base_output is None:
            base_output = self.base_model(x_t, t, cond)
        return self.adapter_network(base_output, t, cond)
```

#### Option B: LoRA on Attention Projections
Target the QKV and output projections in attention:

```python
# Target modules for LoRA injection
target_modules = [
    # DoubleStreamBlocks
    "double_blocks.*.img_attn.qkv",
    "double_blocks.*.img_attn.proj",
    "double_blocks.*.txt_attn.qkv",
    "double_blocks.*.txt_attn.proj",
    # SingleStreamBlocks
    "single_blocks.*.linear1",  # Contains fused QKV
    "single_blocks.*.linear2",  # Contains output proj
]
```

#### Option C: Custom Processor Pattern (Most Flexible)
Replace block processors for deep integration:

```python
class AdaptedDoubleStreamBlockProcessor:
    def __init__(self, adapter_module):
        self.adapter = adapter_module

    def __call__(self, attn, img, txt, vec, pe):
        # Standard processing
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # ... attention computation ...

        # Inject adapter residual
        img_adapted = img + self.adapter(img, vec)

        return img_adapted, txt

# Apply to model
for block in model.double_blocks:
    block.set_processor(AdaptedDoubleStreamBlockProcessor(adapter))
```

#### Option D: HyperAlign-style Feature Extraction
Extract intermediate features from blocks:

```python
class OpenSoraFeatureExtractor:
    def __init__(self):
        self.features = []

    def hook(self, module, input, output):
        self.features.append(output.detach())

    def attach(self, model):
        for i, block in enumerate(model.double_blocks):
            block.register_forward_hook(self.hook)
```

### 5.4 Key Adaptations Needed

1. **No `input_blocks`**: HyperAlign's `_infer_encoder_feature_dims` won't work
   - Need new feature extraction from `double_blocks`

2. **Packed Sequences**: Inputs are (B, seq_len, hidden) not (B, C, T, H, W)
   - Pool features differently for hypernetwork memory

3. **Dual Streams**: Image and text processed together
   - Decide whether to adapt img stream, txt stream, or both

4. **Modulation (AdaLN)**: Timestep conditioning via modulation, not cross-attention
   - Adapters can inject additional modulation

5. **Flow Matching Loss**: Use velocity prediction loss, not noise prediction

---

## 6. Cautions and Considerations

### 6.1 Memory Requirements

| Config | GPU Memory (Training) | GPU Memory (Inference) |
|--------|----------------------|------------------------|
| 11B @ bf16 | 80GB+ (H100/A100) | 24GB+ (with offloading) |
| With LoRA | ~40GB | ~20GB |
| Gradient Checkpoint | Reduces by ~40% | N/A |

### 6.2 Technical Challenges

1. **Flash Attention Dependency**: Uses flash_attn v2/v3
   - May need fallback for older GPUs

2. **Liger Kernel**: Uses optimized kernels for RMSNorm/RoPE
   - CPU fallbacks exist but slower

3. **ColossalAI**: Training scripts depend on ColossalAI
   - For adapter training, may want to use standard PyTorch

4. **Fused QKV**: Single Linear for Q/K/V complicates LoRA injection
   - Set `fused_qkv=False` in config for separate projections

5. **torch.compile**: Some functions use @torch.compile
   - May conflict with custom modules

### 6.3 Version Compatibility

```python
# Key dependencies
torch==2.4.0
flash-attn>=2.6.0  # Or v3
colossalai>=0.4.4
liger-kernel==0.5.2
peft>=0.10.0
```

### 6.4 Differences from Flux

Open-Sora's MMDiT is modified from Flux with:
- Conditional embedding (`cond_in`) for I2V
- Selective gradient checkpointing
- Distributed training support (shardformer)

---

## 7. Recommended Integration Approach

### Phase 1: Basic Wrapper
1. Create `OpenSoraModelWrapper` implementing `BaseGenerativeModel`
2. Test with simple output adapter

### Phase 2: LoRA Injection
1. Disable fused_qkv for cleaner injection
2. Use PEFT library (already integrated)
3. Target attention projections

### Phase 3: HyperAlign Adaptation
1. Replace feature extraction to use `double_blocks`
2. Modify memory pooling for packed sequences
3. Query tokens for transformer blocks instead of U-Net layers

### Phase 4: Full Training Pipeline
1. Integrate with existing trainer
2. Use flow matching loss
3. Consider ColossalAI for distributed training

---

## 8. Example Configuration

```python
# configs/adapter/opensora_lora.py

model = dict(
    type="flux",
    from_pretrained="./ckpts/Open_Sora_v2.safetensors",
    in_channels=64,
    hidden_size=3072,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    fused_qkv=False,  # IMPORTANT: Disable for clean LoRA
    grad_ckpt_settings=(8, 100),
)

adapter = dict(
    type="lora",
    rank=16,
    alpha=32,
    target_modules=[
        "double_blocks.*.img_attn.q_proj",
        "double_blocks.*.img_attn.k_proj",
        "double_blocks.*.img_attn.v_proj",
        "double_blocks.*.img_attn.proj",
        "single_blocks.*.q_proj",
        "single_blocks.*.k_proj",
    ],
)

training = dict(
    loss_type="flow_matching",  # Velocity MSE
    lr=1e-4,
    warmup_steps=1000,
    gradient_accumulation=4,
)
```

---

## 9. Key Files Reference

| Purpose | File Path |
|---------|-----------|
| Main Model | `opensora/models/mmdit/model.py:69` |
| Block Layers | `opensora/models/mmdit/layers.py` |
| Training Script | `scripts/diffusion/train.py` |
| Inference Script | `scripts/diffusion/inference.py` |
| Sampling Utils | `opensora/utils/sampling.py` |
| Checkpoint Loading | `opensora/utils/ckpt.py` |
| VAE Model | `opensora/models/hunyuan_vae/vae.py` |
| Text Encoders | `opensora/models/text/conditioner.py` |
| Config Example | `configs/diffusion/inference/256px.py` |

---

## 10. Summary

Open-Sora presents a significantly different architecture from the U-Net-based DynamicCrafter currently used in the adapter framework. Key integration considerations:

1. **Use the processor pattern** for deep integration
2. **Disable fused_qkv** for cleaner LoRA injection
3. **Extract features from double_blocks** instead of input_blocks
4. **Implement flow matching loss** for training
5. **Consider memory constraints** - 11B parameters is substantial

The architecture is well-structured for adaptation with clear separation between streams and explicit processor hooks. The existing PEFT/LoRA support provides a good foundation for parameter-efficient fine-tuning.
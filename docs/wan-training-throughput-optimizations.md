# Wan2.2 shortcut training — throughput optimizations (2026-07-29)

Changes that made the Wan2.2 action-free shortcut run **1.44× faster per
micro-step** without touching the training objective, plus one latent bug fix
that also doubles per-optimizer-step throughput (at a semantics cost — read
§3 before taking it).

Measured on an interactive H100-94GB (`gcn158`), ACWM-Phys `kinematics/robot_arm`,
`configs/wan22/diffusion_wan22_shortcut_actionfree_robotarm.yaml`, `bs=12`,
`seq_len=14 175` tokens/clip, warm latent cache, `GFA_PROFILE=1`.
Deeper analysis (including the un-pulled lever in §6) lives in
`thesis-vault/30_Knowledge/tech/wan-shortcut-step-throughput.md`.

---

## 0. Where the time went

`shortcut_anchor_prob: 0.5` splits micro-steps into two very different shapes,
so a single average hides the structure:

| phase | anchor step | supervised step |
|---|---|---|
| data (load + wait) | 60 ms | 60 ms |
| preprocess (latent cache read) | 2 050–2 950 ms | 2 220 ms |
| shortcut target prep | 0.1 ms | 18 760 ms |
| model forward (base + adapter) | 9 370 ms | 9 380 ms |
| backward | 1 090 ms | 1 090 ms |
| **total** | **~13.0 s** | **~31.5 s** |

Average micro-step **22.2 s**. The frozen 5B base forward is **8.93 s** and ran
**2× per micro-step on average — ~81 % of the step**. Every optimization below
therefore targets *how often* the base runs, not how fast it is.

Steady state only: step 0 additionally pays ~9 s of cuDNN/allocator warmup and
~26 s of spawn-context DataLoader startup, plus ~2–4 min of checkpoint load from
GPFS. None of that is per-step cost.

---

## 1. Reuse the frozen-base forward the shortcut prep already took

**Files:** `models/adapted_model.py`, `training/shortcut_targets.py`,
`training/trainer.py`

`Trainer._maybe_prepare_shortcut` → `compute_self_consistency_target_v_flow`
evaluates `v1 = model(x_t, t, cond_half)`. The training forward immediately
after evaluates `model(x_t, t, cond_full)` — **the same `x_t`, the same `t`**.
`cond_half` and `cond_full` come out of the same `_inject_step_level` call and
differ *only* in `step_level`, and `WanTI2VVideoModel.denoise` reads only
`x_t`, `t` and `cond["context"]`. The frozen 5B was producing a bit-identical
result twice, at 8.93 s each.

`AdaptedModel.forward` now accepts `base_output=`, which skips the base forward
and composes onto a prediction the caller already holds.
`compute_self_consistency_target_v_flow(..., return_base=True)` returns its
`v1` base prediction and the trainer feeds it straight into the training
forward. A supervised micro-step goes from **3 base forwards to 2**.

**Safety guard.** Adapters that read the base's *internal* activations (UniCon,
HyperAlign — they expose `clear_captured_base_features`) must not reuse: the
intervening `v2` forward at `(x_mid, t_next)` overwrites those hidden states, so
a reused output would be paired with the wrong internals.
`AdaptedModel.reuses_base_output` encodes exactly that condition, and the
trainer consults it via `_can_reuse_base_output()`. The Wan output adapter used
here qualifies; the hidden-state families keep recomputing.

**Kill switch:** `GFA_NO_BASE_REUSE=1` restores the two-forward behaviour for an
A/B.

---

## 2. Read the precomputed latents in the DataLoader workers

**Files:** `data/latent_prefetch.py` (new), `data/dataset.py`,
`data/translators/base.py`, `data/translators/acwm_phys.py`,
`data/wan_batch_preprocessor.py`, `data/wan22_batch_preprocessor.py`,
`scripts/train_wan22_i2v_metaworld_external.py`

The latent cache already removed the VAE encode from the training step, but not
the **read**. On a cache hit the preprocessor pulled ~5.4 MB × 12 off GPFS
*synchronously, on the training thread*, between the forward passes —
**2.0–2.9 s per micro-step overlapped with nothing**. And the frames were
decoded anyway just to be discarded: `__getitem__` pulled 97 mp4 frames at
480×640 (~89 MB/sample) through decord and the collate, and `_encode_z0` threw
them away the moment the cache hit.

`LatentPrefetchDataset` wraps the clip dataset and inverts the order: resolve
the window's `(episode, start)` identity first (new `TranslatedClipDataset.resolve`),
build the same cache key the preprocessor would, and on a hit return the latent
with **no decode at all** (new `Translator.load_clip_meta` supplies the actions
and identity fields without touching pixels). Because this happens inside
`__getitem__`, `num_workers` copies run in parallel and prefetch ahead of the
GPU, so the cost disappears behind compute instead of adding to it.

**Measured: `preprocess` 2 143 ms → 3.5 ms; `data(load+wait)` 60 ms → 0.2 ms.**

Cold and partially-precomputed caches still train:

- a worker that misses falls back to the normal pixel path;
- `collate_latent_windows` handles an all-hit, all-miss, or **mixed** batch
  (`default_collate` rejects the mixed case outright — the per-sample keys
  differ);
- `WanBatchPreprocessor._encode_mixed` VAE-encodes just the misses and writes
  them back to the cache exactly as before.

`pin_memory=True` rides along, now that a batch is ~65 MB of latents rather than
~1 GB of frames.

### When prefetch is disabled (on purpose)

- `--precompute-latents` — that pass exists to encode pixels.
- **Eval loaders, always.** The native generation grid and the quality metrics
  condition on the observation frame in *pixels* (`raw_batch["video"]`, see
  `Trainer._native_eval_grid`), which a latent-only batch does not carry. Eval
  runs one batch every few hundred steps, so the read cost is irrelevant.
- **Any run whose eval is carved out of the training dataset** (`random_split`
  fallback, `--overfit-index`) — same reason. The script detects this and prints
  `latent prefetch: OFF — ... Pass --eval-data-dir to enable it.`

When it is on, startup prints
`latent prefetch: ON — cached windows load in the N DataLoader worker(s) ...`.

---

## 3. Bug: `grad_accum_steps` was pinned to a hidden minimum of 2

**File:** `training/trainer.py`

```python
accum_steps = max(2, int(self.config.grad_accum_steps))   # was
accum_steps = max(1, int(self.config.grad_accum_steps))   # now
```

`TrainingConfig.grad_accum_steps` is documented as *"1 = no accumulation"* and
defaults to 1, but `max(2, ...)` could never honour that. **Every run in the
repo has been doing two micro-batches per optimizer step** — doubling the
wall-clock of every optimizer step and silently doubling the effective batch.

> ⚠️ **This changes training semantics.** The robot-arm runs so far trained at an
> effective batch of **24** (12 × 2), not 12. To reproduce the old dynamics
> exactly, set `training.grad_accum_steps: 2` in the YAML and give up the 2×.
> Leave it at the default to take the speedup at effective batch 12.

---

## 4. Results

Same H100, same config, warm cache, steady state:

| | before | after | |
|---|---|---|---|
| anchor micro-step | 13.0 s | 10.51 s | −19 % |
| supervised micro-step | 31.5 s | 20.34 s | −35 % |
| **average micro-step** | **22.2 s** | **15.42 s** | **−31 % (1.44×)** |
| per optimizer step | 44.5 s | 15.42 s | 2.9× *(includes §3's semantics change)* |
| VRAM @ bs=12 | 86.5 GiB | 80.9 GiB | −5.6 GiB |

The 1.44× is a pure win — identical objective, identical effective batch. Only
the 2.9× figure carries the §3 caveat.

The freed 5.6 GiB loosens the headroom warning in
`jobs/experiments_cluster/acwm_phys/shortcut/submit_train_wan_shortcut_robotarm.sh`:
the periodic `gen_eval` at step 200 now has ~14 GiB of slack instead of ~8.

---

## 5. Correctness

`tests/test_training_throughput_paths.py` — 7 tests, all passing:

- the frozen base output is **independent of `step_level`** (the precondition
  that makes reuse sound at all);
- reuse is **bit-identical** to recomputing, and provably runs the base zero
  extra times;
- the shortcut prep's returned base output equals the training forward's, and
  `return_base=True` does not perturb the target;
- an adapter that captures base internals **opts out** of reuse;
- the collate survives all-hit, all-miss, and mixed batches.

Two bugs were caught and fixed during the work:

1. `LatentPrefetchDataset.__getattr__` raised `KeyError` instead of
   `AttributeError` during spawn-context unpickling (which probes
   `__setstate__` before `__dict__` exists), aborting every worker.
2. Prefetching the **eval** loader silently broke the generation grid — the
   `video` key it reads simply wasn't there, so it logged nothing rather than
   erroring. Hence the exclusions in §2.

---

## 6. The big lever not pulled

The base forward is still **8.93 s for ~2.0 PFLOP ⇒ ~224 TFLOPS ≈ 23 % MFU** on
a card that does ~990 TFLOPS bf16 dense. The suspect is Wan2.2's
diffusion-forcing timestep path in
`external_repos/Wan2.2/wan/modules/model.py`:

```python
if t.dim() == 1: t = t.expand(t.size(0), seq_len)
with torch.amp.autocast('cuda', dtype=torch.float32):
    e  = self.time_embedding(...)                        # [B, L, 3072]    fp32
    e0 = self.time_projection(e).unflatten(2, (6, dim))  # [B, L, 6, 3072] fp32
```

At `B=12, L=14 175, dim=3072` that `e0` is **12.5 GB of fp32**, and each of the
30 blocks re-materialises `(self.modulation.unsqueeze(0) + e)` at that size
before chunking it six ways — on the order of a **terabyte of HBM traffic per
forward**, which is the classic signature of a ~20 % MFU stall.

The redundancy: our `t` reaches the DiT as `[B, T'=25]` from the diffusion-forcing
preprocessor and is expanded by `_to_dit_timesteps` via `repeat_interleave`, so
`e0` holds only **25 distinct rows per sample — really 2** (0 on observation
frames, σ on the future). Computing the modulation on the compact
`[B, 25, ...]` form and broadcasting at use time should recover most of that
traffic *and* the 12.5 GB transient, which would allow a substantially larger
batch.

Not attempted here: it means patching the vendored DiT and needs a bit-exactness
check against current output before it touches a real run. **This is the
highest-value next optimization.**

Cheaper unexplored extras: TF32 for the fp32 `time_projection` matmul
(~19 TFLOP fp32 per forward, currently running at ~67 TFLOPS), and
FlashAttention-3 (the installed build is FA2; self-attention at `L=14 175` is
~0.30 PFLOP/forward).

---

## 7. Reproducing the measurement

```bash
export GFA_PROFILE=1 WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=../scratch-shared/acwm-phys/kinematics/robot_arm

python scripts/train_wan22_i2v_metaworld_external.py \
    --config configs/wan22/diffusion_wan22_shortcut_actionfree_robotarm.yaml \
    --dataset acwm_phys --data-dir "$ROOT/ind_train" --eval-data-dir "$ROOT/ind_test" \
    --latent-cache-dir "$ROOT/latents.shared" --ckpt-dir ckpts/Wan2.2-TI2V-5B \
    --batch-size 12 --num-windows 8 --max-area 589824 --steps 20 --num-workers 8
```

Read `[prof]` lines from step ≥ 2 only. Add `GFA_NO_BASE_REUSE=1` to A/B §1.

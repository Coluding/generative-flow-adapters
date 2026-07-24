# Online VAE-encode profiling jobs

Three cluster jobs around `scripts/profile_vae_encode.py`. Together they answer:
**at 768² (max_area 589824), can we drop the latent-precompute step and encode
online during training?** Pre-encoding was only ever a workaround for the
native-1280×704 OOM; at 768² the transient is much smaller.

| Job | What it measures | Walltime |
|---|---|---|
| `submit_profile_vae_acwm.sh` | Encode cost + isolated transient on ACWM push_block at the real training geometry | 30 min |
| `submit_profile_vae_metaworld.sh` | Same profile on MetaWorld — isolates dataset-loader overhead from the encode | 30 min |
| `submit_profile_vae_coexist_dit.sh` | Encode transient *alongside the resident 5B* (`--with-dit`) — memory coexistence | 45 min |
| `submit_profile_train_step.sh` | **Closes the verdict:** real training step time vs online-encode cost, same loop | 45 min |

## Run

From the repo root on the login node (weights + venv must already be in place;
no internet on compute nodes):

```bash
sbatch jobs/experiments_cluster/infra/profile_vae/submit_profile_vae_acwm.sh
sbatch jobs/experiments_cluster/infra/profile_vae/submit_profile_vae_metaworld.sh
sbatch jobs/experiments_cluster/infra/profile_vae/submit_profile_vae_coexist_dit.sh
```

Any extra flags are forwarded to the script, e.g. a wider sweep:

```bash
sbatch .../submit_profile_vae_acwm.sh --batch-size 1 2 4 8 --num-batches 20
```

## Reading the output

- `encode/batch` — GPU VAE cost. Compare `ms/clip` to your training step time:
  if encode ≪ step, online encoding is essentially free with a couple of
  dataloader workers and precompute can be dropped.
- `resize+h2d/batch` — CPU LANCZOS resize + host→device. Hidden by workers in
  real training; profiled here with `num-workers 0` so it's visible.
- `peak alloc` — the encode transient. In jobs 1–2 it's in isolation; in job 3
  it sits on top of the resident 5B (`peak_alloc − resident ≈ added transient`).
  Job 3 is the one that decides coexistence — an isolated transient that fits
  the card means nothing if it OOMs next to the resident model.

## The training-step profiler (`submit_profile_train_step.sh`)

The first three jobs measure the *encode* in isolation; they can't say whether
2 s/clip is cheap without the training step time to compare against. This job
provides that comparison **from the real training loop** — no reimplementation.
It sets `GFA_PROFILE=1`, which enables the trainer's per-phase CUDA-synced
timers, and runs ~15 steps at the real geometry with eval off. Per step it
prints `data(load+wait)`, `preprocess (VAE encode + cond)`, `forward`,
`backward`, `optimizer.step`, and `training_step (fwd+bwd+opt) TOTAL`.

It runs two variants back to back:

- **A — online encode** (`--no-latent-cache`): `preprocess` shows the full
  encode+resize tax.
- **B — cached latents** (`--latent-cache-dir`): `preprocess` drops to
  cache-load only. Skipped automatically unless the shared cache is warm (run
  `submit_precompute_acwmphys.sh` first).

**Decision rule:** `A_preprocess − B_preprocess` is the per-step online-encode
tax. If `training_step TOTAL` ≫ that tax → drop precompute (online encode is a
small % overhead and memory already fits). If they're comparable, or you train
many epochs, keep the cache.

## Notes

- `num-workers 0` everywhere: this profile script builds a plain (un-spawned)
  DataLoader, so ACWM's decord readers can fork-deadlock with workers > 0. Zero
  workers keeps it safe and makes the CPU `resize+h2d` cost measurable.
- Geometry is pinned to the ACWM training config (768², 65 frames). Job 3
  forces `provider: wan2.2_external` (real pretrained weights) inside the script.

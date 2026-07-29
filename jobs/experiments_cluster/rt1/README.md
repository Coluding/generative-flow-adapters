# RT-1 in-distribution ACTION test (our adapter)

The decisive follow-up to the AVID-RT1 control (`93qrvr5v`, which *followed*
actions): does **our own lightweight Wan output adapter** follow actions on
in-distribution real robot video (RT-1), where it went **blind** on OOD synthetic
ACWM (`ncztxyyo` 0.0056)? If yes → the action-blindness was the data, and D2 flips
positive. Ticket context:
`30_Knowledge/experiments/20260729-avid-rt1-follows-actions-control.md`.

## Pipeline (in order)

```bash
# 0) RT-1 already downloaded for AVID (avid_official/download_rt1.sh). Then:
sbatch jobs/experiments_cluster/rt1/convert_rt1.sh                  # RLDS -> mp4+metadata (AVID env, local, std-norm actions)
sbatch jobs/experiments_cluster/rt1/submit_precompute_rt1_latents.sh  # Wan VAE latents  ("pre-encode")
sbatch jobs/experiments_cluster/rt1/submit_precompute_rt1_captions.sh # per-clip T5 (natural_language_instruction)
sbatch jobs/experiments_cluster/rt1/submit_train_wan_rt1_action.sh    # THE action test
```

Knobs: `RT1_SPLIT` (default `train[:5000]` — enough to avoid the 64-clip
memorization confound), `RT1_OUT` (default `$HOME/scratch-shared/rt1/train`),
`NUM_WINDOWS` (default 2 — **must match** between the latent precompute and the
train job or the cache won't hit).

## Read the result

Same action diagnostics as the ACWM matrix, now in-distribution:
- **`eval_action_effect_rel`** — blind ~0.003 on ACWM; **higher here ⇒ our
  adapter follows actions in-distribution** (D2 positive).
- `condition_grad_norm` / `action_inject_grad_norm` — does the action pathway get
  gradient (it didn't, effectively, on ACWM).
- `eval_action_base_null_violation` ~0 — trust control.

Notes: actions are per-dim std-normalized (octo convention — the converter does
it; `--no-normalize` to undo). RT-1 is 256×320 native, upsampled to the base's
regime (`max_area 589824`). DC/SkyReels RT-1 variants are a later add if the Wan
result is promising.

# OpenVid shortcut runs (D3 — flow vs diffusion, in-distribution captioned video)

The in-distribution, non-robotic dataset for the few-step **shortcut** test:
real-world captioned web video (OpenVid-1M subset) the frozen bases generate
coherently, so few-step fidelity is well-defined and the flow-vs-diffusion
contrast isn't confounded by OOD drift. Ticket: thesis-vault
`20_Tickets/experiments/exp-shortcut-flow-vs-diffusion-openvid.md`.

TI2V: condition on frame-0 (image anchor) + the clip's **own caption** (per-clip
T5, via `PromptContextProvider` positive mode keyed by `clip_id`). No actions.

## Order

```bash
# 1) LOGIN NODE — download the subset (~a few GB, real captions)
bash   jobs/experiments_cluster/openvid/download_openvid.sh

# 2) precompute per-clip caption T5 table (GPU; writes configs/prompts/openvid_train.contexts.pt)
sbatch jobs/experiments_cluster/openvid/submit_precompute_openvid_captions.sh

# 3) train — Wan FIRST (the headline flow result); DC/SkyReels after Wan shows quality holds
sbatch jobs/experiments_cluster/openvid/submit_train_wan_shortcut_openvid.sh
#      jobs/experiments_cluster/openvid/submit_train_dc_shortcut_openvid.sh       (deferred)
#      jobs/experiments_cluster/openvid/submit_train_skyreels_shortcut_openvid.sh
```

Data lands under `$HOME/scratch-shared/openvid/train` (override with `OPENVID_DIR`);
clip count via `NUM_CLIPS` (default 2000 — too many to memorize).

## Read the result

Few-step rollout fidelity at N ∈ {1,2,4,8,25,50} (the config `eval_step_schedule`)
vs the base's 50-step. **Wan (flow)** holding at small N = the headline. Then the
Wan-vs-DC gap tests the curvature theory (DC uses `endpoint_inversion`; a
`v_average` naive-DC variant is deferred, add post-Wan for the 3-way table).

## Smoke-first (before trusting a full run)

- **DC** — its per-clip caption goes through DC's own CLIP text encoder (not the
  Wan positive table); confirm the caption reaches DC on a short `--steps` run.
- **SkyReels** — confirm `step_level` threads to the adapter at runtime.
- All bases: confirm `shortcut_direction_loss` / `multistep_consistency_loss` are
  > 0 (non-inert) on step 1 — Wan already passed this locally 2026-07-28.

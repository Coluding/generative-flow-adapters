# D3 — few-step shortcut: flow vs diffusion

**Question:** can a step-size-conditioned adapter enable few-step generation on a
frozen base? And does it work on **flow** (Wan/SkyReels, near-straight
trajectory) while failing on **diffusion** (DC, curved trajectory)? Wan working
alone is already the headline; the Wan-vs-DC gap validates the curvature section.
Action-free (isolates D3 from the D2 action problem). Ticket:
`20_Tickets/experiments/exp-shortcut-flow-vs-diffusion-openvid.md`.

**Three metrics, in priority order:**
1. **Non-inert check** — `shortcut_direction_loss` / `multistep_consistency_loss`
   > 0 (the old configs shipped `anchor_prob 1.0` = inert; ours is 0.5). Gate.
2. **`eval_stepsize_effect_rel`** — does the prediction move when the step-size `d`
   changes? **≈0 ⇒ step-size-blind** (self-consistency collapsed → few-step is
   fake — the copy-the-base failure in D3 form). Null: `eval_stepsize_base_null_violation ≈0`.
3. **Few-step quality N-sweep** — the `eval_step_grid/*` videos (rows N∈{1,2,4,8,25,50},
   cols gt|base|adapted) + FID/FVD. The payoff: does adapted at small N ≈ base at 50?

---

## Substrate A — ACWM Robot Arm (ready; reuses the action runs' data)

```bash
sbatch jobs/experiments_cluster/acwm_phys/shortcut/submit_train_wan_shortcut_robotarm.sh   # flow — FIRST
sbatch jobs/experiments_cluster/acwm_phys/shortcut/submit_train_dc_shortcut_robotarm.sh     # diffusion (deferred)
sbatch jobs/experiments_cluster/acwm_phys/shortcut/submit_train_skyreels_shortcut_robotarm.sh
```

## Substrate B — OpenVid (in-distribution captioned real video; cleaner few-step target)

**→ [`jobs/experiments_cluster/openvid/README.md`](../experiments_cluster/openvid/README.md)**
```bash
bash   jobs/experiments_cluster/openvid/download_openvid.sh
sbatch jobs/experiments_cluster/openvid/submit_precompute_openvid_captions.sh
sbatch jobs/experiments_cluster/openvid/submit_train_wan_shortcut_openvid.sh
```

## Offline few-step video generation (figures + inspect a checkpoint)

The in-training `eval_step_grid` videos are logged to wandb during a run. For
reproducible thesis figures — or to inspect an existing run (`pzmc2orq` Wan,
`t4bp8nki` DC) WITHOUT restarting it — generate them offline from a checkpoint:

```bash
CONFIG=configs/wan22/diffusion_wan22_shortcut_actionfree_robotarm.yaml \
CHECKPOINT=outputs/wan-shortcut-actionfree-robotarm/checkpoints/best.pt \
DATA_DIR=$HOME/scratch-shared/acwm-phys/kinematics/robot_arm/ind_train \
sbatch jobs/experiments_cluster/shortcut_eval/submit_generate_fewstep.sh
```
Saves `fewstep_sample*.mp4` (rows N∈{1,2,4,8,25,50}, cols gt|base|adapted) +
`stepsize_perturb_sample*.mp4` (50 DDIM steps at wrong vs correct step_level —
**identical rows ⇒ step-size-blind**, the visual twin of `eval_stepsize_effect_rel`).
Script: `scripts/generate_shortcut_fewstep.py` (Wan/SkyReels/DC).

## Sequencing (priority)
1. **Wan first** — prove flow few-step works (headline). Batch is LOW (shortcut
   does 2–3× base forwards → OOMs a 24 GB GPU; H100 only).
2. **DC after** — `endpoint_inversion` (curvature-aware). Smoke-first: DC's
   per-clip caption → its own CLIP encoder is untested.
3. Optional **DC + `v_average`** naive variant (deferred, metrics-completeness):
   shows the sagitta bias — a 2-line config clone.

## Result so far (early, `pzmc2orq` Wan / `t4bp8nki` DC)
Wan shortcut losses falling cleanly across all N (learning); DC
`multistep_consistency_loss` ~0.86 vs Wan 0.013 (~65× — the curvature signature,
caveated by different targets `v_average` vs `endpoint_inversion`). Few-step
quality + `eval_stepsize_effect_rel` still open (those runs predate the step-size
probe — restart to log it). Note:
`30_Knowledge/experiments/20260729-shortcut-wan-vs-dc-curvature-signature.md`.

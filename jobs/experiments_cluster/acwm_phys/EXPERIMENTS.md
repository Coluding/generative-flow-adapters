# ACWM-Phys experiment matrix — base × dataset (D2)

Two axes crossed: **base backbone** (Wan2.2-5B strong flow · SkyReels-V2-1.3B
weak flow · DynamiCrafter weak diffusion) × **dataset** (Push Cube flat-2D ·
Robot Arm rich-3D), plus one Wan intervention run on Push Cube.

**Why this matrix.** On flat Push Cube the frozen base is near-perfect (masked
denoise loss ~0.036 at 17f) so the adapter clones it; on Robot Arm the base
leaves ~8.7× more residual (**0.314**, measured 2026-07-25 with a random
adapter) for the action adapter to close. The base axis tests whether a *weaker*
base leaves more residual and is easier to adapt. See thesis-vault
`30_Knowledge/writing/ablation-axes.md` and the per-run tickets under
`20_Tickets/experiments/`.

Clean baseline recipe (all runs unless noted): output adapter, `mask_mix`
composition, base-input, action conditioning. No gate_cap / warmup / sigma_shift
except where a run's purpose is that intervention.

## The runs

| # | Base | Dataset | Script | Ticket |
|---|------|---------|--------|--------|
| 1 | Wan2.2 | Robot Arm | `wan/submit_train_wan_robotarm.sh` | exp-backbone-wan-robotarm-run |
| 2 | SkyReels | Push Cube | `skyreels/submit_train_skyreels_pushblock.sh` | exp-backbone-skyreels-pushblock-run |
| 3 | SkyReels | Robot Arm | `skyreels/submit_train_skyreels_robotarm.sh` | exp-backbone-skyreels-robotarm-run |
| 4 | DynamiCrafter | Robot Arm | `dc/submit_train_dc_robotarm.sh` | exp-backbone-dc-robotarm-run |
| 5 | DynamiCrafter | Push Cube | `dc/submit_train_dc_pushblock.sh` | exp-backbone-dc-pushblock-run |
| 6 | Wan2.2 | Push Cube · **cap 0.5 + warmup 500** | `wan/submit_train_wan_pushblock_cap50_warmup.sh` | exp-adapter-wan-cap50-warmup-pushblock-run |

## Readiness & prerequisites

| # | State | Prereq before `sbatch` |
|---|-------|------------------------|
| 6 | **launch-ready** | push_block latents (already precomputed) |
| 1 | ready | `infra/download_acwmphys_robotarm.sh` → `infra/submit_precompute_acwmphys_robotarm.sh` (login node) |
| 4 | ready* | `infra/download_acwmphys_robotarm.sh` only (DC encodes live — no precompute). *DC base-coherence probe recommended first |
| 5 | ready* | push_block raw data in-repo. *flat-2D likely OOD for DC512 — probe first |
| 2,3 | **code-complete, GPU-val pending** | SkyReels preprocessor + `train_skyreels_acwm.py` + sbatch all built & smoke-green; needs a GPU shakedown (8 inline `# GPU-VALIDATE` items) before trusting a run. Live-encode, no precompute (only `z0` cached). |

## Data locations (differ by dataset — mind this)

- **Push Cube:** in-repo `ds/acwm-phys/rigid_dynamics/push_block/{ind_train,ind_test}` (+ `latents.shared` cache for Wan).
- **Robot Arm:** scratch-shared `$HOME/scratch-shared/acwm-phys/kinematics/robot_arm/{ind_train,ind_test}` (+ `latents.shared` for Wan).

## Launch order (suggested)

1. **Run 6** now — fastest signal on whether interventions rescue base-parity on
   the near-zero-residual domain.
2. **Run 1** after robot_arm download+precompute — the Wan Robot-Arm baseline.
3. **Runs 4/5** (DC) after a quick DC base-coherence probe per domain.
4. **Runs 2/3** (SkyReels) once the entrypoint lands.

## Reading the results

- Primary signal per run: **adapted denoise loss vs frozen-base loss** and
  **pred-base cosine** (clone ⇒ cosine ~0.85, gate saturates). Robot-Arm cells
  should show the adapted loss dropping below 0.314; Push-Cube cells are
  expected to hit the flat-visuals ceiling.
- Action-following probe: `scripts/generate_wan22_i2v_compare.py --sigma-sweep
  --action-probe` (shuffle-gap > 0 = the base is using actions).
- **No unsourced numbers** in write-ups (thesis hard rule 8): every metric cites
  a wandb run id + ckpt + commit. Push-Cube base residual = 0.036, Robot-Arm =
  0.314 (matched 17f masked, random adapter, 2026-07-25).

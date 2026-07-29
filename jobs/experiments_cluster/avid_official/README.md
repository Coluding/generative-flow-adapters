# Official AVID reproduction on RT-1 (cluster)

Reproduce the AVID paper's own latent-diffusion training on its OFFICIAL data
(RT-1 / `fractal20220817_data`, Open X-Embodiment) and probe it for action
sensitivity — the **control** for our ACWM finding
([[../../../.. ]] thesis-vault `30_Knowledge/experiments/20260728-acwm-robotarm-matrix-action-blind.md`).

**The question:** we measured all three adapter backbones as action-blind on
ACWM (`eval_action_effect_rel` ~0.001–0.006), and the AVID checkpoint on a
64-clip ACWM smoke run was also blind (0.0015) — but that was a memorization
confound. Does AVID's *own recipe on its own data* follow actions? If **yes**,
the ACWM blindness is about our data/adaptation. If **no**, it's about the
recipe or our probe.

This uses the **UNMODIFIED** `configs/train/avid/avid_11M.yaml` (RTXDataModule +
`act_cond_diffusion_11M.yaml`, `action_dims 7`) — only the data path is
localized, batch is shrunk to one GPU, and wandb is online.

## Prerequisite — get the AVID repo onto the cluster

`external_repos/` is gitignored (`.gitignore:12`), so `git pull` brings these
`jobs/` scripts but **NOT** the AVID repo itself. Transfer it once (code only,
~11 MB) — same as however `external_repos/Wan2.2` reached the cluster:

```bash
# from the local machine:
rsync -av --exclude=outputs --exclude='.venv*' --exclude='*.ckpt' \
  external_repos/avid/  <cluster>:~/generative-flow-adapters/external_repos/avid/
```

And commit + push the main-repo changes first (these scripts + the tracked code)
so the cluster's `git pull` has them.

## Steps (in order)

```bash
# 1) LOGIN NODE — download RT-1 (~111 GB, public bucket, resumable)   [DONE 2026-07-28]
bash jobs/experiments_cluster/avid_official/download_rt1.sh
#    -> $HOME/scratch-shared/oxe/fractal20220817_data/0.1.0/
#    verified complete: 1024/1024 shards, 119,259,986,287 B = 111.07 GiB

# 2) LOGIN NODE — build the AVID env (poetry.lock -> uv venv; torch 2.1+cu118 / PL 1.9.3 / tfds 4.9.2)
bash jobs/experiments_cluster/avid_official/setup_avid_env_cluster.sh
#    -> external_repos/avid/latent_diffusion/.venv

# 3) COMPUTE — train (unmodified avid_11M.yaml on RT-1, online wandb project avid-rt1-official)
cd ~/generative-flow-adapters
mkdir -p logs/avid-rt1        # Slurm does NOT create --output's directory; without
                              # this the job dies instantly and writes no log at all
sbatch jobs/experiments_cluster/avid_official/submit_train_avid_rt1.sh
```

Run outputs (workdir, checkpoints, wandb) go to `/scratch-shared/$USER/avid-rt1`,
not the repo — the vendored configs' `/host_home/*` is a path from the authors'
Docker mount and does not exist here. Override with `LOGDIR=… sbatch …`.

## Known unknowns (validate cluster-side)

- **Env build** took three cluster-specific workarounds, all commented in step 2's
  script: the repo's `uv.lock` is an EMPTY STUB (0 packages, so `uv sync` installs
  nothing — only `poetry.lock`'s 277 are real); poetry's own env creation is broken
  here (`CPython2macOsArmFramework`, survives pinning poetry 1.8.2 *and* virtualenv
  20.26.6); and uv's isolated build envs inherit this interpreter's setuptools-80.9
  distutils hack, so every source build fails with `No module named
  'packaging.utils'` until isolation is turned off. Net: poetry exports the lock,
  uv builds the venv, installs run `--no-build-isolation`.
  Still genuinely unverified: torch `cu118` against the node CUDA driver — that
  only shows up on a GPU node.
- **`gsutil`**: not preinstalled; installed per-user (5.37, `~/.local/bin`). The
  download script puts that on PATH itself, since job 24986795 failed exactly
  there — a Slurm batch shell does not source `.bashrc`.
- **Compute nodes DO have outbound internet** (250 online wandb runs from `gcn*`
  nodes in `~/scratch-shared/wandb`, 0 offline), so `offline=False` is fine. An
  earlier version of the download script asserted the opposite and hard-failed
  under Slurm; that guard is gone.
- **Batch size**: `avid_11M.yaml` ships `batch_size 16` (4×A100). The job sets 8
  for a single H100 — raise via `BATCH=… sbatch …` if VRAM allows.
- **tcmalloc**: the RT-1 tfds loader is CPU-memory heavy (AVID README); set
  `LD_PRELOAD` to a `libtcmalloc.so` if a gperftools/jemalloc module exists.

## The probe (final step — after a real amount of training)

The action-sensitivity probe lives at
`external_repos/avid/latent_diffusion/scripts/probe_action_sensitivity.py` and
already works AVID-side (perturb `cond["act"]`, measure
`‖pred_true − pred_perturbed‖/‖pred_true‖` + cosine, with the frozen base as the
null control). **Caveat:** it currently pulls batches from the ACWM datamodule —
for RT-1 it needs to pull from `RTXDataModule` instead (same `"act"` key, so it's
a datamodule swap / a `--datamodule rtx` flag, not new logic). Point it at the
latest `/scratch-shared/$USER/avid-rt1/avid_rt1_11M/checkpoints/epoch=*-step=*.ckpt`
(`avid_rt1_11M` is `name:` in `avid_11M.yaml`; `init_workspace` makes the workdir
`$logdir/$name`) and report
`action_effect_rel` next to our runs (Wan 0.0056 / DC 0.0034 / SkyReels 0.0013).

Compare on a matched footing: this is RT-1 (official), ours are ACWM — but both
run the same probe + null control, so the action_effect_rel numbers are directly
comparable in what they measure.

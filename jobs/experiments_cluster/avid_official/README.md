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
# 1) LOGIN NODE — download RT-1 (~111 GB, public bucket, resumable)
bash jobs/experiments_cluster/avid_official/download_rt1.sh
#    -> $HOME/scratch-shared/oxe/fractal20220817_data/<version>/

# 2) LOGIN NODE — build the AVID env (poetry; reproduces torch 2.1+cu118 / PL 1.9.3 / tfds)
bash jobs/experiments_cluster/avid_official/setup_avid_env_cluster.sh

# 3) COMPUTE — train (unmodified avid_11M.yaml on RT-1, online wandb project avid-rt1-official)
sbatch jobs/experiments_cluster/avid_official/submit_train_avid_rt1.sh
```

## Known unknowns (validate cluster-side)

- **Env build** is the one piece untested remotely: torch `cu118` must match the
  node CUDA driver (cu118 is widely forward-compatible). `poetry.lock` makes the
  resolution deterministic, so any failure is a wheel/native-lib issue, not
  versions — iterate in step 2.
- **`gsutil`** may not be preinstalled; the download script prints the fallback
  (`pip install --user gsutil` or `gcloud storage cp`).
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
latest `outputs/avid_rt1_official_11M/checkpoints/epoch=*-step=*.ckpt` and report
`action_effect_rel` next to our runs (Wan 0.0056 / DC 0.0034 / SkyReels 0.0013).

Compare on a matched footing: this is RT-1 (official), ours are ACWM — but both
run the same probe + null control, so the action_effect_rel numbers are directly
comparable in what they measure.

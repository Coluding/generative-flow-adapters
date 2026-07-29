# Mission: Sharding data-preprocessing pipelines

## Why
Lukas is preprocessing large robot-video corpora (RT-1, ACWM-Phys, OpenVid) into
mp4+metadata+latents for adapter training, on a Slurm cluster where a serial pass
takes 21 hours and a silent preprocessing bug costs a whole experiment. The goal
is to be able to parallelise the *next* dataset without supervision, and to read a
broken job and know what to do — rather than re-deriving the RT-1 decisions each
time.

## Success looks like
- Given a new dataset and a preprocessing stage, decide **unaided** whether it can
  be sharded, and if not, what has to move to a merge step.
- Read a failed Slurm array and tell the difference between "re-run this index"
  and "the merged output is silently corrupt".
- Follow every line of `submit_convert_rt1_shards.sh` and `merge_rt1_shards.py`,
  including *why* each guard is there.
- Name the failure modes that produce **wrong-but-plausible data** instead of a
  crash, and the check that catches each one.

## Constraints
- Time is thesis time: lessons must be short and immediately applicable.
- Learning happens against a live pipeline — worked examples should use the real
  RT-1 / ACWM / OpenVid code in this repo, not toy analogies.
- Cluster is Snellius (Slurm, `genoa` CPU + `gpu_h100`).

## Out of scope
- Defending the preprocessing choices in thesis prose (explicitly not the goal).
- Distributed-training parallelism (FSDP/DDP) — this mission is about *data*
  preprocessing, not model sharding.

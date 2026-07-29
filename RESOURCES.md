# Sharding Data-Preprocessing Pipelines — Resources

## Knowledge

- [Paper: _MapReduce: Simplified Data Processing on Large Clusters_ — Dean & Ghemawat, OSDI 2004](https://www.usenix.org/conference/osdi-04/mapreduce-simplified-data-processing-large-clusters)
  The canonical statement of the shard/merge split: a `map` that is independent
  per record, and a `reduce` that owns everything requiring the whole set.
  Use for: the underlying model of *why* a global coupling must be deferred, not
  distributed. Sections 2–3 are enough.

- [Docs: _TFDS and determinism_ — TensorFlow Datasets](https://www.tensorflow.org/datasets/determinism)
  States the ordering guarantees we depend on when slicing a split into shards.
  Key line: *"The example order is only guaranteed to be the same for a fixed
  value of interleave args"*, and `ds.take(x)` is **not** equivalent to
  `split='train[:x]'`.
  Use for: any time a shard boundary is defined by a split slice.

- [Docs: _Splits and slicing_ — TensorFlow Datasets](https://www.tensorflow.org/datasets/splits)
  The slice syntax itself: absolute (`train[123:450]`), percent, and `shard`
  based selection.
  Use for: writing the per-shard range arithmetic.

- [Docs: _Job Array Support_ — SchedMD (Slurm)](https://slurm.schedmd.com/job_array.html)
  Primary source for `--array`, `SLURM_ARRAY_TASK_ID`, `SLURM_ARRAY_JOB_ID`, and
  how array elements are scheduled and accounted independently.
  Use for: writing and debugging array jobs; understanding `sacct` output for
  `<jobid>_<index>`.

- [Docs: _sbatch_ — SchedMD (Slurm)](https://slurm.schedmd.com/sbatch.html)
  Use for: the exact semantics of `--array=0-17%N`, `--ntasks-per-node`,
  `--cpus-per-task`, and how they interact with node packing.

## Wisdom (Communities)

- [SURF / Snellius user support](https://servicedesk.surf.nl/)
  The people who actually own this cluster's partitions, quotas, and login-node
  watchdog. Use for: partition choice, fairshare, why a job was killed.
- [r/HPC](https://reddit.com/r/HPC)
  Practitioner-level Slurm and cluster-storage discussion. Use for: sanity-checking
  an array layout or a filesystem-pressure question before burning node-hours.

_No community preference expressed yet — ask before leaning on these._

## Gaps

- No high-trust source yet on **filesystem pressure from many small files**
  (the latent cache writes ~169k `.pt` files on GPFS). Worth finding before the
  next dataset — inode limits and metadata-server load are real constraints here.
- No primary source yet on **idempotent/resumable batch conversion** patterns;
  the RT-1 converter is not resumable and that shaped the whole sharding design.

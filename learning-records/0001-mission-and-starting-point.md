# Starting point: built the RT-1 sharding live, wants transfer + debugging

Lukas set the mission as three of four offered outcomes — **apply the strategy to other
datasets, debug it when it breaks, and follow the existing code line-by-line** — and
explicitly *not* defending the choices in thesis prose. Lessons should therefore stay
practitioner-facing: decision procedures and failure modes, not justification narratives.

## Evidence of prior knowledge

He watched the RT-1 sharding get designed and launched in the same session (18-way
Slurm array + `merge_rt1_shards.py`), and had already absorbed the surrounding pipeline:
he asked why only 9,690 latents existed and pushed back on the dataset size being smaller
than expected — both times reasoning about the data rather than the code. So he has
**exposure** to the shard/merge split and the RT-1 specifics, but has not yet applied the
audit unaided to a stage he did not watch being built.

## Implications

- Do **not** re-teach the RT-1 pipeline mechanics as new material; use them as the worked
  example and spend the lesson budget on transfer.
- The zone of proximal development is *classification under interleaving* — recognising a
  coupling in ACWM/OpenVid/an unseen dataset — not recall of what RT-1 did.
- He is operationally fluent with Slurm (submits arrays, reads `sacct`), so array
  mechanics can be assumed; the gap is in **which failures are silent**, not in how to
  read the queue.
- Lesson 02 should be the debugging strand: reading a failed array and distinguishing
  "re-run index 7" from "the merged output is silently corrupt".

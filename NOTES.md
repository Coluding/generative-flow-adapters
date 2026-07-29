# Working notes

## Preferences observed
- Wants **concrete over abstract**: pushes back with real numbers ("why only 9690?",
  "I thought RT-1 was larger"). Worked examples must use this repo's live code and
  measured figures, never toy analogies.
- Reads code directly and asks for line-level explanation — file:line citations land well.
- Terse questions, often typo-heavy and fast. Answer the question asked; don't pad.

## Workspace placement
The teaching workspace currently sits at the **repo root**
(`MISSION.md`, `RESOURCES.md`, `NOTES.md`, `lessons/`, `reference/`, `assets/`,
`learning-records/`). This is what the skill prescribes, but it does add seven
untracked top-level entries to a git repo. Offer to either gitignore them or relocate
under `docs/teaching/` if `git status` noise becomes annoying.

## Open threads (not lesson material)
- `/mattpocock-skills:setup-matt-pocock-skills` was mid-flight when `/teach` was invoked:
  drafts approved-pending for `docs/agents/issue-tracker.md` (GitHub, `gh` not yet
  installed on Snellius) and `docs/agents/domain.md`, plus an `## Agent skills` block for
  `CLAUDE.md`. Nothing written yet — awaiting go-ahead.
- Live RT-1 full-dataset run in progress (18-shard array 25052058 + captions 25051937).

## Lesson backlog
1. ✅ 01 — Finding the global coupling
2. Reading a failed Slurm array: "re-run index 7" vs "the merge is corrupt"
3. Resumability: why `metadata.pt`-at-the-end made the 21h job unrecoverable, and what
   an idempotent converter looks like
4. Cache keys as contracts: what invalidates a latent cache and what silently aliases
5. Spaced review of 01 (retrieval, ~1 week out)

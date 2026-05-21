# Architecture figures

Self-contained HTML + Mermaid sources for the architecture figures used
in advisor presentations and (later) the thesis.

## View

Open any `<name>.html` in a browser. Mermaid is loaded from a CDN and
lays the diagram out automatically — no compile step.

To export for slides, use the browser's print-to-PDF (works well in
Chrome / Firefox / Safari).

## Files

- `hyperalign.html` — HyperAlign hypernetwork.
- `avid.html` — AVID-style output adapter.
- `shortcut-training.html` — shortcut training data flow.
- `structural-encoder.html` — structural condition encoder + its two
  injection points.
- `shortcut-explainer.html` — pedagogical multi-section walkthrough:
  ODE trajectories → shortcut models → our adapter framing. Uses
  hand-drawn SVG instead of Mermaid because it needs smooth Bezier
  curves and free-positioned tangent/chord vectors.

## Style conventions

Shared across all four figures (no central preamble — Mermaid class
defs live inline at the bottom of each file's `<pre class="mermaid">`
block):

- **Frozen** modules: slate fill, thin border.
- **Trainable** modules: blue fill, thicker border.
- **Data / latent tensors**: green stadium (rounded) shape.
- **Conditioning / injection points**: amber.
- **Stop-grad / detached targets**: violet.
- **Composition / loss nodes**: red.

## Per-figure design notes

Design intent — what each figure asserts (with file:line code anchors),
what it deliberately omits, decisions taken — is documented in the
thesis vault at:

```
thesis-vault/30_Knowledge/writing/figure-<slug>.md
```

The vault notes are the source of truth for "why this figure looks like
this." Update the vault note before changing a figure substantively.

## Legacy Typst sources

The directory also contains `*.typ` files (`hyperalign.typ`, `avid.typ`,
`shortcut-training.typ`) from an earlier rendering attempt with Typst +
Fletcher. They are kept for reference but **the HTML files are the
canonical source**. The Typst attempt was abandoned because the grid
spacing parameter was too small for the node sizes and everything
overlapped; if you want to revisit Typst, fixing `spacing: (12pt, 10pt)`
→ `spacing: (60pt, 50pt)` (or similar) in each `.typ` should resolve
the layout, but the HTML versions are simpler to iterate.

## Editing tips

- Mermaid labels can contain inline HTML when `htmlLabels: true` (set
  in the `mermaid.initialize` call at the top of each file): `<b>`,
  `<small>`, `<sub>`, `<sup>`, `<code>`, `<br/>` all work.
- **Don't use `&quot;` HTML entities inside Mermaid labels.** The
  browser decodes them to `"` before Mermaid sees the source, which
  breaks the label string literal. Use plain text or backticks
  instead (e.g. write `cond.embedding`, not `cond[&quot;embedding&quot;]`).
- Avoid deeply nested `<sub>`/`<sup>` — they sometimes render oddly in
  Mermaid's HTML labels.
- **Keep each node label on one source line.** Use `<br/>` inside the
  label for visual line breaks; do not break the source string across
  physical lines. (Embedded newlines inside `"..."` labels are
  tolerated under default `securityLevel: 'strict'` but become parse
  errors under `'loose'`.)
- **Avoid the `Mermaid click ... call f()` directive with
  `securityLevel: 'loose'`** unless absolutely needed. We attempted an
  on-click zoom-panel for the HyperAlign decoder and the combination
  failed to render — likely the loose-security sanitizer path is less
  tolerant of HTML-in-labels. If you need interactive zoom-style
  drill-down, render the detail view as inline SVG below the diagram
  instead.

## Dependencies

None at build time. At view time: any modern browser. Mermaid is loaded
from `cdn.jsdelivr.net/npm/mermaid@10`.

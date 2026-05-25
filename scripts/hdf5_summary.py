"""Summarize the contents of an HDF5 file.

Recursively walks the file's group/dataset hierarchy and prints every key
along with per-dataset metadata (shape, dtype, chunks, compression, attrs)
and summary statistics (min/max/mean/std, NaN/inf counts).

Statistics are computed by sampling so the script stays fast on large
files (e.g. the multi-GB MetaWorld dumps in ``ds/``): datasets above
``--max-stat-elements`` are subsampled along the first axis before stats
are computed. Pass ``--full-stats`` to read everything instead.

Run:
    python scripts/hdf5_summary.py ds/metaworld_corner2.hdf5
    python scripts/hdf5_summary.py ds/metaworld_corner2_large.hdf5 \\
        --max-children 3 --max-depth 4
    python scripts/hdf5_summary.py ds/metaworld_corner2.hdf5 --no-stats
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "h5py is required for this script. "
            "Install with `pip install h5py` or `pip install -e .[dev]`."
        ) from exc
    return h5py


def _human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TiB"  # pragma: no cover - unreachable


def _format_attrs(attrs) -> str:
    if len(attrs) == 0:
        return ""
    parts = []
    for key in attrs:
        val = attrs[key]
        if isinstance(val, bytes):
            val = val.decode("utf-8", "replace")
        if isinstance(val, np.ndarray):
            val = np.array2string(val, threshold=8, precision=4)
        parts.append(f"{key}={val}")
    return "  {" + ", ".join(parts) + "}"


def _sample_dataset(dset, max_elements: int, full: bool) -> tuple[np.ndarray, bool]:
    """Return a (possibly subsampled) flat array of the dataset's values.

    Sampling is done by taking a strided slice along the first axis so we
    never materialize the whole dataset in memory. Returns the values and a
    flag indicating whether sampling occurred.
    """
    n_total = int(np.prod(dset.shape)) if dset.shape else 1
    if full or n_total <= max_elements or dset.ndim == 0:
        return np.asarray(dset[()]).ravel(), False

    per_row = max(1, int(np.prod(dset.shape[1:]))) if dset.ndim > 1 else 1
    rows_wanted = max(1, max_elements // per_row)
    n_rows = dset.shape[0]
    if rows_wanted >= n_rows:
        return np.asarray(dset[()]).ravel(), False

    stride = max(1, math.ceil(n_rows / rows_wanted))
    return np.asarray(dset[::stride]).ravel(), True


def _stats_line(dset, max_elements: int, full: bool) -> str:
    kind = dset.dtype.kind
    if kind in ("b",):  # boolean
        values, sampled = _sample_dataset(dset, max_elements, full)
        n = values.size
        if n == 0:
            return "stats: (empty)"
        true_frac = float(values.mean())
        tag = " [sampled]" if sampled else ""
        return f"stats: true={true_frac:.4f} false={1 - true_frac:.4f}{tag}"

    if kind not in ("i", "u", "f"):  # not numeric (strings, objects, etc.)
        try:
            sample = np.asarray(dset[(0,) * dset.ndim]) if dset.ndim else dset[()]
            preview = str(sample)
            if len(preview) > 60:
                preview = preview[:57] + "..."
            return f"stats: non-numeric ({dset.dtype}), e.g. {preview!r}"
        except Exception:
            return f"stats: non-numeric ({dset.dtype})"

    values, sampled = _sample_dataset(dset, max_elements, full)
    n = values.size
    if n == 0:
        return "stats: (empty)"

    finite = values[np.isfinite(values)]
    n_nan = int(np.isnan(values).sum()) if kind == "f" else 0
    n_inf = int(np.isinf(values).sum()) if kind == "f" else 0
    tag = " [sampled]" if sampled else ""

    if finite.size == 0:
        return f"stats: all non-finite (nan={n_nan} inf={n_inf}){tag}"

    line = (
        f"stats: min={finite.min():.4g} max={finite.max():.4g} "
        f"mean={finite.mean():.4g} std={finite.std():.4g}"
    )
    if n_nan or n_inf:
        line += f" nan={n_nan} inf={n_inf}"
    return line + tag


def _walk(
    name: str,
    node,
    *,
    depth: int,
    args: argparse.Namespace,
    counts: dict[str, int],
    h5py,
) -> None:
    indent = "  " * depth
    label = name.split("/")[-1] or "/"

    if isinstance(node, h5py.Group):
        counts["groups"] += 1
        print(f"{indent}{label}/{_format_attrs(node.attrs)}")
        if args.max_depth is not None and depth >= args.max_depth:
            n_children = len(node)
            if n_children:
                print(f"{indent}  ... {n_children} child(ren) (max depth reached)")
            return

        keys = sorted(node.keys())
        shown = keys if args.max_children <= 0 else keys[: args.max_children]
        for key in shown:
            _walk(
                f"{name}/{key}",
                node[key],
                depth=depth + 1,
                args=args,
                counts=counts,
                h5py=h5py,
            )
        hidden = len(keys) - len(shown)
        if hidden > 0:
            print(f"{indent}  ... {hidden} more child(ren) (use --max-children 0 for all)")
        return

    # Dataset
    counts["datasets"] += 1
    nbytes = node.dtype.itemsize * int(np.prod(node.shape)) if node.shape else node.dtype.itemsize
    counts["bytes"] += int(nbytes)
    meta = f"shape={node.shape} dtype={node.dtype}"
    if node.chunks is not None:
        meta += f" chunks={node.chunks}"
    if node.compression is not None:
        meta += f" compression={node.compression}"
    meta += f" size={_human_bytes(nbytes)}"
    print(f"{indent}{label}  [{meta}]{_format_attrs(node.attrs)}")

    if not args.no_stats:
        print(f"{indent}  {_stats_line(node, args.max_stat_elements, args.full_stats)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("path", type=Path, help="Path to the .hdf5 / .h5 file")
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="List keys and metadata only, skip computing statistics.",
    )
    parser.add_argument(
        "--full-stats",
        action="store_true",
        help="Read every element for stats instead of sampling (slow on large files).",
    )
    parser.add_argument(
        "--max-stat-elements",
        type=int,
        default=5_000_000,
        help="Subsample datasets larger than this many elements before computing stats.",
    )
    parser.add_argument(
        "--max-children",
        type=int,
        default=10,
        help="Max children to expand per group (0 = no limit). Useful for many episodes.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Max hierarchy depth to descend into (default: unlimited).",
    )
    args = parser.parse_args()

    if not args.path.exists():
        parser.error(f"file not found: {args.path}")

    h5py = _require_h5py()
    counts = {"groups": 0, "datasets": 0, "bytes": 0}

    print(f"File: {args.path}  ({_human_bytes(args.path.stat().st_size)} on disk)")
    with h5py.File(args.path, "r") as f:
        if len(f.attrs):
            print(f"root attrs:{_format_attrs(f.attrs)}")
        keys = sorted(f.keys())
        shown = keys if args.max_children <= 0 else keys[: args.max_children]
        for key in shown:
            _walk(key, f[key], depth=0, args=args, counts=counts, h5py=h5py)
        hidden = len(keys) - len(shown)
        if hidden > 0:
            print(f"... {hidden} more top-level group(s) (use --max-children 0 for all)")

    print(
        f"\nTotals: {counts['groups']} groups, {counts['datasets']} datasets, "
        f"{_human_bytes(counts['bytes'])} uncompressed"
    )


if __name__ == "__main__":
    main()

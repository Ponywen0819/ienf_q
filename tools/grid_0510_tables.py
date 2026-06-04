"""Print perturbation tables for every axis swept under output/grid_0510/.

For each subdirectory of GRID_ROOT listed in SUBDIR_AXIS this tool:
  1. filters per_combo/ to combos matching FIXED_PARAMS on every key except
     the swept axis (stale combos that vary other axes are ignored),
  2. picks the combo at FIXED_PARAMS[axis] as the reference,
  3. compares every non-reference value of the axis against the reference using
     a paired one-sided Wilcoxon signed-rank test (reference-is-better),
  4. prints one table per axis with HD95, AvgHD, and clDice means + p-values.

Reuses helpers from tools/compute_pvalues.py.

Run:
    uv run python tools/grid_0510_tables.py
    uv run python tools/grid_0510_tables.py --output-dir output/grid_0510/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_pvalues import (  # noqa: E402
    METRIC_SPECS,
    _hashable,
    load_combo_samples,
    paired_values,
    run_tests,
)

# ---------------------------------------------------------------------------
# Reference configuration. Mirrors staged_grid_search.FIXED_PARAMS.
# ---------------------------------------------------------------------------
FIXED_PARAMS: Dict[str, Any] = {
    "offset_px": 50,
    "bg_kernel_size": 5,
    "clahe_clip": 30.0,
    "clahe_grid": (768, 768),
    "sato_sigmas_start": 1,
    "sato_sigmas_stop": 4,
    "connectivity": 8,
    "prune_threshold": 20.0,
    "segment_length": 100.0,
    "stub_length_threshold": 3,
    "min_tree_components": 1,
}

# Each subdirectory of GRID_ROOT maps to exactly one swept axis. Combos that
# diverge from FIXED_PARAMS on any other key are ignored.
SUBDIR_AXIS: Dict[str, str] = {
    "bg": "bg_kernel_size",
    "clahe_clip": "clahe_clip",
    "clahe_grid": "clahe_grid",
    "sato_s": "sato_sigmas_start",
    "sato_e": "sato_sigmas_stop",
    "stub": "stub_length_threshold",
    "threshold": "prune_threshold",
}

GRID_ROOT = Path("output/grid_0510")
METRICS: List[str] = ["hd95", "avg_hd", "cldice"]
METRIC_LABELS: Dict[str, str] = {"hd95": "HD95", "avg_hd": "AvgHD", "cldice": "clDice"}
DEFAULT_OUTPUT_DIR = GRID_ROOT / "tables"


# ── Helpers ─────────────────────────────────────────────────────────────────


def _normalise(value: Any) -> Any:
    """Recursively convert tuples to lists so JSON-loaded combo params and
    FIXED_PARAMS (which uses tuples for clahe_grid) compare equal."""
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    return value


def load_combos(per_combo_dir: Path) -> List[Dict[str, Any]]:
    combos = []
    for combo_file in sorted(per_combo_dir.glob("combo_*.json")):
        with combo_file.open() as f:
            combos.append(json.load(f)["params"])
    return combos


def discover_values(
    combos: List[Dict[str, Any]], axis: str, held: Dict[str, Any]
) -> List[Any]:
    """Distinct values of ``axis`` across combos that match ``held`` on every
    other key. Returns original (non-normalised) values, sorted by their
    hashable normalised form."""
    held_norm = {k: _normalise(v) for k, v in held.items()}
    seen: Dict[Any, Any] = {}
    for c in combos:
        if axis not in c:
            continue
        if any(_normalise(c.get(k)) != v for k, v in held_norm.items()):
            continue
        v = c[axis]
        seen.setdefault(_hashable(_normalise(v)), v)

    def sort_key(hashable):
        return hashable if isinstance(hashable, tuple) else (hashable,)

    return [seen[k] for k in sorted(seen, key=sort_key)]


# ── Formatting ──────────────────────────────────────────────────────────────


def fmt_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        v = list(value)
        if len(v) == 2 and v[0] == v[1]:
            return str(v[0])
        return "x".join(str(x) for x in v)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def sig_mark(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "-"
    text = f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"
    return text + sig_mark(p)


def fmt_mean(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "-"
    return f"{v:.4f}"


# ── Table building ──────────────────────────────────────────────────────────


def build_table(grid_dir: Path, axis: str) -> Dict[str, Any]:
    """Build one perturbation table for ``axis`` in ``grid_dir``.

    All non-axis keys are held at FIXED_PARAMS. The reference combo is the
    one whose ``axis`` value equals FIXED_PARAMS[axis].
    """
    # Normalise tuples to lists so values compare equal against JSON-loaded params.
    ref_value = _normalise(FIXED_PARAMS[axis])
    held = {k: _normalise(v) for k, v in FIXED_PARAMS.items() if k != axis}

    combos = load_combos(grid_dir / "per_combo")
    values = discover_values(combos, axis, held)

    if not values:
        raise ValueError(
            f"no combos in {grid_dir} match FIXED_PARAMS on every key except '{axis}'"
        )

    ref_hashable = _hashable(_normalise(ref_value))
    if ref_hashable not in {_hashable(_normalise(v)) for v in values}:
        raise ValueError(
            f"reference value {ref_value!r} for '{axis}' not found in "
            f"{grid_dir} (discovered: {values})"
        )

    ref_match = {**held, axis: ref_value}
    ref_samples, _ = load_combo_samples(grid_dir, ref_match)

    rows: List[Dict[str, Any]] = []
    for value in values:
        is_ref = _hashable(_normalise(value)) == ref_hashable
        match = {**held, axis: value}
        cmp_samples, _ = load_combo_samples(grid_dir, match)

        row: Dict[str, Any] = {"value": value, "is_ref": is_ref}
        for metric in METRICS:
            metric_key, lower_is_better = METRIC_SPECS[metric]
            ref_vals, cmp_vals = paired_values(ref_samples, cmp_samples, metric_key)
            stats = run_tests(ref_vals, cmp_vals, lower_is_better)
            row[f"{metric}_mean"] = stats["cmp_mean"]
            row[f"{metric}_n"] = stats["n"]
            row[f"{metric}_p"] = (
                float("nan") if is_ref else stats["wilcoxon_p_ref_better"]
            )
        rows.append(row)

    return {
        "axis": axis,
        "grid_dir": str(grid_dir),
        "ref_value": ref_value,
        "rows": rows,
    }


# ── Output ──────────────────────────────────────────────────────────────────


def print_table(table: Dict[str, Any]) -> None:
    print()
    print(
        f"### {Path(table['grid_dir']).name}  ·  {table['axis']}  "
        f"[ref = {fmt_value(table['ref_value'])}]"
    )
    header = (
        f"{'value':>10} {'n':>4} "
        f"{'HD95':>9} {'HD95 p':>11}  "
        f"{'AvgHD':>9} {'AvgHD p':>11}  "
        f"{'clDice':>9} {'clDice p':>11}"
    )
    print(header)
    print("-" * len(header))
    for r in table["rows"]:
        tag = "  <- ref" if r["is_ref"] else ""
        print(
            f"{fmt_value(r['value']):>10} {r['hd95_n']:>4} "
            f"{fmt_mean(r['hd95_mean']):>9} {fmt_p(r['hd95_p']):>11}  "
            f"{fmt_mean(r['avg_hd_mean']):>9} {fmt_p(r['avg_hd_p']):>11}  "
            f"{fmt_mean(r['cldice_mean']):>9} {fmt_p(r['cldice_p']):>11}"
            f"{tag}"
        )
    print(
        "  p = one-sided Wilcoxon (reference is better);  *  p<0.05,  **  p<0.01"
    )


def write_csv(table: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["value", "is_ref", "n"]
    for m in METRICS:
        fieldnames += [f"{m}_mean", f"{m}_wilcoxon_p_ref_better"]
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for r in table["rows"]:
            row = [fmt_value(r["value"]), r["is_ref"], r["hd95_n"]]
            for m in METRICS:
                mean = r[f"{m}_mean"]
                p = r[f"{m}_p"]
                row.append("" if mean is None or not np.isfinite(mean) else f"{mean:.6f}")
                row.append("" if not np.isfinite(p) else f"{p:.6f}")
            writer.writerow(row)


def markdown_table(table: Dict[str, Any]) -> str:
    title = f"{Path(table['grid_dir']).name} · `{table['axis']}`"
    lines = [
        f"### {title}",
        "",
        f"Reference: **{fmt_value(table['ref_value'])}**. "
        "p = one-sided Wilcoxon (reference better); `*` p<0.05, `**` p<0.01.",
        "",
        "| value | n | HD95 | HD95 p | AvgHD | AvgHD p | clDice | clDice p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in table["rows"]:
        value = fmt_value(r["value"]) + (" **(ref)**" if r["is_ref"] else "")
        lines.append(
            f"| {value} | {r['hd95_n']} | "
            f"{fmt_mean(r['hd95_mean'])} | {fmt_p(r['hd95_p'])} | "
            f"{fmt_mean(r['avg_hd_mean'])} | {fmt_p(r['avg_hd_p'])} | "
            f"{fmt_mean(r['cldice_mean'])} | {fmt_p(r['cldice_p'])} |"
        )
    lines.append("")
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-root",
        type=Path,
        default=GRID_ROOT,
        help=f"Root containing per-axis subdirectories (default: {GRID_ROOT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write CSV + Markdown output (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print tables to stdout only; do not write CSV/Markdown files.",
    )
    args = parser.parse_args()

    md_parts = [
        "# Grid 0510 Perturbation Tables",
        "",
        "Each table sweeps one parameter; every non-reference value is compared "
        "against the reference (FIXED_PARAMS) with a paired one-sided Wilcoxon "
        "signed-rank test for HD95, AvgHD, and clDice.",
        "",
    ]

    written = 0
    for subdir_name, axis in SUBDIR_AXIS.items():
        grid_dir = args.grid_root / subdir_name
        if not (grid_dir / "per_combo").is_dir():
            print(f"[skip] {subdir_name}: per_combo/ missing", file=sys.stderr)
            continue
        try:
            table = build_table(grid_dir, axis)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[skip] {subdir_name}/{axis}: {exc}", file=sys.stderr)
            continue

        print_table(table)

        if not args.no_write:
            csv_name = f"{subdir_name}__{axis}.csv"
            write_csv(table, args.output_dir / csv_name)
            md_parts.append(markdown_table(table))
            written += 1

    if not args.no_write and written:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        md_path = args.output_dir / "tables.md"
        md_path.write_text("\n".join(md_parts), encoding="utf-8")
        print(
            f"\nWrote {written} CSV table(s) + {md_path.name} to {args.output_dir}/"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

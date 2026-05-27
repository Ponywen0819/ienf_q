"""Generate parameter-perturbation tables from the staged grid-search output.

For each swept parameter this builds one table whose rows are the values that
parameter took. Every non-reference value is compared against the reference
setting with a paired one-sided Wilcoxon signed-rank test (the same machinery
as tools/compute_pvalues.py), so each row reports:

    value | n | hd95 mean | hd95 p | clDice mean | clDice p

`hd95 p` / `clDice p` are one-sided Wilcoxon p-values for the hypothesis
"the reference setting is better than this value". A `*` marks p < 0.05 and
`**` marks p < 0.01 (i.e. perturbing away from the reference significantly
degrades that metric). When a value is significantly *better* than the
reference the one-sided p is near 1 -- check the mean column for direction.

Reference setting (one value held per parameter):
    bg_kernel_size    = 5
    clahe_clip        = 30.0
    clahe_grid        = [768, 768]
    sato_sigmas_start = 1
    sato_sigmas_stop  = 4
    prune_threshold   = 20.0

When a parameter shares a grid directory with another swept parameter, the
other parameter is held at its reference value (see `fixed` in TABLE_SPECS).

Run:
    uv run python tools/grid_perturbation_tables.py
    uv run python tools/grid_perturbation_tables.py --output-dir output/grid_0510/perturbation_tables
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# Reuse the paired-test helpers from compute_pvalues.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_pvalues import (  # noqa: E402
    METRIC_SPECS,
    _hashable,
    load_combo_samples,
    paired_values,
    run_tests,
)

# ---------------------------------------------------------------------------
# One entry per output table.
#   name      : parameter being perturbed (also the per-combo params key)
#   grid_dir  : grid-search output directory containing per_combo/
#   fixed     : other swept params held at their reference value
#   ref_value : the reference value of `name`
# ---------------------------------------------------------------------------
TABLE_SPECS: list[dict] = [
    {
        "name": "bg_kernel_size",
        "grid_dir": "output/grid_0510/bg",
        "fixed": {},
        "ref_value": 5,
    },
    {
        "name": "clahe_clip",
        "grid_dir": "output/grid_0510/clahe_grid",
        "fixed": {"clahe_grid": [768, 768]},
        "ref_value": 30.0,
    },
    {
        "name": "clahe_grid",
        "grid_dir": "output/grid_0510/clahe_grid",
        "fixed": {"clahe_clip": 30.0},
        "ref_value": [768, 768],
    },
    {
        "name": "sato_sigmas_start",
        "grid_dir": "output/grid_0510/sato_n",
        "fixed": {"sato_sigmas_stop": 4},
        "ref_value": 1,
    },
    {
        "name": "sato_sigmas_stop",
        "grid_dir": "output/grid_0510/sato_n",
        "fixed": {"sato_sigmas_start": 1},
        "ref_value": 4,
    },
    {
        "name": "prune_threshold",
        "grid_dir": "output/grid_0510/threshold",
        "fixed": {},
        "ref_value": 20.0,
    },
]

METRICS: list[str] = ["hd95", "cldice"]
DEFAULT_OUTPUT_DIR = Path("output/grid_0510/perturbation_tables")


def discover_values(grid_dir: Path, sweep_key: str, fixed: dict) -> list:
    """Distinct values of `sweep_key` among combos whose params match `fixed`."""
    per_combo_dir = grid_dir / "per_combo"
    if not per_combo_dir.is_dir():
        raise FileNotFoundError(f"per_combo/ not found under {grid_dir}")

    seen: dict = {}
    for combo_file in sorted(per_combo_dir.glob("combo_*.json")):
        with combo_file.open() as f:
            params = json.load(f).get("params", {})
        if sweep_key not in params:
            continue
        if any(params.get(k) != v for k, v in fixed.items()):
            continue
        value = params[sweep_key]
        seen.setdefault(_hashable(value), value)
    return [seen[k] for k in sorted(seen, key=lambda x: (x if isinstance(x, tuple) else (x,)))]


def fmt_value(value) -> str:
    """Compact display string for a swept value."""
    if isinstance(value, list):
        # Square CLAHE grids -> show the single side length.
        if len(value) == 2 and value[0] == value[1]:
            return str(value[0])
        return "x".join(str(v) for v in value)
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


def build_table(spec: dict) -> dict:
    """Compute all rows for one perturbation table."""
    sweep_key = spec["name"]
    grid_dir = Path(spec["grid_dir"])
    fixed = spec["fixed"]
    ref_value = spec["ref_value"]

    values = discover_values(grid_dir, sweep_key, fixed)
    ref_hashable = _hashable(ref_value)
    if ref_hashable not in {_hashable(v) for v in values}:
        raise ValueError(
            f"reference value {ref_value!r} for '{sweep_key}' not found in "
            f"{grid_dir} (discovered: {values})"
        )

    ref_match = {**fixed, sweep_key: ref_value}
    ref_samples, _ = load_combo_samples(grid_dir, ref_match)

    rows: list[dict] = []
    for value in values:
        is_ref = _hashable(value) == ref_hashable
        match = {**fixed, sweep_key: value}
        cmp_samples, _ = load_combo_samples(grid_dir, match)

        row: dict = {"value": value, "is_ref": is_ref}
        for metric in METRICS:
            metric_key, lower_is_better = METRIC_SPECS[metric]
            ref_vals, cmp_vals = paired_values(ref_samples, cmp_samples, metric_key)
            stats = run_tests(ref_vals, cmp_vals, lower_is_better)
            row[f"{metric}_mean"] = stats["cmp_mean"]
            row[f"{metric}_n"] = stats["n"]
            # One-sided Wilcoxon p: "reference is better than this value".
            row[f"{metric}_p"] = float("nan") if is_ref else stats["wilcoxon_p_ref_better"]
        rows.append(row)

    return {
        "name": sweep_key,
        "grid_dir": str(grid_dir),
        "fixed": fixed,
        "ref_value": ref_value,
        "rows": rows,
    }


def print_table(table: dict) -> None:
    fixed_str = (
        "  (" + ", ".join(f"{k}={fmt_value(v)}" for k, v in table["fixed"].items()) + ")"
        if table["fixed"]
        else ""
    )
    print()
    print(f"### {table['name']}  [ref = {fmt_value(table['ref_value'])}]{fixed_str}")
    header = (
        f"{'value':>10} {'n':>4} "
        f"{'hd95_mean':>11} {'hd95_p':>13} "
        f"{'cldice_mean':>13} {'cldice_p':>13}"
    )
    print(header)
    print("-" * len(header))
    for r in table["rows"]:
        tag = "  <- ref" if r["is_ref"] else ""
        print(
            f"{fmt_value(r['value']):>10} {r['hd95_n']:>4} "
            f"{r['hd95_mean']:>11.4f} {fmt_p(r['hd95_p']):>13} "
            f"{r['cldice_mean']:>13.4f} {fmt_p(r['cldice_p']):>13}"
            f"{tag}"
        )


def write_csv(table: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["value", "is_ref", "n", "hd95_mean", "hd95_wilcoxon_p_ref_better",
             "cldice_mean", "cldice_wilcoxon_p_ref_better"]
        )
        for r in table["rows"]:
            writer.writerow([
                fmt_value(r["value"]),
                r["is_ref"],
                r["hd95_n"],
                f"{r['hd95_mean']:.6f}",
                "" if not np.isfinite(r["hd95_p"]) else f"{r['hd95_p']:.6f}",
                f"{r['cldice_mean']:.6f}",
                "" if not np.isfinite(r["cldice_p"]) else f"{r['cldice_p']:.6f}",
            ])


def markdown_table(table: dict) -> str:
    fixed_str = (
        " (" + ", ".join(f"`{k}`={fmt_value(v)}" for k, v in table["fixed"].items()) + ")"
        if table["fixed"]
        else ""
    )
    lines = [
        f"### `{table['name']}`",
        "",
        f"Reference value: **{fmt_value(table['ref_value'])}**{fixed_str}. "
        f"p = one-sided Wilcoxon (reference better); `*` p<0.05, `**` p<0.01.",
        "",
        "| value | n | hd95 mean | hd95 p | clDice mean | clDice p |",
        "|---|---|---|---|---|---|",
    ]
    for r in table["rows"]:
        value = fmt_value(r["value"]) + (" **(ref)**" if r["is_ref"] else "")
        lines.append(
            f"| {value} | {r['hd95_n']} | {r['hd95_mean']:.4f} | {fmt_p(r['hd95_p'])} "
            f"| {r['cldice_mean']:.4f} | {fmt_p(r['cldice_p'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSV/Markdown output (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    md_parts = [
        "# Parameter Perturbation Tables",
        "",
        "Each table sweeps one preprocessing/reconstruction parameter; every "
        "non-reference value is compared to the reference setting with a paired "
        "one-sided Wilcoxon signed-rank test.",
        "",
    ]

    for spec in TABLE_SPECS:
        try:
            table = build_table(spec)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[skip] {spec['name']}: {exc}", file=sys.stderr)
            continue

        print_table(table)
        csv_path = args.output_dir / f"{table['name']}.csv"
        write_csv(table, csv_path)
        md_parts.append(markdown_table(table))

    md_path = args.output_dir / "perturbation_tables.md"
    md_path.write_text("\n".join(md_parts), encoding="utf-8")

    print(f"\nCSV tables + {md_path.name} written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

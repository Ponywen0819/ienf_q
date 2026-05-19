"""Compute a wide comparison table (hd95 / cldice / mae / rmse) against a
reference setting, marking cells with `*` when the comparison is
*significantly worse* than the reference (paired one-sided Wilcoxon
signed-rank test, p < 0.05). Direction of "worse" is per-metric:
hd95 / mae / rmse — cmp > ref; cldice — cmp < ref.

This is the *legacy-format* counterpart of ``tools/compute_pvalues.py``: it
reads the per-setting ``results.json`` produced by the older evaluation
pipeline (one subdirectory per setting, ``samples`` keyed by sample_id),
rather than the gridsearch ``per_combo/combo_*.json`` files.

The reference row is rendered at the bottom of the table (no p-tests
against itself).

Edit the ``REFERENCE``, ``COMPARISONS`` and ``METRICS`` lists below.

Run:
    uv run python tools/compute_pvalues_legacy.py
    uv run python tools/compute_pvalues_legacy.py --output output/pvalues_legacy.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration. Each path should be a directory containing a `results.json`
# (legacy format: { "samples": { sample_id: {hd95, cldice, count_error, ...}}}).
# REFERENCE is the row rendered at the bottom of the table.
# ---------------------------------------------------------------------------
GRID_DIR = Path("output/ref")

REFERENCE: Path = GRID_DIR / "annotation_grow"

COMPARISONS: list[Path] = [
    GRID_DIR / "skeleton",
    GRID_DIR / "mst",
]

METRICS: list[str] = ["hd95", "cldice", "mae", "rmse"]

ALPHA: float = 0.05
# ---------------------------------------------------------------------------

# Per-metric: (per-sample extractor, aggregate function, lower_is_better)
#   - per-sample extractor: dict -> float | None (None to skip the sample)
#   - aggregate function: 1-D float array -> float (for the displayed cell)
# Wilcoxon is always run on the raw per-sample values returned by the extractor.
METRIC_SPECS: dict[str, tuple] = {
    "hd95":   (lambda s: s.get("hd95"),
               lambda v: float(np.mean(v)),
               True),
    "cldice": (lambda s: s.get("cldice"),
               lambda v: float(np.mean(v)),
               False),
    "mae":    (lambda s: s.get("count_error"),          # already |pred - gt|
               lambda v: float(np.mean(v)),
               True),
    "rmse":   (lambda s: (None if s.get("count_error") is None
                          else float(s["count_error"]) ** 2),
               lambda v: float(math.sqrt(np.mean(v))),
               True),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def resolve_results_json(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "results.json"
        if not candidate.exists():
            raise FileNotFoundError(f"No results.json under {path}")
        return candidate
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    return path


def load_samples(results_json: Path) -> dict[str, dict]:
    with results_json.open() as f:
        data = json.load(f)
    return {
        sid: s
        for sid, s in data.get("samples", {}).items()
        if s.get("status") == "success"
    }


def display_name(path: Path) -> str:
    return path.parent.name if path.name == "results.json" else path.name


# ---------------------------------------------------------------------------
# Per-setting aggregates + paired Wilcoxon vs. reference
# ---------------------------------------------------------------------------
def extract_values(samples: dict[str, dict], extractor) -> dict[str, float]:
    """Return {sample_id: value} keeping only finite values."""
    out: dict[str, float] = {}
    for sid, s in samples.items():
        v = extractor(s)
        if v is None:
            continue
        v = float(v)
        if not np.isfinite(v):
            continue
        out[sid] = v
    return out


def aggregate(values: dict[str, float], agg) -> float:
    if not values:
        return float("nan")
    return agg(np.asarray(list(values.values()), dtype=float))


def wilcoxon_p_cmp_worse(
    ref_vals: dict[str, float],
    cmp_vals: dict[str, float],
    lower_is_better: bool,
) -> tuple[float, int]:
    """Paired one-sided Wilcoxon testing whether cmp is *worse* than ref.

    lower_is_better=True  -> alternative: cmp > ref (cmp is larger = worse)
    lower_is_better=False -> alternative: cmp < ref (cmp is smaller = worse)
    """
    common = sorted(set(ref_vals) & set(cmp_vals))
    if len(common) < 2:
        return float("nan"), len(common)
    r = np.asarray([ref_vals[s] for s in common])
    c = np.asarray([cmp_vals[s] for s in common])
    if np.all(r == c):
        return float("nan"), len(common)
    alternative = "greater" if lower_is_better else "less"
    try:
        res = stats.wilcoxon(c, r, zero_method="wilcox", alternative=alternative)
    except ValueError:
        return float("nan"), len(common)
    return float(res.pvalue), len(common)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------
def fmt_cell(metric: str, value: float, p: float | None) -> str:
    if not np.isfinite(value):
        return "nan"
    if metric == "cldice":
        body = f"{value:.4f}"
    else:
        body = f"{value:.3f}"
    if p is not None and np.isfinite(p) and p < ALPHA:
        return body + "*"
    return body


def print_table(
    settings_rows: list[dict],
    ref_row: dict,
    metrics: list[str],
) -> None:
    name_w = max(
        max((len(r["name"]) for r in settings_rows), default=0),
        len(ref_row["name"]) + len(" (ref)"),
        12,
    )
    col_w = 12  # value+marker fits comfortably

    header = f"{'setting':<{name_w}} {'n':>3}"
    for m in metrics:
        header += f" {m:>{col_w}}"
    bar = "-" * len(header)

    print()
    print(header)
    print(bar)
    for r in settings_rows:
        line = f"{r['name']:<{name_w}} {r['n']:>3}"
        for m in metrics:
            line += f" {r['cells'][m]:>{col_w}}"
        print(line)
    print(bar)
    line = f"{ref_row['name'] + ' (ref)':<{name_w}} {ref_row['n']:>3}"
    for m in metrics:
        line += f" {ref_row['cells'][m]:>{col_w}}"
    print(line)
    print()
    print(
        f"  * = paired one-sided Wilcoxon p < {ALPHA:g}: "
        f"comparison is significantly WORSE than reference"
    )
    print("  (lower is better: hd95, mae, rmse; higher is better: cldice)")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
def write_csv(
    settings_rows: list[dict],
    ref_row: dict,
    metrics: list[str],
    out_path: Path,
) -> None:
    fieldnames = ["setting", "is_reference", "n"]
    for m in metrics:
        fieldnames += [m, f"{m}_p_cmp_worse_than_ref"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in settings_rows + [ref_row]:
            row = {
                "setting": r["name"],
                "is_reference": r is ref_row,
                "n": r["n"],
            }
            for m in metrics:
                row[m] = r["values"][m]
                row[f"{m}_p_cmp_worse_than_ref"] = r["pvalues"][m]
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_row(
    path: Path,
    metrics: list[str],
    ref_extracted: dict[str, dict[str, float]] | None,
) -> dict:
    samples = load_samples(resolve_results_json(path))
    extracted: dict[str, dict[str, float]] = {}
    values: dict[str, float] = {}
    pvalues: dict[str, float | None] = {}
    cells: dict[str, str] = {}

    for metric in metrics:
        extractor, agg, lower_is_better = METRIC_SPECS[metric]
        per_sample = extract_values(samples, extractor)
        extracted[metric] = per_sample
        v = aggregate(per_sample, agg)
        values[metric] = v

        if ref_extracted is None:
            pvalues[metric] = None
        else:
            p, _ = wilcoxon_p_cmp_worse(
                ref_extracted[metric], per_sample, lower_is_better
            )
            pvalues[metric] = p
        cells[metric] = fmt_cell(metric, v, pvalues[metric])

    return {
        "name": display_name(path),
        "n": len(samples),
        "values": values,
        "pvalues": pvalues,
        "cells": cells,
        "extracted": extracted,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional CSV output path.")
    args = parser.parse_args()

    try:
        ref_row = build_row(REFERENCE, METRICS, ref_extracted=None)
    except FileNotFoundError as exc:
        print(f"Reference not found: {exc}", file=sys.stderr)
        return 1

    settings_rows: list[dict] = []
    for path in COMPARISONS:
        try:
            row = build_row(path, METRICS, ref_extracted=ref_row["extracted"])
        except FileNotFoundError as exc:
            print(f"  skip: {exc}", file=sys.stderr)
            continue
        settings_rows.append(row)

    if not settings_rows:
        print("No comparison rows produced.", file=sys.stderr)
        return 1

    print_table(settings_rows, ref_row, METRICS)

    if args.output is not None:
        write_csv(settings_rows, ref_row, METRICS, args.output)
        print(f"CSV written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

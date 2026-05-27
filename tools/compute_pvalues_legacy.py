"""Compute a wide comparison table (hd95 / cldice / mae / rmse) against a
reference setting, marking cells with `*` when the comparison is
*significantly worse* than the reference.

Each comparison is tested with one or more paired one-sided significance
tests (select with ``--tests``):

  * ``wilcoxon`` — Wilcoxon signed-rank test (non-parametric, the original).
  * ``ttest``    — paired Student's t-test (parametric).

Both test, per metric, whether the comparison is *worse* than the reference.
Direction of "worse" is per-metric: hd95 / mae / rmse — cmp > ref;
cldice — cmp < ref. Running both lets you check whether a conclusion holds
regardless of the test's parametric assumptions.

This is the *legacy-format* counterpart of ``tools/compute_pvalues.py``: it
reads the per-setting ``results.json`` produced by the older evaluation
pipeline (one subdirectory per setting, ``samples`` keyed by sample_id),
rather than the gridsearch ``per_combo/combo_*.json`` files.

The reference row is rendered at the bottom of the table (no p-tests
against itself).

Edit the ``REFERENCE``, ``COMPARISONS`` and ``METRICS`` lists below.

Run:
    uv run python tools/compute_pvalues_legacy.py
    uv run python tools/compute_pvalues_legacy.py --tests wilcoxon ttest
    uv run python tools/compute_pvalues_legacy.py --tests ttest
    uv run python tools/compute_pvalues_legacy.py --output output/pvalues_legacy.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration. Each path should be a directory containing a `results.json`
# (legacy format: { "samples": { sample_id: {hd95, cldice, count_error, ...}}}).
# REFERENCE is the row rendered at the bottom of the table.
# ---------------------------------------------------------------------------
# GRID_DIR = Path("output/orig")
GRID_DIR = Path("output/ref")

# REFERENCE: Path = GRID_DIR / "annotation_cldice_3"
# REFERENCE: Path = GRID_DIR / "annotation_0.32"
# REFERENCE: Path = GRID_DIR / "annotation_um_1.28"
# REFERENCE: Path = GRID_DIR / "annotation_grow_3"
REFERENCE: Path = GRID_DIR / "annotation_grow"

COMPARISONS: list[Path] = [
    GRID_DIR / "skeleton",
    # GRID_DIR / "annotation_grow_0",
    # GRID_DIR / "annotation_grow",
    # GRID_DIR / "annotation_grow_7",
    GRID_DIR / "mst_15",
    # GRID_DIR / "skel",
    # GRID_DIR / "mst_um_1.28",
    # GRID_DIR / "mst_cldice_1",
]

# METRICS: list[str] = ["hausdorff","hd95", "cldice","cldice_tprec","cldice_tsens","mae", "rmse"]
METRICS: list[str] = ["hausdorff"]


ALPHA: float = 0.05
# ---------------------------------------------------------------------------

# Per-metric: (per-sample extractor, aggregate function, lower_is_better)
#   - per-sample extractor: dict -> float | None (None to skip the sample)
#   - aggregate function: 1-D float array -> float (for the displayed cell)
# The significance tests are always run on the raw per-sample values returned
# by the extractor.
METRIC_SPECS: dict[str, tuple] = {
    "hausdorff":   (lambda s: s.get("hausdorff_distance"),
               lambda v: float(np.mean(v)),
               True),
    "hd95":   (lambda s: s.get("hd95"),
               lambda v: float(np.mean(v)),
               True),
    "cldice": (lambda s: s.get("cldice"),
               lambda v: float(np.mean(v)),
               False),

    "cldice_tprec": (lambda s: s.get("cldice_tprec"),
               lambda v: float(np.mean(v)),
               False),

    "cldice_tsens": (lambda s: s.get("cldice_tsens"),
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
# Per-setting aggregates + paired significance tests vs. reference
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


def _paired_arrays(
    ref_vals: dict[str, float],
    cmp_vals: dict[str, float],
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Pair ref/cmp by common sample IDs.

    Returns (ref_array, cmp_array, n_common); arrays are None when there are
    fewer than 2 paired samples.
    """
    common = sorted(set(ref_vals) & set(cmp_vals))
    if len(common) < 2:
        return None, None, len(common)
    r = np.asarray([ref_vals[s] for s in common], dtype=float)
    c = np.asarray([cmp_vals[s] for s in common], dtype=float)
    return r, c, len(common)


def wilcoxon_p_cmp_worse(
    ref_vals: dict[str, float],
    cmp_vals: dict[str, float],
    lower_is_better: bool,
) -> tuple[float, int]:
    """Paired one-sided Wilcoxon signed-rank test: is cmp *worse* than ref?

    lower_is_better=True  -> alternative: cmp > ref (cmp is larger = worse)
    lower_is_better=False -> alternative: cmp < ref (cmp is smaller = worse)
    """
    r, c, n = _paired_arrays(ref_vals, cmp_vals)
    if r is None or np.all(r == c):
        return float("nan"), n
    alternative = "greater" if lower_is_better else "less"
    try:
        res = stats.wilcoxon(c, r, zero_method="wilcox", alternative=alternative)
    except ValueError:
        return float("nan"), n
    return float(res.pvalue), n  # type: ignore[union-attr]


def ttest_p_cmp_worse(
    ref_vals: dict[str, float],
    cmp_vals: dict[str, float],
    lower_is_better: bool,
) -> tuple[float, int]:
    """Paired one-sided Student's t-test: is cmp *worse* than ref?

    Parametric counterpart of :func:`wilcoxon_p_cmp_worse` — same alternative
    hypothesis, but assumes the per-sample differences are roughly normal.

    lower_is_better=True  -> alternative: cmp > ref (cmp is larger = worse)
    lower_is_better=False -> alternative: cmp < ref (cmp is smaller = worse)
    """
    r, c, n = _paired_arrays(ref_vals, cmp_vals)
    if r is None or np.all(r == c):
        return float("nan"), n
    alternative = "greater" if lower_is_better else "less"
    try:
        res = stats.ttest_rel(c, r, alternative=alternative)
    except ValueError:
        return float("nan"), n
    p = float(res.pvalue)  # type: ignore[union-attr]
    return (p if np.isfinite(p) else float("nan")), n


# A paired test: (ref_vals, cmp_vals, lower_is_better) -> (p_value, n_common).
PairedTest = Callable[
    [dict[str, float], dict[str, float], bool], tuple[float, int]
]

# Registry of available paired tests: name -> (label, abbreviation, function).
TESTS: dict[str, tuple[str, str, PairedTest]] = {
    "wilcoxon": ("Wilcoxon signed-rank", "W", wilcoxon_p_cmp_worse),
    "ttest": ("paired t-test", "t", ttest_p_cmp_worse),
}
DEFAULT_TESTS: list[str] = ["wilcoxon", "ttest"]


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------
def fmt_cell(metric: str, value: float, p: float | None) -> str:
    """Format the value cell; append `*` when the primary test flags it worse."""
    if not np.isfinite(value):
        return "nan"
    if metric == "cldice":
        body = f"{value:.4f}"
    else:
        body = f"{value:.3f}"
    if p is not None and np.isfinite(p) and p < ALPHA:
        return body + "*"
    return body


def fmt_p(p: float | None) -> str:
    """Render a one-sided p-value; '-' for the reference row / undefined.

    Appends `*` when p < ALPHA (significantly worse).
    """
    if p is None or not np.isfinite(p):
        return "-"
    body = f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"
    return body + "*" if p < ALPHA else body


def print_table(
    settings_rows: list[dict],
    ref_row: dict,
    metrics: list[str],
    tests: list[str],
) -> None:
    name_w = max(
        max((len(r["name"]) for r in settings_rows), default=0),
        len(ref_row["name"]) + len(" (ref)"),
        12,
    )
    val_w = 11  # value+marker fits comfortably
    p_w = 11    # one-sided p-value column (value + '*')

    header = f"{'setting':<{name_w}} {'n':>3}"
    for m in metrics:
        header += f" {m:>{val_w}}"
        for t in tests:
            abbr = TESTS[t][1]
            header += f" {m + ' p(' + abbr + ')':>{p_w}}"
    bar = "-" * len(header)

    def _row(r: dict) -> str:
        line = f"{r['name']:<{name_w}} {r['n']:>3}"
        for m in metrics:
            line += f" {r['cells'][m]:>{val_w}}"
            for t in tests:
                line += f" {fmt_p(r['pvalues'][m][t]):>{p_w}}"
        return line

    print()
    print(header)
    print(bar)
    for r in settings_rows:
        print(_row(r))
    print(bar)
    print(_row({**ref_row, "name": ref_row["name"] + " (ref)"}))
    print()
    print("  paired one-sided p-value: comparison is WORSE than reference")
    for t in tests:
        label, abbr, _ = TESTS[t]
        print(f"    p({abbr}) = {label}")
    primary_abbr = TESTS[tests[0]][1]
    print(
        f"  * on a p-value = that p < {ALPHA:g} (significantly worse); "
        f"'-' = reference row"
    )
    print(
        f"  * on a value   = primary test p(of {primary_abbr}) < {ALPHA:g}"
    )
    print("  (lower is better: hd95, mae, rmse; higher is better: cldice)")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
def write_csv(
    settings_rows: list[dict],
    ref_row: dict,
    metrics: list[str],
    tests: list[str],
    out_path: Path,
) -> None:
    fieldnames = ["setting", "is_reference", "n"]
    for m in metrics:
        fieldnames.append(m)
        for t in tests:
            fieldnames.append(f"{m}_p_{t}_cmp_worse_than_ref")

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
                for t in tests:
                    row[f"{m}_p_{t}_cmp_worse_than_ref"] = r["pvalues"][m][t]
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_row(
    path: Path,
    metrics: list[str],
    tests: list[str],
    ref_extracted: dict[str, dict[str, float]] | None,
) -> dict:
    samples = load_samples(resolve_results_json(path))
    extracted: dict[str, dict[str, float]] = {}
    values: dict[str, float] = {}
    pvalues: dict[str, dict[str, float | None]] = {}
    cells: dict[str, str] = {}

    for metric in metrics:
        extractor, agg, lower_is_better = METRIC_SPECS[metric]
        per_sample = extract_values(samples, extractor)
        extracted[metric] = per_sample
        v = aggregate(per_sample, agg)
        values[metric] = v

        metric_pvalues: dict[str, float | None] = {}
        for test_name in tests:
            if ref_extracted is None:
                metric_pvalues[test_name] = None
            else:
                test_fn = TESTS[test_name][2]
                p, _ = test_fn(ref_extracted[metric], per_sample, lower_is_better)
                metric_pvalues[test_name] = p
        pvalues[metric] = metric_pvalues

        # `*` on the value cell is driven by the primary (first) test.
        primary_p = metric_pvalues[tests[0]]
        cells[metric] = fmt_cell(metric, v, primary_p)

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
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=list(TESTS.keys()),
        default=DEFAULT_TESTS,
        help=(
            "Paired significance test(s) to run, in order. The first one is "
            "the 'primary' test that drives the `*` marker on value cells. "
            f"Default: {' '.join(DEFAULT_TESTS)}"
        ),
    )
    args = parser.parse_args()
    tests: list[str] = args.tests

    try:
        ref_row = build_row(REFERENCE, METRICS, tests, ref_extracted=None)
    except FileNotFoundError as exc:
        print(f"Reference not found: {exc}", file=sys.stderr)
        return 1

    settings_rows: list[dict] = []
    for path in COMPARISONS:
        try:
            row = build_row(
                path, METRICS, tests, ref_extracted=ref_row["extracted"]
            )
        except FileNotFoundError as exc:
            print(f"  skip: {exc}", file=sys.stderr)
            continue
        settings_rows.append(row)

    if not settings_rows:
        print("No comparison rows produced.", file=sys.stderr)
        return 1

    print_table(settings_rows, ref_row, METRICS, tests)

    if args.output is not None:
        write_csv(settings_rows, ref_row, METRICS, tests, args.output)
        print(f"CSV written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


# setting                 n   hausdorff hausdorff p(W)        hd95   hd95 p(W)      cldice cldice p(W) cldice_tprec cldice_tprec p(W) cldice_tsens cldice_tsens p(W)         mae    mae p(W)        rmse   rmse p(W)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# skeleton               77      4.643*   1.23e-14*     15.239*   1.80e-14*     0.7117*   1.23e-14*       0.937      0.9945      0.587*   1.23e-14*     21.948*   1.78e-14*     24.644*   1.80e-14*
# mst_n                  77       1.806      0.0729      2.709*   9.54e-07*      0.9049      0.5030       0.931      0.9986       0.882      0.3969       6.701      0.0539      8.614*     0.0495*
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# annotation_grow (ref)  77       1.778           -       2.471           -      0.9042           -       0.929           -       0.881           -       5.532           -       7.274           -


# setting                 n   hausdorff hausdorff p(t)        hd95   hd95 p(t)      cldice cldice p(t) cldice_tprec cldice_tprec p(t) cldice_tsens cldice_tsens p(t)         mae    mae p(t)        rmse   rmse p(t)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# skeleton               77      4.643*   4.91e-17*     15.239*   7.71e-12*     0.7117*   2.95e-29*       0.937      0.9972      0.587*   1.07e-35*     21.948*   3.03e-22*     24.644*   1.94e-13*
# mst_n                  77      1.806*     0.0264*      2.709*   3.06e-06*      0.9049      0.6911       0.931      0.9985       0.882      0.5075      6.701*     0.0313*      8.614*     0.0298*
# mst                    77      2.470*   7.80e-22*      6.584*   1.29e-22*     0.8654*   3.05e-23*      0.826*   1.36e-36*       0.911      1.0000     12.649*   5.50e-10*     15.158*   2.65e-08*
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# annotation_grow (ref)  77       1.778           -       2.471           -      0.9042           -       0.929           -       0.881           -       5.532           -       7.274           -


# setting                 n   hausdorff hausdorff p(t)        hd95   hd95 p(t)      cldice cldice p(t) cldice_tprec cldice_tprec p(t) cldice_tsens cldice_tsens p(t)         mae    mae p(t)        rmse   rmse p(t)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# skeleton               77      4.643*   4.91e-17*     15.239*   7.71e-12*     0.7117*   2.95e-29*       0.937      0.9972      0.587*   1.07e-35*     21.948*   3.03e-22*     24.644*   1.94e-13*
# mst.                   77      1.944*   3.84e-07*      3.044*   5.91e-07*     0.8188*   2.56e-33*      0.895*   6.10e-16*      0.757*   1.75e-39*       6.000      0.1593       7.945      0.0910
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# annotation_grow (ref)  77       1.778           -       2.471           -      0.9042           -       0.929           -       0.881           -       5.532           -       7.274           -

# setting                 n   hausdorff hausdorff p(W) hausdorff p(t)        hd95   hd95 p(W)   hd95 p(t)      cldice cldice p(W) cldice p(t) cldice_tprec cldice_tprec p(W) cldice_tprec p(t) cldice_tsens cldice_tsens p(W) cldice_tsens p(t)         mae    mae p(W)    mae p(t)        rmse   rmse p(W)   rmse p(t)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# skeleton               77      4.643*   1.23e-14*   4.91e-17*     15.239*   1.80e-14*   7.71e-12*     0.7117*   1.23e-14*   2.95e-29*       0.937      0.9945      0.9972      0.587*   1.23e-14*   1.07e-35*     21.948*   1.78e-14*   3.03e-22*     24.644*   1.80e-14*   1.94e-13*
# annotation_grow_0      77       1.778           -           -       2.471           -           -     0.8258*   1.23e-14*   3.86e-21*      0.879*   1.23e-14*   1.55e-20*      0.782*   1.23e-14*   2.57e-21*       5.753      0.2172      0.3608       7.335      0.0846      0.4708
# mst_15                 77      1.944*   9.20e-07*   3.84e-07*      3.044*   3.17e-06*   5.91e-07*     0.8188*   1.23e-14*   2.56e-33*      0.895*   2.04e-13*   6.10e-16*      0.757*   1.23e-14*   1.75e-39*       6.000      0.2439      0.1593       7.945      0.2261      0.0910
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# annotation_grow (ref)  77       1.778           -           -       2.471           -           -      0.9042           -           -       0.929           -           -       0.881           -           -       5.532           -           -       7.274           -           -


# setting              n   hausdorff hausdorff p(W) hausdorff p(t)        hd95   hd95 p(W)   hd95 p(t)      cldice cldice p(W) cldice p(t) cldice_tprec cldice_tprec p(W) cldice_tprec p(t) cldice_tsens cldice_tsens p(W) cldice_tsens p(t)         mae    mae p(W)    mae p(t)        rmse   rmse p(W)   rmse p(t)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# skeleton            77      4.643*   1.23e-14*   4.91e-17*     15.239*   1.80e-14*   7.71e-12*     0.7117*   1.44e-14*   1.51e-22*       0.937      1.0000      1.0000      0.587*   1.28e-14*   2.02e-31*     21.948*   1.78e-14*   3.03e-22*     24.644*   1.80e-14*   1.94e-13*
# annotation_grow_0   77       1.778           -           -       2.471           -           -      0.8258           -           -       0.879           -           -       0.782           -           -       5.753      0.2172      0.3608       7.335      0.0846      0.4708
# mst_15              77      1.944*   9.20e-07*   3.84e-07*      3.044*   3.17e-06*   5.91e-07*     0.8188*     0.0328*     0.0300*       0.895      1.0000      1.0000      0.757*   2.02e-05*   1.44e-05*       6.000      0.2439      0.1593       7.945      0.2261      0.0910
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# annotation_n (ref)  77       1.778           -           -       2.471           -           -      0.8258           -           -       0.879           -           -       0.782           -           -       5.532           -           -       7.274           -           -


# setting                 n   hausdorff hausdorff p(W) hausdorff p(t)        hd95   hd95 p(W)   hd95 p(t)      cldice cldice p(W) cldice p(t) cldice_tprec cldice_tprec p(W) cldice_tprec p(t) cldice_tsens cldice_tsens p(W) cldice_tsens p(t)         mae    mae p(W)    mae p(t)        rmse   rmse p(W)   rmse p(t)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# skeleton               77      4.643*   1.23e-14*   4.91e-17*     15.239*   1.80e-14*   7.71e-12*     0.7117*   1.23e-14*   2.95e-29*       0.937      0.9945      0.9972      0.587*   1.23e-14*   1.07e-35*     21.948*   1.78e-14*   3.03e-22*     24.644*   1.80e-14*   1.94e-13*
# mst_15                 77      1.944*   9.20e-07*   3.84e-07*      3.044*   3.17e-06*   5.91e-07*     0.8188*   1.23e-14*   2.56e-33*      0.895*   2.04e-13*   6.10e-16*      0.757*   1.23e-14*   1.75e-39*       6.000      0.2439      0.1593       7.945      0.2261      0.0910
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# annotation_grow (ref)  77       1.778           -           -       2.471           -           -      0.9042           -           -       0.929           -           -       0.881           -           -       5.532           -           -       7.274           -           -


# setting                 n   hausdorff hausdorff p(W) hausdorff p(t)        hd95   hd95 p(W)   hd95 p(t)      cldice cldice p(W) cldice p(t) cldice_tprec cldice_tprec p(W) cldice_tprec p(t) cldice_tsens cldice_tsens p(W) cldice_tsens p(t)         mae    mae p(W)    mae p(t)        rmse   rmse p(W)   rmse p(t)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# skeleton               77      4.643*   1.23e-14*   4.91e-17*     15.239*   1.80e-14*   7.71e-12*     0.7117*   1.23e-14*   2.95e-29*       0.937      0.9945      0.9972      0.587*   1.23e-14*   1.07e-35*     21.948*   1.78e-14*   3.03e-22*     24.644*   1.80e-14*   1.94e-13*
# mst                 77      1.944*   9.20e-07*   3.84e-07*      3.044*   3.17e-06*   5.91e-07*     0.8930*     0.0003*     0.0001*       0.940      1.0000      1.0000      0.852*   6.71e-08*   1.07e-08*       6.000      0.2439      0.1593       7.945      0.2261      0.0910
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# annotation_grow (ref)  77       1.778           -           -       2.471           -           -      0.9042           -           -       0.929           -           -       0.881           -           -       5.532           -           -       7.274           -           -

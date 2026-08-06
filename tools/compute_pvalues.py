"""Compute paired p-values (Wilcoxon signed-rank + paired t-test) for hd95 and
clDice between a reference parameter setting and every other setting present
in the grid-search output directory.

Comparisons are discovered automatically: each combo_*.json under per_combo/
is projected onto the keys of REFERENCE_PARAMS, deduplicated, and the
reference's own projection is removed. Set REFERENCE_PARAMS to the axes you
want to sweep.

Run:
    uv run python tools/compute_pvalues.py
    uv run python tools/compute_pvalues.py --output output/pvalues.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Edit these to change what is compared.
# GRID_DIR must contain a per_combo/ subdirectory produced by staged_grid_search.py.
# REFERENCE_PARAMS' keys define the axes; comparisons are auto-discovered by
# scanning per_combo/ for every unique value combination along those axes.
# ---------------------------------------------------------------------------
# GRID_DIR = Path("output/0510_grid")
# GRID_DIR = Path("output/grid_0510/bg")
GRID_DIR = Path("output/grid_0510/clahe_grid")
# GRID_DIR = Path("output/grid_fk/clahe_grid")
# GRID_DIR = Path("output/grid_0510/clahe_clip")
# GRID_DIR = Path("output/grid_0510/clahe_clip_half")
# GRID_DIR = Path("output/grid_0510/sato_s")
# GRID_DIR = Path("output/grid_0510/sato_e")
# GRID_DIR = Path("output/grid_0510/stub")

# REFERENCE_PARAMS: dict = {"bg_kernel_size": 5}
REFERENCE_PARAMS: dict = {"clahe_grid": [768, 768]}
# REFERENCE_PARAMS: dict = {"clahe_clip": 40.0}
# REFERENCE_PARAMS: dict = {"sato_sigmas_start": 2,
#       "sato_sigmas_stop": 6,}
# REFERENCE_PARAMS: dict = {"sato_sigmas_start": 1}
# REFERENCE_PARAMS: dict = { "sato_sigmas_stop": 4,}
# REFERENCE_PARAMS: dict = {"prune_threshold": 20}
# REFERENCE_PARAMS: dict = { "stub_length_threshold": 3,}
METRICS: list[str] = ["hd95", "cldice", "avg_hd"]
# ---------------------------------------------------------------------------

# Per-metric: (json key in samples[*], lower_is_better)
METRIC_SPECS: dict[str, tuple[str, bool]] = {
    "hd95": ("hd95", True),
    "cldice": ("cldice", False),
    "avg_hd": ("avg_hd", True),
}


def _hashable(v):
    """Make nested lists hashable so projections can go into a set."""
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    return v


def discover_comparisons(
    grid_dir: Path, axis_keys: list[str], exclude: dict | None = None
) -> list[dict]:
    """
    Scan grid_dir/per_combo/combo_*.json and return every unique parameter
    projection onto ``axis_keys``. The projection equal to ``exclude`` (if
    given) is dropped from the result. Combos missing any axis key are
    skipped.

    Returns:
        Sorted list of dicts, each with exactly the keys in ``axis_keys``.
    """
    per_combo_dir = grid_dir / "per_combo"
    if not per_combo_dir.is_dir():
        raise FileNotFoundError(f"per_combo/ not found under {grid_dir}")

    exclude_key = (
        tuple(_hashable(exclude[k]) for k in axis_keys) if exclude is not None else None
    )

    seen: dict[tuple, dict] = {}
    duplicates = 0
    for combo_file in sorted(per_combo_dir.glob("combo_*.json")):
        with combo_file.open() as f:
            data = json.load(f)
        params = data.get("params", {})
        if any(k not in params for k in axis_keys):
            continue
        proj = {k: params[k] for k in axis_keys}
        key = tuple(_hashable(proj[k]) for k in axis_keys)
        if key == exclude_key:
            continue
        if key in seen:
            duplicates += 1
            continue
        seen[key] = proj

    if duplicates:
        print(
            f"  note: {duplicates} combo(s) shared an axis projection with another "
            f"(grid varies along axes outside {axis_keys}); first match kept.",
            file=sys.stderr,
        )

    return [seen[k] for k in sorted(seen)]


def load_combo_samples(
    grid_dir: Path, match_params: dict
) -> tuple[dict[str, dict], str]:
    """
    Scan grid_dir/per_combo/combo_*.json and return samples from the first
    combo whose params contain all key-value pairs in match_params.

    Returns:
        (sample_dict, display_name)
        sample_dict: {sample_id: {hd95, cldice, tprec, tsens, ...}}
    """
    per_combo_dir = grid_dir / "per_combo"
    if not per_combo_dir.is_dir():
        raise FileNotFoundError(f"per_combo/ not found under {grid_dir}")

    for combo_file in sorted(per_combo_dir.glob("combo_*.json")):
        with combo_file.open() as f:
            data = json.load(f)
        params = data.get("params", {})
        if all(params.get(k) == v for k, v in match_params.items()):
            samples = {
                s["sample_id"]: s
                for s in data.get("samples", [])
                if s.get("status") == "success"
            }
            name = ", ".join(f"{k}={v}" for k, v in match_params.items())
            return samples, name

    raise FileNotFoundError(f"No combo matching {match_params} in {per_combo_dir}")


def paired_values(
    ref_samples: dict[str, dict],
    cmp_samples: dict[str, dict],
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    common_ids = sorted(set(ref_samples) & set(cmp_samples))
    ref_vals: list[float] = []
    cmp_vals: list[float] = []
    for sid in common_ids:
        rv = ref_samples[sid].get(metric_key)
        cv = cmp_samples[sid].get(metric_key)
        if rv is None or cv is None:
            continue
        if not (np.isfinite(rv) and np.isfinite(cv)):
            continue
        ref_vals.append(float(rv))
        cmp_vals.append(float(cv))
    return np.asarray(ref_vals), np.asarray(cmp_vals)


def run_tests(
    ref: np.ndarray,
    cmp: np.ndarray,
    lower_is_better: bool,
) -> dict[str, float]:
    """Paired-test stats. Two-sided p, plus one-sided p for "ref is better"."""
    n = len(ref)
    diff = ref - cmp  # > 0 means reference is larger than comparison

    out: dict[str, float] = {
        "n": n,
        "ref_mean": float(np.mean(ref)) if n else float("nan"),
        "cmp_mean": float(np.mean(cmp)) if n else float("nan"),
        "cmp_std": float(np.std(cmp, ddof=1)) if n > 1 else float("nan"),
        "ref_median": float(np.median(ref)) if n else float("nan"),
        "cmp_median": float(np.median(cmp)) if n else float("nan"),
        "mean_diff": float(np.mean(diff)) if n else float("nan"),
        "wins_ref": int(np.sum(diff < 0) if lower_is_better else np.sum(diff > 0)),
        "wins_cmp": int(np.sum(diff > 0) if lower_is_better else np.sum(diff < 0)),
        "ties": int(np.sum(diff == 0)),
    }

    nan_keys = (
        "wilcoxon_stat",
        "wilcoxon_p",
        "wilcoxon_p_ref_better",
        "ttest_stat",
        "ttest_p",
        "ttest_p_ref_better",
    )
    if n < 2 or np.all(diff == 0):
        for k in nan_keys:
            out[k] = float("nan")
        return out

    alt_better = "less" if lower_is_better else "greater"

    try:
        w_two = stats.wilcoxon(ref, cmp, zero_method="wilcox", alternative="two-sided")
        out["wilcoxon_stat"] = float(w_two.statistic)
        out["wilcoxon_p"] = float(w_two.pvalue)
        w_one = stats.wilcoxon(ref, cmp, zero_method="wilcox", alternative=alt_better)
        out["wilcoxon_p_ref_better"] = float(w_one.pvalue)
    except ValueError:
        out["wilcoxon_stat"] = float("nan")
        out["wilcoxon_p"] = float("nan")
        out["wilcoxon_p_ref_better"] = float("nan")

    t_two = stats.ttest_rel(ref, cmp)
    out["ttest_stat"] = float(t_two.statistic)
    out["ttest_p"] = float(t_two.pvalue)
    t_one = stats.ttest_rel(ref, cmp, alternative=alt_better)
    out["ttest_p_ref_better"] = float(t_one.pvalue)

    return out


def display_name(path: Path) -> str:
    p = path
    if p.name == "results.json":
        return p.parent.name
    if p.is_dir():
        return p.name
    return p.stem


def format_p(p: float) -> str:
    if not np.isfinite(p):
        return "nan"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def print_human(rows: list[dict], metrics: list[str], ref_name: str) -> None:
    by_metric: dict[str, list[dict]] = {m: [] for m in metrics}
    for r in rows:
        by_metric[r["metric"]].append(r)

    cmp_width = max((len(r["comparison"]) for r in rows), default=20)
    cmp_width = max(cmp_width, 12)

    def _sort_key(r: dict, lower_is_better: bool) -> float:
        v = r.get("cmp_mean")
        if v is None or not np.isfinite(v):
            return float("inf")  # invalid → last
        # Best first: ascending for lower-better, descending for higher-better.
        return float(v) if lower_is_better else -float(v)

    for m in metrics:
        _, lower_better = METRIC_SPECS[m]
        better = "lower" if lower_better else "higher"
        print()
        print(f"=== {m}  ({better} is better; reference = {ref_name}) ===")
        header = (
            f"{'comparison':<{cmp_width}} {'n':>3} "
            f"{'ref_mean':>9} {'cmp_mean':>9} {'Δmean':>9} "
            f"{'wilcox_p':>10} {'wilcox_p<':>11}"
        )
        print(header)
        print("-" * len(header))
        for r in sorted(by_metric[m], key=lambda r: _sort_key(r, lower_better)):
            print(
                f"{r['comparison']:<{cmp_width}} {r['n']:>3} "
                f"{r['ref_mean']:>9.4f} {r['cmp_mean']:>9.4f} "
                f"{r['mean_diff']:>+9.4f} "
                f"{format_p(r['wilcoxon_p']):>10} "
                f"{format_p(r['wilcoxon_p_ref_better']):>11}"
            )
        print(
            "  wilcox_p<  : one-sided p-value for "
            "'reference is better than comparison'."
        )


CSV_METRIC_ORDER = ["hd95", "avg_hd", "cldice"]


def write_csv(wide_rows: list[dict], setting_col: str, out_path: Path) -> None:
    fieldnames = [setting_col]
    for m in CSV_METRIC_ORDER:
        fieldnames += [m, f"{m}_std", f"{m}_p_cmp_worse_than_ref"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in wide_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path (default: <GRID_DIR>/pvalues.csv).",
    )
    args = parser.parse_args()

    try:
        ref_samples, ref_name = load_combo_samples(GRID_DIR, REFERENCE_PARAMS)
    except FileNotFoundError as exc:
        print(f"Reference not found: {exc}", file=sys.stderr)
        return 1
    print(f"Reference: {ref_name}  (n_success={len(ref_samples)})")

    axis_keys = list(REFERENCE_PARAMS.keys())
    try:
        comparisons = discover_comparisons(
            GRID_DIR, axis_keys, exclude=REFERENCE_PARAMS
        )
    except FileNotFoundError as exc:
        print(f"Could not discover comparisons: {exc}", file=sys.stderr)
        return 1
    print(f"Discovered {len(comparisons)} comparison(s) over axes: {axis_keys}")

    # Axis values are numeric (kernel sizes, thresholds, ...) — sort ascending.
    comparisons.sort(key=lambda p: tuple(p[k] for k in axis_keys))

    setting_col = axis_keys[0] if len(axis_keys) == 1 else "setting"

    def setting_value(params: dict) -> object:
        return params[axis_keys[0]] if len(axis_keys) == 1 else str(params)

    # Reference row: mean/std of its own samples, no p-value (nothing to compare against).
    ref_row: dict = {setting_col: setting_value(REFERENCE_PARAMS)}
    for metric in METRICS:
        metric_key, _ = METRIC_SPECS[metric]
        vals = [
            float(s[metric_key]) for s in ref_samples.values()
            if s.get(metric_key) is not None and np.isfinite(s[metric_key])
        ]
        ref_row[metric] = float(np.mean(vals)) if vals else float("nan")
        ref_row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
    wide_rows: list[dict] = [ref_row]

    rows: list[dict] = []
    for cmp_params in comparisons:
        try:
            cmp_samples, cmp_name = load_combo_samples(GRID_DIR, cmp_params)
        except FileNotFoundError as exc:
            print(f"  skip: {exc}", file=sys.stderr)
            continue

        wide_row: dict = {setting_col: setting_value(cmp_params)}
        for metric in METRICS:
            metric_key, lower_is_better = METRIC_SPECS[metric]
            ref_vals, cmp_vals = paired_values(ref_samples, cmp_samples, metric_key)
            stats_out = run_tests(ref_vals, cmp_vals, lower_is_better)
            rows.append({
                "metric": metric,
                "reference": ref_name,
                "comparison": cmp_name,
                **stats_out,
            })
            wide_row[metric] = stats_out["cmp_mean"]
            wide_row[f"{metric}_std"] = stats_out["cmp_std"]
            wide_row[f"{metric}_p_cmp_worse_than_ref"] = stats_out["wilcoxon_p_ref_better"]
        wide_rows.append(wide_row)

    if not rows:
        print("No comparison results produced.", file=sys.stderr)
        return 1

    print_human(rows, METRICS, ref_name)

    wide_rows.sort(key=lambda r: r[setting_col])

    out_path = args.output if args.output is not None else GRID_DIR / "pvalues.csv"
    write_csv(wide_rows, setting_col, out_path)
    print(f"\nCSV written to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

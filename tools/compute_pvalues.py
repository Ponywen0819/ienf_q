"""Compute paired p-values (Wilcoxon signed-rank + paired t-test) for hd95 and
clDice between a reference parameter setting and a hard-coded list of
comparison settings.

Edit `REFERENCE`, `COMPARISONS`, and `METRICS` below to change what is tested.

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
# Paths are relative to the project root and should each contain a
# `results.json` produced by the evaluation pipeline.
# ---------------------------------------------------------------------------
GRID_DIR = Path("output/grid")

REFERENCE = GRID_DIR / "bg_5_clahe_768_20_sato_3_8_20"

COMPARISONS: list[Path] = [
    # GRID_DIR / "bg_0",
    # GRID_DIR / "bg_3",
    # GRID_DIR / "bg_6",
    # GRID_DIR / "bg_8",
    # GRID_DIR / "bg_16",
    GRID_DIR / "bg_0_clahe_768_20_sato_3_8_20",
    # GRID_DIR / "bg_3_clahe_768_20_sato_3_8_20",
    GRID_DIR / "bg_5_clahe_768_20_sato_3_8_20",
    GRID_DIR / "bg_7_clahe_768_20_sato_3_8_20",
    GRID_DIR / "bg_11_clahe_768_20_sato_3_8_20",
    GRID_DIR / "bg_15_clahe_768_20_sato_3_8_20",
    GRID_DIR / "bg_21_clahe_768_20_sato_3_8_20",
    GRID_DIR / "bg_25_clahe_768_20_sato_3_8_20",
    GRID_DIR / "bg_31_clahe_768_20_sato_3_8_20",
    GRID_DIR / "bg_41_clahe_768_20_sato_3_8_20",
    # GRID_DIR / "bg_31_clahe_768_20_sato_3_7",
    # GRID_DIR / "bg_31_clahe_768_20_sato_3_10",
    # GRID_DIR / "bg_31_clahe_768_20_sato_3_8_10",
    # GRID_DIR / "bg_31_clahe_768_20_sato_3_8_30",
    # GRID_DIR / "bg_31_clahe_768_20_sato_3_8_40",
    # GRID_DIR / "bg_31_clahe_768_20_sato_3_8_50",
]

METRICS: list[str] = ["hd95", "cldice"]
# ---------------------------------------------------------------------------

# Per-metric: (json key in samples[*], lower_is_better)
METRIC_SPECS: dict[str, tuple[str, bool]] = {
    "hd95": ("hd95", True),
    "cldice": ("cldice", False),
    "hausdorff": ("hausdorff_distance", True),
}


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

    for m in metrics:
        spec = METRIC_SPECS[m]
        better = "lower" if spec[1] else "higher"
        print()
        print(f"=== {m}  ({better} is better; reference = {ref_name}) ===")
        header = (
            f"{'comparison':<{cmp_width}} {'n':>3} "
            f"{'ref_mean':>9} {'cmp_mean':>9} {'Δmean':>9} "
            f"{'wilcox_p':>10} {'wilcox_p<':>11} "
            f"{'ttest_p':>10} {'ttest_p<':>10}"
        )
        print(header)
        print("-" * len(header))
        for r in by_metric[m]:
            print(
                f"{r['comparison']:<{cmp_width}} {r['n']:>3} "
                f"{r['ref_mean']:>9.4f} {r['cmp_mean']:>9.4f} "
                f"{r['mean_diff']:>+9.4f} "
                f"{format_p(r['wilcoxon_p']):>10} "
                f"{format_p(r['wilcoxon_p_ref_better']):>11} "
                f"{format_p(r['ttest_p']):>10} "
                f"{format_p(r['ttest_p_ref_better']):>10}"
            )
        print(
            "  wilcox_p<  / ttest_p<  : one-sided p-value for "
            "'reference is better than comparison'."
        )


def write_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = [
        "metric",
        "reference",
        "comparison",
        "n",
        "ref_mean",
        "cmp_mean",
        "ref_median",
        "cmp_median",
        "mean_diff",
        "wins_ref",
        "wins_cmp",
        "ties",
        "wilcoxon_stat",
        "wilcoxon_p",
        "wilcoxon_p_ref_better",
        "ttest_stat",
        "ttest_p",
        "ttest_p_ref_better",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    args = parser.parse_args()

    ref_json = resolve_results_json(REFERENCE)
    ref_samples = load_samples(ref_json)
    ref_name = display_name(ref_json)
    print(f"Reference: {ref_name}  (n_success={len(ref_samples)})")

    rows: list[dict] = []
    for cmp_path in COMPARISONS:
        try:
            cmp_json = resolve_results_json(cmp_path)
        except FileNotFoundError as exc:
            print(f"  skip: {exc}", file=sys.stderr)
            continue
        cmp_samples = load_samples(cmp_json)
        cmp_name = display_name(cmp_json)

        for metric in METRICS:
            metric_key, lower_is_better = METRIC_SPECS[metric]
            ref_vals, cmp_vals = paired_values(ref_samples, cmp_samples, metric_key)
            row = {
                "metric": metric,
                "reference": ref_name,
                "comparison": cmp_name,
                **run_tests(ref_vals, cmp_vals, lower_is_better),
            }
            rows.append(row)

    if not rows:
        print("No comparison results produced.", file=sys.stderr)
        return 1

    print_human(rows, METRICS, ref_name)

    if args.output is not None:
        write_csv(rows, args.output)
        print(f"\nCSV written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

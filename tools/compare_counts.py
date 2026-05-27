"""Compare two count JSON files by MAE and correlation coefficient.

Each JSON file is expected to map sample_id -> integer count, e.g.:
    {"S222-2_a": 11, "S222-2_b": 12, ...}

Usage:
    uv run python tools/compare_counts.py data_0510/count.json data_0510/count_orig.json
    uv run python tools/compare_counts.py file1.json file2.json --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


def load_counts(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): float(v) for k, v in data.items()}


def compare(
    counts_a: dict[str, float],
    counts_b: dict[str, float],
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, float]]:
    keys_a, keys_b = set(counts_a), set(counts_b)
    common = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    if not common:
        raise ValueError("No overlapping sample_ids between the two files.")

    a = np.array([counts_a[k] for k in common], dtype=float)
    b = np.array([counts_b[k] for k in common], dtype=float)

    diff = a - b
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    mean_bias = float(np.mean(diff))  # a - b

    if np.std(a) == 0 or np.std(b) == 0:
        pearson_r, pearson_p = float("nan"), float("nan")
    else:
        pearson_r, pearson_p = pearsonr(a, b)

    spearman_r, spearman_p = spearmanr(a, b)

    metrics = {
        "n": len(common),
        "mae": mae,
        "rmse": rmse,
        "mean_bias_a_minus_b": mean_bias,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "only_in_a": only_a,
        "only_in_b": only_b,
    }
    return common, a, b, metrics


def write_csv(path: Path, keys: list[str], a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", label_a, label_b, "abs_diff", "diff_a_minus_b"])
        for k, va, vb in zip(keys, a, b):
            writer.writerow([k, va, vb, abs(va - vb), va - vb])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file_a", type=Path, help="First count JSON (treated as A)")
    parser.add_argument("file_b", type=Path, help="Second count JSON (treated as B)")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV path for per-sample diffs")
    args = parser.parse_args()

    counts_a = load_counts(args.file_a)
    counts_b = load_counts(args.file_b)

    common, a, b, metrics = compare(counts_a, counts_b)

    label_a = args.file_a.stem
    label_b = args.file_b.stem

    print(f"Comparing: {args.file_a}  vs  {args.file_b}")
    print(f"  A = {label_a}   (n={len(counts_a)})")
    print(f"  B = {label_b}   (n={len(counts_b)})")
    print(f"  Common sample_ids: {metrics['n']}")
    if metrics["only_in_a"]:
        print(f"  Only in A ({len(metrics['only_in_a'])}): {metrics['only_in_a']}")
    if metrics["only_in_b"]:
        print(f"  Only in B ({len(metrics['only_in_b'])}): {metrics['only_in_b']}")

    print("\n=== Summary ===")
    print(f"  MAE                 : {metrics['mae']:.4f}")
    print(f"  RMSE                : {metrics['rmse']:.4f}")
    print(f"  Mean bias (A - B)   : {metrics['mean_bias_a_minus_b']:+.4f}")
    print(f"  Pearson  r          : {metrics['pearson_r']:.4f}   (p={metrics['pearson_p']:.3g})")
    print(f"  Spearman r          : {metrics['spearman_r']:.4f}   (p={metrics['spearman_p']:.3g})")

    if args.output is not None:
        write_csv(args.output, common, a, b, label_a, label_b)
        print(f"\nPer-sample diffs written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

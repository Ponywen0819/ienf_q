"""Automatically identify the best parameter combination from staged grid search results.

Reads per_combo/ JSON files, ranks all combinations by hd95_mean, then runs
paired Wilcoxon tests to find which combinations are NOT significantly worse
than the best. For multi-parameter grids, also reports the per-parameter
marginal analysis.

Run:
    uv run python tools/find_best_params.py --grid-dir output/0510_grid
    uv run python tools/find_best_params.py --grid-dir output/0510_grid --metric cldice --top 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


METRIC_LOWER_IS_BETTER = {
    "hd95": True,
    "cldice": False,
    "tprec": False,
    "tsens": False,
}


# ── Data loading ─────────────────────────────────────────────────────────────


def load_combos(grid_dir: Path) -> list[dict]:
    """
    Load all per_combo/combo_*.json files.

    Returns list of dicts, each with:
      combo_index, params, samples (filtered to status==success, metric not None)
    """
    per_combo_dir = grid_dir / "per_combo"
    if not per_combo_dir.is_dir():
        raise FileNotFoundError(f"per_combo/ not found under {grid_dir}")

    combos = []
    for f in sorted(per_combo_dir.glob("combo_*.json")):
        with f.open(encoding="utf-8") as fh:
            combos.append(json.load(fh))
    return combos


def get_paired_values(
    ref_samples: dict[str, float],
    cmp_samples: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    common = sorted(set(ref_samples) & set(cmp_samples))
    rv, cv = [], []
    for sid in common:
        r, c = ref_samples[sid], cmp_samples[sid]
        if r is not None and c is not None and np.isfinite(r) and np.isfinite(c):
            rv.append(r)
            cv.append(c)
    return np.array(rv), np.array(cv)


def combo_metric_map(combo: dict, metric: str) -> dict[str, float]:
    """Return {sample_id: metric_value} for successful samples."""
    return {
        s["sample_id"]: s[metric]
        for s in combo.get("samples", [])
        if s.get("status") == "success" and s.get(metric) is not None
    }


# ── Statistics ────────────────────────────────────────────────────────────────


def wilcoxon_one_sided(
    ref: np.ndarray,
    cmp: np.ndarray,
    lower_is_better: bool,
) -> float:
    """
    One-sided p-value for "ref is better than cmp".
    Returns NaN if test cannot be run.
    """
    diff = ref - cmp
    if len(diff) < 2 or np.all(diff == 0):
        return float("nan")
    alternative = "less" if lower_is_better else "greater"
    try:
        result = stats.wilcoxon(ref, cmp, zero_method="wilcox", alternative=alternative)
        return float(result.pvalue)
    except ValueError:
        return float("nan")


def wilcoxon_two_sided(ref: np.ndarray, cmp: np.ndarray) -> float:
    diff = ref - cmp
    if len(diff) < 2 or np.all(diff == 0):
        return float("nan")
    try:
        result = stats.wilcoxon(ref, cmp, zero_method="wilcox", alternative="two-sided")
        return float(result.pvalue)
    except ValueError:
        return float("nan")


# ── Formatting ────────────────────────────────────────────────────────────────


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "  n/a  "
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def fmt_val(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "   n/a"
    return f"{v:8.4f}"


def params_label(params: dict, varied_keys: list[str]) -> str:
    return ", ".join(f"{k}={params[k]}" for k in varied_keys)


# ── Analysis ──────────────────────────────────────────────────────────────────


def rank_combos(
    combos: list[dict],
    metric: str,
    lower_is_better: bool,
) -> list[tuple[float, int, dict]]:
    """Return [(mean_metric, combo_index, combo)] sorted best-first."""
    rows = []
    for c in combos:
        vals = [v for v in combo_metric_map(c, metric).values() if np.isfinite(v)]
        if not vals:
            continue
        rows.append((float(np.mean(vals)), c["combo_index"], c))
    rows.sort(key=lambda x: x[0], reverse=not lower_is_better)
    return rows


def pairwise_vs_champion(
    ranked: list[tuple[float, int, dict]],
    metric: str,
    lower_is_better: bool,
    alpha: float,
) -> list[dict]:
    """
    For each combo, compute paired Wilcoxon (two-sided + one-sided "champion is better").
    Returns list of result dicts, ordered as ranked.
    """
    if not ranked:
        return []

    champ_map = combo_metric_map(ranked[0][2], metric)
    rows = []
    for mean_val, idx, combo in ranked:
        cmp_map = combo_metric_map(combo, metric)
        ref_arr, cmp_arr = get_paired_values(champ_map, cmp_map)
        n = len(ref_arr)

        p_two = wilcoxon_two_sided(ref_arr, cmp_arr)
        p_one = wilcoxon_one_sided(ref_arr, cmp_arr, lower_is_better)

        rows.append(
            {
                "combo_index": idx,
                "params": combo["params"],
                "n": n,
                "mean": mean_val,
                "median": float(np.median(cmp_arr)) if n else float("nan"),
                "p_two": p_two,
                "p_champ_better": p_one,
                "not_sig_worse": not np.isfinite(p_one) or p_one >= alpha,
            }
        )
    return rows


def per_param_analysis(
    combos: list[dict],
    param_grid: dict,
    metric: str,
    lower_is_better: bool,
    alpha: float,
) -> dict[str, list[dict]]:
    """
    For each parameter in param_grid, pool all combos sharing the same value of
    that parameter and compare values using paired Wilcoxon on the pooled samples.

    Returns {param_name: [row per value, sorted best-first]}.
    """
    results: dict[str, list[dict]] = {}

    for param_name, values in param_grid.items():
        # Group combos by their value for this parameter
        by_value: dict = {}
        for combo in combos:
            v = combo["params"].get(param_name)
            # Normalise lists to tuples for hashing
            key = tuple(v) if isinstance(v, list) else v
            by_value.setdefault(key, []).append(combo)

        # Build pooled {sample_id: list[metric_val]} per value — one observation per combo
        # We want paired tests so we need per-sample averages across combos with that value
        pooled: dict = {}  # value → {sample_id: mean_metric}
        for val_key, val_combos in by_value.items():
            sample_vals: dict[str, list[float]] = {}
            for combo in val_combos:
                for sid, mv in combo_metric_map(combo, metric).items():
                    if np.isfinite(mv):
                        sample_vals.setdefault(sid, []).append(mv)
            pooled[val_key] = {
                sid: float(np.mean(vs)) for sid, vs in sample_vals.items()
            }

        # Rank by mean of the pooled values
        ranked_vals = sorted(
            pooled.items(),
            key=lambda kv: np.mean(list(kv[1].values())) if kv[1] else float("inf"),
            reverse=not lower_is_better,
        )
        if not ranked_vals:
            continue

        best_val_key, best_map = ranked_vals[0]

        rows = []
        for val_key, val_map in ranked_vals:
            ref_arr, cmp_arr = get_paired_values(best_map, val_map)
            n = len(ref_arr)
            mean_val = float(np.mean(list(val_map.values()))) if val_map else float("nan")
            p_two = wilcoxon_two_sided(ref_arr, cmp_arr)
            p_one = wilcoxon_one_sided(ref_arr, cmp_arr, lower_is_better)
            rows.append(
                {
                    "value": val_key,
                    "n": n,
                    "mean": mean_val,
                    "p_two": p_two,
                    "p_best_better": p_one,
                    "not_sig_worse": not np.isfinite(p_one) or p_one >= alpha,
                    "is_best": val_key == best_val_key,
                }
            )
        results[param_name] = rows

    return results


# ── Printing ──────────────────────────────────────────────────────────────────


def print_overall_ranking(
    pairwise_rows: list[dict],
    varied_keys: list[str],
    metric: str,
    lower_is_better: bool,
    alpha: float,
    top: int,
) -> None:
    better = "lower" if lower_is_better else "higher"
    print(f"\n{'='*72}")
    print(f"Overall ranking by {metric} ({better} is better)  [α={alpha}]")
    print(f"{'='*72}")
    label_w = max((len(params_label(r["params"], varied_keys)) for r in pairwise_rows), default=20)
    label_w = max(label_w, 12)
    header = (
        f"{'params':<{label_w}}  {'n':>4}  {'mean':>8}  {'median':>8}"
        f"  {'p(2-sided)':>10}  {'p(champ>)':>10}  sig_worse"
    )
    print(header)
    print("-" * len(header))

    for i, row in enumerate(pairwise_rows[:top]):
        label = params_label(row["params"], varied_keys)
        if i == 0:
            sig_col = "← best"
        elif row["not_sig_worse"]:
            sig_col = "-"
        else:
            sig_col = "YES *"
        print(
            f"{label:<{label_w}}  {row['n']:>4}  {fmt_val(row['mean'])}  {fmt_val(row['median'])}"
            f"  {fmt_p(row['p_two']):>10}  {fmt_p(row['p_champ_better']):>10}  {sig_col}"
        )

    not_worse = [r for r in pairwise_rows if r["not_sig_worse"]]
    print(
        f"\n{len(not_worse)}/{len(pairwise_rows)} combos are NOT significantly worse "
        f"than the best (p_champ_better >= {alpha})."
    )
    if not_worse:
        print("  Equivalent to best:")
        for r in not_worse:
            print(f"    {params_label(r['params'], varied_keys)}")


def print_per_param(
    param_analysis: dict[str, list[dict]],
    metric: str,
    lower_is_better: bool,
    alpha: float,
) -> None:
    better = "lower" if lower_is_better else "higher"
    print(f"\n{'='*72}")
    print(f"Per-parameter marginal analysis ({metric}, {better} is better)")
    print(f"{'='*72}")

    for param_name, rows in param_analysis.items():
        print(f"\n  {param_name}:")
        val_w = max((len(str(r["value"])) for r in rows), default=6)
        val_w = max(val_w, 6)
        print(
            f"    {'value':<{val_w}}  {'n':>4}  {'mean':>8}  {'p(best>)':>10}  note"
        )
        print(f"    {'-'*(val_w + 40)}")
        for row in rows:
            note = ""
            if row["is_best"]:
                note = "← best"
            elif not row["not_sig_worse"]:
                note = "significantly worse *"
            print(
                f"    {str(row['value']):<{val_w}}  {row['n']:>4}  "
                f"{fmt_val(row['mean'])}  {fmt_p(row['p_best_better']):>10}  {note}"
            )


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-dir", type=Path, required=True,
        help="Grid search output directory (contains per_combo/ and grid_search_results.json).",
    )
    parser.add_argument(
        "--metric", choices=list(METRIC_LOWER_IS_BETTER), default="hd95",
        help="Metric to optimise (default: hd95).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level for Wilcoxon tests (default: 0.05).",
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="Show only the top-N combinations in the ranking table (default: all).",
    )
    args = parser.parse_args()

    combos = load_combos(args.grid_dir)
    if not combos:
        print("No combo files found.", file=sys.stderr)
        return 1

    # Load param_grid to know which parameters were actually varied
    results_json = args.grid_dir / "grid_search_results.json"
    param_grid: dict = {}
    if results_json.exists():
        with results_json.open(encoding="utf-8") as f:
            doc = json.load(f)
        param_grid = doc.get("param_grid", {})

    # Determine which keys actually vary across combos
    all_keys = list(combos[0]["params"].keys()) if combos else []
    varied_keys = (
        [k for k in param_grid]
        if param_grid
        else [
            k for k in all_keys
            if len({str(c["params"].get(k)) for c in combos}) > 1
        ]
    )
    if not varied_keys:
        varied_keys = all_keys

    lower_is_better = METRIC_LOWER_IS_BETTER[args.metric]
    top = args.top or len(combos)

    ranked = rank_combos(combos, args.metric, lower_is_better)
    if not ranked:
        print(f"No valid {args.metric} values found in combo files.", file=sys.stderr)
        return 1

    print(f"\nGrid dir : {args.grid_dir}")
    print(f"Metric   : {args.metric}  ({'lower' if lower_is_better else 'higher'} is better)")
    print(f"Combos   : {len(ranked)}")
    print(f"Alpha    : {args.alpha}")

    pairwise_rows = pairwise_vs_champion(ranked, args.metric, lower_is_better, args.alpha)
    print_overall_ranking(pairwise_rows, varied_keys, args.metric, lower_is_better, args.alpha, top)

    if len(varied_keys) > 1 or (len(varied_keys) == 1 and param_grid):
        param_analysis = per_param_analysis(
            combos, param_grid, args.metric, lower_is_better, args.alpha
        )
        if param_analysis:
            print_per_param(param_analysis, args.metric, lower_is_better, args.alpha)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

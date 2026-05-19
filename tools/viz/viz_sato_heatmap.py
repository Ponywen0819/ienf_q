"""
Render hd95 and clDice heatmaps over (sato_sigmas_start, sato_sigmas_stop).

Reads combo_*.json under each GRID_DIRS[i]/per_combo/ (produced by
staged_grid_search.py), merges them into a single (start × stop) grid (first
dir wins on conflicts), computes the per-cell metric mean across successful
samples, and plots two heatmaps — one per metric — with cell values annotated.

Each non-reference cell is also paired-tested against REFERENCE_PARAMS using
one-sided Wilcoxon signed-rank ("reference is better than comparison"); cells
with p < 0.05 get a trailing '*', otherwise the p-value is shown below.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy import stats

GRID_DIRS: list[Path] = [
    Path("/home/pony/projects/ienf_q/output/grid_0510/sato_n"),
    # Path("/home/pony/projects/ienf_q/output/grid_0510/sato_lower"),
    # Path("/home/pony/projects/ienf_q/output/grid_0510/sato_lower_1"),
    # Path("/home/pony/projects/ienf_q/output/grid_0510/sato_lower_4_5"),
    # Path("/home/pony/projects/ienf_q/output/grid_0510/sato_lower_2_3"),
]
OUT_DIR = Path(__file__).parent

X_KEY = "sato_sigmas_start"
Y_KEY = "sato_sigmas_stop"

REFERENCE_PARAMS: dict = {X_KEY: 1, Y_KEY: 4}

# Optional axis cap — drop cells where Y_KEY > Y_MAX. None disables the cap.
Y_MAX: int | None = 6

# {metric_field: lower_is_better}
METRICS = {
    "hd95": True,
    "cldice": False,
}


def load_grid() -> dict:
    """Return {(x, y): {metric: {sample_id: value}}} merged across GRID_DIRS.

    On (x, y) collision the first dir in GRID_DIRS wins and a note is printed
    so the conflict is visible.
    """
    out: dict = {}
    source: dict[tuple, Path] = {}
    for grid_dir in GRID_DIRS:
        per_combo_dir = grid_dir / "per_combo"
        if not per_combo_dir.is_dir():
            print(f"  warning: {per_combo_dir} not found, skipping")
            continue
        for combo_file in sorted(per_combo_dir.glob("combo_*.json")):
            with combo_file.open() as f:
                data = json.load(f)
            params = data.get("params", {})
            if X_KEY not in params or Y_KEY not in params:
                continue
            x, y = params[X_KEY], params[Y_KEY]
            if (x, y) in out:
                print(
                    f"  note: ({X_KEY}={x}, {Y_KEY}={y}) seen in both "
                    f"{source[(x, y)].name} and {grid_dir.name}; "
                    f"keeping {source[(x, y)].name}"
                )
                continue
            per_metric: dict[str, dict[str, float]] = {m: {} for m in METRICS}
            for s in data.get("samples", []):
                if s.get("status") != "success":
                    continue
                sid = s["sample_id"]
                for m in METRICS:
                    v = s.get(m)
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        per_metric[m][sid] = float(v)
            out[(x, y)] = per_metric
            source[(x, y)] = grid_dir
    return out


def wilcoxon_p_ref_better(
    ref_vals: dict[str, float],
    cmp_vals: dict[str, float],
    lower_is_better: bool,
) -> float:
    """One-sided Wilcoxon p for 'reference is better than comparison'."""
    common = sorted(set(ref_vals) & set(cmp_vals))
    if len(common) < 2:
        return float("nan")
    ref = np.array([ref_vals[s] for s in common])
    cmp = np.array([cmp_vals[s] for s in common])
    if np.all(ref == cmp):
        return float("nan")
    alt = "less" if lower_is_better else "greater"
    try:
        return float(
            stats.wilcoxon(ref, cmp, zero_method="wilcox", alternative=alt).pvalue
        )
    except ValueError:
        return float("nan")


def build_matrices(
    grid: dict, metric: str, lower_is_better: bool, ref_xy: tuple
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Return (mean_matrix, p_matrix, xs asc, ys asc)."""
    xs = sorted({x for (x, _), v in grid.items() if v.get(metric)})
    ys = sorted({y for (_, y), v in grid.items() if v.get(metric)})
    if Y_MAX is not None:
        ys = [y for y in ys if y <= Y_MAX]
    mean_mat = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
    p_mat = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
    ref_vals = grid.get(ref_xy, {}).get(metric, {})
    for (x, y), per_metric in grid.items():
        if x not in xs or y not in ys:
            continue
        vals = per_metric.get(metric, {})
        if not vals:
            continue
        iy, ix = ys.index(y), xs.index(x)
        mean_mat[iy, ix] = float(np.mean(list(vals.values())))
        if (x, y) != ref_xy and ref_vals:
            p_mat[iy, ix] = wilcoxon_p_ref_better(ref_vals, vals, lower_is_better)
    return mean_mat, p_mat, xs, ys




def plot_heatmap(
    mean_mat: np.ndarray,
    p_mat: np.ndarray,
    xs: list[int],
    ys: list[int],
    ref_xy: tuple,
    metric: str,
    lower_is_better: bool,
    out_path: Path,
) -> None:
    # Wistia stays in the bright ivory→orange range end-to-end, so black text
    # is readable on every cell. hd95 → Wistia (ivory = low = good);
    # cldice → Wistia_r (ivory = high = good).
    cmap = plt.get_cmap("Wistia" if lower_is_better else "Wistia_r").copy()
    cmap.set_bad(color="lightgray")

    fig, ax = plt.subplots(figsize=(1.6 * len(xs) + 2, 1.6 * len(ys) + 1.5),
                           constrained_layout=True)
    im = ax.imshow(mean_mat, cmap=cmap, origin="lower", aspect="equal")

    finite = np.isfinite(mean_mat)
    for iy in range(mean_mat.shape[0]):
        for ix in range(mean_mat.shape[1]):
            if not finite[iy, ix]:
                continue
            val = float(mean_mat[iy, ix])
            is_ref = (xs[ix], ys[iy]) == ref_xy
            p = float(p_mat[iy, ix])
            if is_ref:
                label = f"{val:.4f}\n(ref)"
            elif np.isfinite(p) and p < 0.05:
                label = f"{val:.4f}*"
            elif np.isfinite(p):
                label = f"{val:.4f}\np={p:.2f}"
            else:
                label = f"{val:.4f}"
            ax.text(
                ix, iy, label,
                ha="center", va="center",
                color="black", fontsize=16, linespacing=1.15,
            )
            if is_ref:
                ax.add_patch(Rectangle(
                    (ix - 0.5, iy - 0.5), 1, 1,
                    fill=False, edgecolor="black", linewidth=2.0,
                ))

    # Y axis shows max σ actually used: sato_sigmas_stop is the exclusive end
    # of range(start, stop), so the largest σ evaluated is stop − 1.
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([str(x) for x in xs], fontsize=16)
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels([str(y - 1) for y in ys], fontsize=16)
    ax.set_xlabel("σ min", fontsize=18)
    ax.set_ylabel(f"σ max", fontsize=18)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(metric, fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}  ({mean_mat.shape[1]}×{mean_mat.shape[0]} cells)")


def main() -> None:
    grid = load_grid()
    if not grid:
        raise SystemExit(
            f"No combos with both {X_KEY} and {Y_KEY} under any of {GRID_DIRS}"
        )

    ref_xy = (REFERENCE_PARAMS[X_KEY], REFERENCE_PARAMS[Y_KEY])
    if ref_xy not in grid:
        print(f"  warning: reference {ref_xy} not present in grid")

    for metric, lower_is_better in METRICS.items():
        mean_mat, p_mat, xs, ys = build_matrices(grid, metric, lower_is_better, ref_xy)
        if mean_mat.size == 0:
            print(f"  skip {metric}: no data")
            continue
        out_path = OUT_DIR / f"viz_sato_heatmap_{metric}.png"
        plot_heatmap(mean_mat, p_mat, xs, ys, ref_xy, metric, lower_is_better, out_path)


if __name__ == "__main__":
    main()

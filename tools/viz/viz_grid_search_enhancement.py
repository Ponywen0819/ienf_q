"""Visualize grid search enhancement results as presentation-ready figures.

Reads grid_search_enhancement_results.json and produces:
  - heatmap.png : Fisher score mean over (clip_limit, tile_size) grid
  - lineplot.png : Fisher score mean ± std vs tile_size, lines per clip_limit
  - ranking.png : Bar chart of all configs ranked by Fisher mean
  - summary.png : Combined multi-panel figure for slides/reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: Path) -> tuple[list[dict], dict]:
    with path.open() as f:
        data = json.load(f)
    return data["results"], data["param_grid"]


def build_grid(
    results: list[dict],
    clip_limits: list[float],
    tile_sizes: list[int],
    metric: str,
) -> np.ndarray:
    grid = np.full((len(clip_limits), len(tile_sizes)), np.nan)
    for r in results:
        i = clip_limits.index(r["clip_limit"])
        j = tile_sizes.index(r["tile_size"])
        grid[i, j] = r[metric]
    return grid


def plot_heatmap(
    grid: np.ndarray,
    clip_limits: list[float],
    tile_sizes: list[int],
    out_path: Path,
    title: str = "Fisher Score Mean across CLAHE Parameters",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")

    ax.set_xticks(range(len(tile_sizes)))
    ax.set_xticklabels(tile_sizes)
    ax.set_yticks(range(len(clip_limits)))
    ax.set_yticklabels([f"{c:g}" for c in clip_limits])
    ax.set_xlabel("Tile Size (pixels)", fontsize=12)
    ax.set_ylabel("Clip Limit", fontsize=12)
    ax.set_title(title, fontsize=13, pad=12)

    best_i, best_j = np.unravel_index(np.nanargmax(grid), grid.shape)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid[i, j]
            color = "white" if value < np.nanmean(grid) else "black"
            weight = "bold" if (i, j) == (best_i, best_j) else "normal"
            ax.text(
                j, i, f"{value:.3f}",
                ha="center", va="center",
                color=color, fontsize=10, fontweight=weight,
            )

    rect = plt.Rectangle(
        (best_j - 0.5, best_i - 0.5), 1, 1,
        fill=False, edgecolor="red", linewidth=2.5,
    )
    ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fisher Score (mean)", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lines(
    results: list[dict],
    clip_limits: list[float],
    tile_sizes: list[int],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.get_cmap("tab10").colors

    for idx, clip in enumerate(clip_limits):
        means = []
        stds = []
        for ts in tile_sizes:
            match = next(
                (r for r in results if r["clip_limit"] == clip and r["tile_size"] == ts),
                None,
            )
            means.append(match["fisher_mean"] if match else np.nan)
            stds.append(match["fisher_std"] if match else np.nan)
        means = np.array(means)
        stds = np.array(stds)
        ax.errorbar(
            tile_sizes, means, yerr=stds,
            label=f"clip_limit = {clip:g}",
            marker="o", markersize=7, linewidth=2,
            capsize=4, color=colors[idx],
        )

    ax.set_xlabel("Tile Size (pixels)", fontsize=12)
    ax.set_ylabel("Fisher Score (mean ± std)", fontsize=12)
    ax.set_title("Fisher Score vs Tile Size", fontsize=13, pad=12)
    ax.set_xscale("log", base=2)
    ax.set_xticks(tile_sizes)
    ax.set_xticklabels(tile_sizes)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=11, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ranking(results: list[dict], out_path: Path, top_k: int | None = None) -> None:
    sorted_results = sorted(results, key=lambda r: r["fisher_mean"], reverse=True)
    if top_k:
        sorted_results = sorted_results[:top_k]

    labels = [
        f"clip={r['clip_limit']:g}, tile={r['tile_size']}"
        for r in sorted_results
    ]
    means = [r["fisher_mean"] for r in sorted_results]
    stds = [r["fisher_std"] for r in sorted_results]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(labels) + 1)))

    norm = plt.Normalize(min(means), max(means))
    colors = plt.get_cmap("viridis")(norm(means))

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, means, xerr=stds, color=colors, edgecolor="black",
            linewidth=0.6, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Fisher Score (mean ± std)", fontsize=12)
    ax.set_title("Configurations Ranked by Fisher Score", fontsize=13, pad=12)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.05, i, f"{m:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_summary(
    grid: np.ndarray,
    results: list[dict],
    clip_limits: list[float],
    tile_sizes: list[int],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.1], wspace=0.3)

    # Left: heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(grid, cmap="viridis", aspect="auto")
    ax1.set_xticks(range(len(tile_sizes)))
    ax1.set_xticklabels(tile_sizes)
    ax1.set_yticks(range(len(clip_limits)))
    ax1.set_yticklabels([f"{c:g}" for c in clip_limits])
    ax1.set_xlabel("Tile Size (pixels)", fontsize=11)
    ax1.set_ylabel("Clip Limit", fontsize=11)
    ax1.set_title("(a) Fisher Score Heatmap", fontsize=12, pad=10)

    best_i, best_j = np.unravel_index(np.nanargmax(grid), grid.shape)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid[i, j]
            color = "white" if value < np.nanmean(grid) else "black"
            weight = "bold" if (i, j) == (best_i, best_j) else "normal"
            ax1.text(j, i, f"{value:.2f}", ha="center", va="center",
                     color=color, fontsize=9, fontweight=weight)
    rect = plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                         fill=False, edgecolor="red", linewidth=2.5)
    ax1.add_patch(rect)
    fig.colorbar(im, ax=ax1, label="Fisher Score (mean)")

    # Right: line plot
    ax2 = fig.add_subplot(gs[0, 1])
    colors = plt.get_cmap("tab10").colors
    for idx, clip in enumerate(clip_limits):
        means = []
        stds = []
        for ts in tile_sizes:
            match = next(
                (r for r in results if r["clip_limit"] == clip and r["tile_size"] == ts),
                None,
            )
            means.append(match["fisher_mean"] if match else np.nan)
            stds.append(match["fisher_std"] if match else np.nan)
        means = np.array(means)
        stds = np.array(stds)
        ax2.errorbar(tile_sizes, means, yerr=stds,
                     label=f"clip_limit = {clip:g}",
                     marker="o", markersize=7, linewidth=2,
                     capsize=4, color=colors[idx])
    ax2.set_xlabel("Tile Size (pixels)", fontsize=11)
    ax2.set_ylabel("Fisher Score (mean ± std)", fontsize=11)
    ax2.set_title("(b) Fisher Score vs Tile Size", fontsize=12, pad=10)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(tile_sizes)
    ax2.set_xticklabels(tile_sizes)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(loc="best", fontsize=10, framealpha=0.9)

    best = max(results, key=lambda r: r["fisher_mean"])
    fig.suptitle(
        f"CLAHE Grid Search — Best: clip_limit={best['clip_limit']:g}, "
        f"tile_size={best['tile_size']} → {best['fisher_mean']:.3f}",
        fontsize=13, y=1.02,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("output/grid/bg_finsherscore/grid_search_enhancement_results.json"),
        help="Path to grid_search_enhancement_results.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("output/grid/bg_finsherscore/figures"),
        help="Directory to write figures into",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, param_grid = load_results(args.input)
    clip_limits = sorted(param_grid["clip_limit"])
    tile_sizes = sorted(param_grid["tile_size"])

    grid = build_grid(results, clip_limits, tile_sizes, "fisher_mean")

    plot_heatmap(grid, clip_limits, tile_sizes, args.output_dir / "heatmap.png")
    plot_lines(results, clip_limits, tile_sizes, args.output_dir / "lineplot.png")
    plot_ranking(results, args.output_dir / "ranking.png")
    plot_summary(grid, results, clip_limits, tile_sizes,
                 args.output_dir / "summary.png")

    best = max(results, key=lambda r: r["fisher_mean"])
    print(f"Wrote figures to {args.output_dir}/")
    print(f"  Best: clip_limit={best['clip_limit']:g}, "
          f"tile_size={best['tile_size']} -> "
          f"fisher_mean={best['fisher_mean']:.4f} "
          f"(std={best['fisher_std']:.4f})")


if __name__ == "__main__":
    main()

"""Synthetic illustration of multi_source_dijkstra diffusion.

Renders a static grayscale "background" (the cost map / underlying medium
the algorithm runs on), highlights the source region in the center, and
overlays three iso-cost contours to illustrate the propagation front at
three different cost levels.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from neural_reconstruction.algorithms.annotation_grow.dijkstra import (
    multi_source_dijkstra,
)


OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

H, W = 200, 200
RNG = np.random.default_rng(7)


def make_irregular_source(shape, center, base_radius=18):
    """Return a binary mask shaped like an irregular blob around `center`."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    cy, cx = center

    n_modes = 6
    amps = RNG.uniform(0.25, 0.65, size=n_modes)
    phases = RNG.uniform(0, 2 * np.pi, size=n_modes)
    theta = np.arctan2(yy - cy, xx - cx)
    radial = np.zeros_like(theta)
    for k, (a, p) in enumerate(zip(amps, phases), start=1):
        radial += a * np.cos(k * theta + p)
    radius = base_radius * (1.0 + 0.45 * radial)

    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return dist <= radius


def build_background_and_cost():
    """Return (background, cumulative_cost, source_mask).

    Background: two irregular blobs at maximum brightness, blurred outward
    so they fade into a black surround.

    Cumulative cost: integrated from the blob *centers* (point sources)
    through a smooth traversal-cost field; iso-contours of this field are
    the diffusion fronts.
    """
    seeds = [(H // 2 - 40, W // 2 - 60), (H // 2 + 30, W // 2 + 50)]
    blob_masks = [make_irregular_source((H, W), c, base_radius=22) for c in seeds]
    source_mask = np.zeros((H, W), dtype=bool)
    for b in blob_masks:
        source_mask |= b

    # --- Background: blobs (1.0) blurred outward, black elsewhere ---
    blob_field = source_mask.astype(float)
    background = gaussian_filter(blob_field, sigma=10)
    if background.max() > 0:
        background = background / background.max()

    # --- Per-pixel traversal cost map (smooth, all positive) ---
    # cost_noise = RNG.standard_normal((H, W))
    # smooth = gaussian_filter(cost_noise, sigma=40)
    # smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-9)
    # cost_map = (0.4 + 1.6 * smooth).astype(np.float32)
    cost_map = np.exp(1.0 - (background.astype(np.float32))) - 1.0

    # --- Run the project's multi_source_dijkstra ---
    # Use the full blob mask as seeds (each blob is its own component, all
    # pixels in the blob start at cost 0). This matches how the real pipeline
    # invokes the algorithm and produces natural-looking, irregular fronts
    # rather than the grid-aligned octagons of point seeds.
    annot_labeled = np.zeros((H, W), dtype=np.int32)
    for cid, blob in enumerate(blob_masks, start=1):
        annot_labeled[blob] = cid
    # for cid, (cy, cx) in enumerate(seeds, start=1):
    #     annot_labeled[cy, cx] = cid

    owner_map, dist_map, _, _ = multi_source_dijkstra(
        cost_map=cost_map,
        annot_labeled=annot_labeled,
        connectivity=8,
    )

    cumulative = dist_map.astype(np.float64)
    finite_max = float(cumulative[np.isfinite(cumulative)].max())
    cumulative[~np.isfinite(cumulative)] = finite_max

    # Cosmetic smoothing: removes 1-pixel staircase artifacts from the
    # discrete grid without changing the overall front shape.
    cumulative = gaussian_filter(cumulative, sigma=2)

    return background, cumulative, source_mask, owner_map


def main():
    background, cumulative, source_mask, owner_map = build_background_and_cost()

    # Three iso-cost levels chosen to extend visibly past the blob boundary
    # (blob radius ~22 px, average traversal ~1.2 -> cost at blob edge ~26).
    # Levels are absolute, in cost units; partial (open) contours are fine.
    diag = float(np.hypot(H, W))

    levels = [0.005 * i * i * diag for i in range(1, 6)]
    level_labels = [f"c = {lvl:.1f}" for lvl in levels]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Static grayscale background — NOT the cumulative cost.
    ax.imshow(background, cmap="gray", vmin=0, vmax=1, origin="upper")
    # show owner_map as a color overlay
    # owner_overlay = np.zeros((*owner_map.shape, 4), dtype=float)
    # owner_overlay[owner_map == 1] = (1.0, 1.0, 0.0, 0.2)  # yellow with alpha
    # owner_overlay[owner_map == 2] = (0.0, 1.0, 1.0, 0.2)  # cyan with alpha
    # ax.imshow(owner_overlay, origin="upper")

    # Source blobs: paint pure white (brightest pixels in the figure).
    src_overlay = np.zeros((*source_mask.shape, 4), dtype=float)
    src_overlay[source_mask] = (1.0, 1.0, 1.0, 1.0)
    ax.imshow(src_overlay, origin="upper")

    # Three diffusion fronts.
    contour_colors = ["#ff3030", "#ff8c1a", "#3ddc97", "#1e90ff", "#8a2be2"]
    cs = ax.contour(
        cumulative,
        levels=levels,
        colors=contour_colors,
        linewidths=2.4,
        linestyles=["-", "--", ":"],
    )
    fmt = {lvl: lbl for lvl, lbl in zip(levels, level_labels)}
    ax.clabel(cs, fmt=fmt, fontsize=11, inline=True)

    ax.axis("off")
    fig.tight_layout()

    out_path = OUTPUT_DIR / "viz_region_grow_demo.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    fig, ax = plt.subplots(figsize=(8, 8))

    # Static grayscale background — NOT the cumulative cost.
    ax.imshow(background, cmap="gray", vmin=0, vmax=1, origin="upper")
    # show owner_map as a color overlay
    owner_overlay = np.zeros((*owner_map.shape, 4), dtype=float)
    owner_overlay[owner_map == 1] = (1.0, 1.0, 0.0, 0.2)  # yellow with alpha
    owner_overlay[owner_map == 2] = (0.0, 1.0, 1.0, 0.2)  # cyan with alpha
    ax.imshow(owner_overlay, origin="upper")

    # Source blobs: paint pure white (brightest pixels in the figure).
    src_overlay = np.zeros((*source_mask.shape, 4), dtype=float)
    src_overlay[source_mask] = (1.0, 1.0, 1.0, 1.0)
    ax.imshow(src_overlay, origin="upper")

    # # Three diffusion fronts.
    # contour_colors = ["#ff3030", "#ff8c1a", "#3ddc97", "#1e90ff", "#8a2be2"]
    # cs = ax.contour(
    #     cumulative,
    #     levels=levels,
    #     colors=contour_colors,
    #     linewidths=2.4,
    #     linestyles=["-", "--", ":"],
    # )
    # fmt = {lvl: lbl for lvl, lbl in zip(levels, level_labels)}
    # ax.clabel(cs, fmt=fmt, fontsize=11, inline=True)

    ax.axis("off")
    fig.tight_layout()

    out_path = OUTPUT_DIR / "viz_region_grow_demo_2.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

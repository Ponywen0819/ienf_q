"""
CLAHE tile-size (n_tile) visualisation against the background-removed input.

For several CLAHE tile sizes n_tile on the same fixed crop used by the other
paper viz scripts, renders both the raw CLAHE output and its *signed difference*
against the CLAHE input — the background-removed green channel (bg_removed, NO
CLAHE):

    diff(n) = CLAHE(n) − bg_removed

The CLAHE input pipeline is: green channel → morphological background removal
→ CLAHE(n_tile) (clip limit and bg-removal held at pipeline defaults). The diff
baseline is bg_removed (the direct CLAHE input), so each diff isolates the net
effect of CLAHE itself. The diff is non-negative, so it uses a discrete
sequential colormap on a 0 → max scale (flat colour bands, not a gradient),
shared across all n_tile so they are directly comparable. The mapping is
non-linear (gamma / PowerNorm): high-value fibre pixels get most of the colour
resolution while the lower background offset is compressed toward the floor
(DIFF_GAMMA controls how hard).

Outputs (one bare panel each, written into this script's directory):
  viz_clahe_grid_n{n}.png        — raw CLAHE output for each n_tile
  viz_clahe_grid_n{n}_diff.png   — CLAHE(n) − bg_removed for each n_tile
"""

import csv
from pathlib import Path

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# === Edit these ===========================================================
BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S236-2_a"
BASE_PATH = BASE_PATH / f"data_0331/{IMAGE_ID}"

# Crop with the largest per-fibre tile=672 vs tile=704 difference across every
# sample (the smallest-tile end of the sweep, where the grid changes fastest).
# Picked by tools/viz/find_672_704.py.
CROP_Y0, CROP_X0, CROP_H, CROP_W = 1016, 4140, 200, 200

# Background-removal kernel and CLAHE clip limit: pipeline defaults.
BG_KERNEL_SIZE = 5
CLAHE_CLIP = 40.0

# Tile sizes n_tile to sweep. Every diff is against the raw green channel, not an n.
TILE_SIZES = [672,704, 736, 768, 800, 832]

# Sequential colormap for the (non-negative) differences, quantised into
# DIFF_LEVELS discrete bands so each magnitude step is a flat block of colour
# (easier to read off than a smooth gradient). Range is 0 → max diff.
DIFF_CMAP = "magma"
DIFF_LEVELS = 15
# Non-linear mapping (PowerNorm gamma) for the diff panels: gamma > 1 compresses
# the low background values toward the floor and spends most of the colour bands
# on the high-value fibre pixels, so fibres stand out. 1.0 = linear.
DIFF_GAMMA = 3.0
# ==========================================================================

OUT_DIR = Path(__file__).parent


def _crop(arr: np.ndarray) -> np.ndarray:
    """Return the fixed crop used by the paper visualisations."""
    return arr[CROP_Y0 : CROP_Y0 + CROP_H, CROP_X0 : CROP_X0 + CROP_W]


def _save_panel(
    img: np.ndarray,
    label: str,
    out_name: str,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    colorbar: bool = False,
    levels: int | None = None,
    gamma: float | None = None,
) -> None:
    """Render a bare image panel (no title/axes) on a fixed scale.

    If ``levels`` is given, the colormap is quantised into that many discrete
    bands (flat colour blocks instead of a smooth gradient). If ``gamma`` is
    given, the value→colour mapping is non-linear (``PowerNorm``): gamma > 1
    compresses low values and expands high ones, so combined with ``levels`` the
    discrete bands are spaced non-linearly (more bands at the high end). If
    ``colorbar`` is set, a colour bar is drawn with numeric ticks at the band
    boundaries (which sit at non-linear value positions when ``gamma`` is used).
    """
    cmap_obj = plt.get_cmap(cmap, levels) if levels else cmap
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    if gamma is not None:
        norm = mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
        im = ax.imshow(img, cmap=cmap_obj, norm=norm)
    else:
        im = ax.imshow(img, cmap=cmap_obj, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        n_ticks = levels + 1 if levels else 5
        edges = np.linspace(0.0, 1.0, n_ticks)
        # Band boundaries in value space (inverse of the PowerNorm) so the ticks
        # line up with the discrete colour blocks.
        frac = edges ** (1.0 / gamma) if gamma is not None else edges
        ticks = (vmin + (vmax - vmin) * frac).tolist()
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.0f}" for t in ticks])
        cbar.ax.tick_params(labelsize=11, length=3, width=1.0)
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved: {out_path}  ({label})")


def main() -> None:
    # ── load input + background removal (the CLAHE input) ────────────────────
    image_rgb = cv2.imread(f"{BASE_PATH}/image.png", cv2.IMREAD_COLOR_RGB)
    if image_rgb is None:
        raise FileNotFoundError(f"image.png not found under {BASE_PATH}")
    green = image_rgb[:, :, 1]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (BG_KERNEL_SIZE, BG_KERNEL_SIZE)
    )
    background = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
    bg_removed = cv2.subtract(green, background)

    # The diff baseline: the background-removed green crop (the direct CLAHE
    # input, NO CLAHE; int16 so the signed difference cannot under/overflow).
    base_crop = _crop(bg_removed).astype(np.int16)

    # ── label (ground-truth annotation) crop, for spatial reference ──────────
    label = cv2.imread(f"{BASE_PATH}/label.png", cv2.IMREAD_GRAYSCALE)
    if label is None:
        print(f"label.png not found under {BASE_PATH} — skipping label panel")
    else:
        _save_panel(
            _crop(label),
            "label crop (ground truth)",
            "viz_clahe_grid_label.png",
            cmap="gray",
            vmin=0,
            vmax=255,
        )

    # ── CLAHE output crop per n_tile ──────────────────────────────────────────
    clahe_crops: dict[int, np.ndarray] = {}
    for n in TILE_SIZES:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(n, n))
        # int16 so the signed difference below cannot under/overflow.
        clahe_crops[n] = _crop(clahe.apply(bg_removed)).astype(np.int16)

    # ── raw output + diff vs bg_removed, shared 0 → max scale ─────────────────
    diffs = {n: crop - base_crop for n, crop in clahe_crops.items()}
    # CLAHE only brightens bg_removed here, so the diff is non-negative; the
    # scale runs 0 → max (any stray negative pixel clips to the lowest band).
    vmax = max(1, max(int(diffs[n].max()) for n in TILE_SIZES))

    for n in TILE_SIZES:
        _save_panel(
            clahe_crops[n],
            f"n_tile={n} (raw output)",
            f"viz_clahe_grid_n{n}.png",
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        _save_panel(
            diffs[n],
            f"n_tile={n} − bg_removed",
            f"viz_clahe_grid_n{n}_diff.png",
            cmap=DIFF_CMAP,
            vmin=0,
            vmax=vmax,
            colorbar=True,
            levels=DIFF_LEVELS,
            gamma=DIFF_GAMMA,
        )
    print(f"Shared diff scale: 0 → {vmax} (grey 8-bit levels, {DIFF_LEVELS} bands, "
          f"γ={DIFF_GAMMA} non-linear)")

    _report_diff_stats(diffs, base_crop)


def _report_diff_stats(diffs: dict[int, np.ndarray], baseline: np.ndarray) -> None:
    """Print (and CSV-dump) per-n_tile difference statistics, fiber vs background.

    Fibre / background pixels are separated by Otsu on the bg_removed crop
    (the diff baseline), so the stats answer the question the diff *image*
    cannot: does each n_tile change fibres more than background? Per region we
    report the signed mean (direction: darker/brighter than bg_removed) and the
    mean magnitude |Δ|; the fibre/background |Δ| ratio summarises "how
    concentrated on fibres" the change is (>1 = mostly on fibres).
    """
    base_u8 = np.clip(baseline, 0, 255).astype(np.uint8)
    _, fiber = cv2.threshold(
        base_u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    fiber = fiber.astype(bool)
    bg = ~fiber
    frac = float(fiber.mean())

    header = (
        f"{'n_tile':>6} {'mean|d|':>8} {'fib_mean':>9} {'bg_mean':>8} "
        f"{'fib|d|':>7} {'bg|d|':>7} {'fib/bg':>7}"
    )
    print(
        f"\nDiff statistics vs bg_removed "
        f"(Otsu fibre mask: {frac*100:.1f}% of crop is fibre)"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for n in TILE_SIZES:
        d = diffs[n].astype(np.float64)
        ad = np.abs(d)
        g = float(ad.mean())
        fib_m = float(d[fiber].mean())
        bg_m = float(d[bg].mean())
        fib_a = float(ad[fiber].mean())
        bg_a = float(ad[bg].mean())
        ratio = fib_a / bg_a if bg_a > 0 else float("inf")
        rows.append((n, g, fib_m, bg_m, fib_a, bg_a, ratio))
        print(
            f"{n:>6} {g:>8.3f} {fib_m:>+9.3f} {bg_m:>+8.3f} "
            f"{fib_a:>7.3f} {bg_a:>7.3f} {ratio:>7.2f}"
        )

    csv_path = OUT_DIR / "viz_clahe_grid_diff_stats.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["n_tile", "mean_abs_diff", "fiber_mean_signed", "bg_mean_signed",
             "fiber_mean_abs", "bg_mean_abs", "fiber_bg_abs_ratio", "fiber_fraction"]
        )
        for n, g, fib_m, bg_m, fib_a, bg_a, ratio in rows:
            w.writerow([n, g, fib_m, bg_m, fib_a, bg_a, ratio, frac])
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()

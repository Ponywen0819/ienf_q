"""
CLAHE tile-size (n_tile) difference visualisation.

For several CLAHE tile sizes n_tile on the same fixed crop used by the other
paper viz scripts, renders the *signed difference* of each n_tile's CLAHE
output against the n_tile = 768 baseline:

    diff(n) = CLAHE(n) − CLAHE(768)

The pipeline up to CLAHE is: green channel → morphological background removal
→ CLAHE(n_tile) (clip limit and bg-removal held at pipeline defaults).
Differences use a diverging colormap centred at zero, and all diff panels share
one symmetric scale so they are directly comparable across n_tile. The
n_tile = 768 panel instead shows the raw CLAHE output (the baseline itself).

Outputs (one bare panel each, written into this script's directory):
  viz_clahe_grid_n768.png        — raw CLAHE output for the n_tile = 768 baseline
  viz_clahe_grid_n{n}_diff.png   — CLAHE(n) − CLAHE(768) for every other n_tile
"""

import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# === Edit these ===========================================================
BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S1196-2_a"
BASE_PATH = BASE_PATH / f"data_0510/{IMAGE_ID}"

# Crop region: placed over a label-dense area of S1196-2_a (the original
# 666,4700 window from the other viz scripts has no annotated fibre here).
CROP_Y0, CROP_X0, CROP_H, CROP_W = 600, 8200, 200, 200

# Background-removal kernel and CLAHE clip limit: pipeline defaults.
BG_KERNEL_SIZE = 5
CLAHE_CLIP = 30.0

# Tile sizes n_tile to sweep; 768 is the baseline every panel is differenced against.
TILE_SIZES = [704, 736, 768, 800, 832]
TILE_REF = 768

# Diverging colormap for signed differences (centred at 0). Positive = red,
# negative = blue.
DIFF_CMAP = "RdBu_r"
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
) -> None:
    """Render a bare image panel (no title/axes) on a fixed scale.

    If ``colorbar`` is set, a colour bar is drawn with numeric ticks spanning
    the symmetric scale (no title, just the tick numbers).
    """
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ticks = np.linspace(vmin, vmax, 5).tolist()
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
    if TILE_REF not in TILE_SIZES:
        raise ValueError(f"TILE_REF={TILE_REF} must be in TILE_SIZES={TILE_SIZES}")

    clahe_crops: dict[int, np.ndarray] = {}
    for n in TILE_SIZES:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(n, n))
        # int16 so the signed difference below cannot under/overflow.
        clahe_crops[n] = _crop(clahe.apply(bg_removed)).astype(np.int16)

    # ── signed differences vs baseline, shared symmetric scale ────────────────
    # The n_tile = TILE_REF panel shows the raw CLAHE output (the baseline
    # itself); every other n shows CLAHE(n) − CLAHE(TILE_REF). The baseline's
    # own diff is zero (kept in the dict only so the stats table is complete).
    baseline = clahe_crops[TILE_REF]
    diffs = {n: crop - baseline for n, crop in clahe_crops.items()}
    vlim = max(1, max(int(np.abs(diffs[n]).max()) for n in TILE_SIZES if n != TILE_REF))

    for n in TILE_SIZES:
        if n == TILE_REF:
            _save_panel(
                baseline,
                f"n_tile={n} (baseline, raw output)",
                f"viz_clahe_grid_n{n}.png",
                cmap="gray",
                vmin=0,
                vmax=255,
            )
        else:
            _save_panel(
                diffs[n],
                f"n_tile={n} − {TILE_REF}",
                f"viz_clahe_grid_n{n}_diff.png",
                cmap=DIFF_CMAP,
                vmin=-vlim,
                vmax=vlim,
                colorbar=True,
            )
    print(f"Shared symmetric diff scale: ±{vlim} (grey 8-bit levels)")

    _report_diff_stats(diffs, baseline)


def _report_diff_stats(diffs: dict[int, np.ndarray], baseline: np.ndarray) -> None:
    """Print (and CSV-dump) per-n_tile difference statistics, fiber vs background.

    Fibre / background pixels are separated by Otsu on the n_tile = TILE_REF
    CLAHE output, so the stats answer the question the diff *image* cannot: does
    each n_tile change fibres more than background? Per region we report the
    signed mean (direction: darker/brighter than baseline) and the mean
    magnitude |Δ|; the fibre/background |Δ| ratio summarises "how concentrated
    on fibres" the change is (>1 = mostly on fibres).
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
        f"\nDiff statistics vs baseline n_tile={TILE_REF} "
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

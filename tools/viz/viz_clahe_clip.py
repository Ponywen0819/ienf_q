"""
CLAHE clip-limit (β) difference visualisation.

For several CLAHE clip limits β on the same fixed crop used by the other paper
viz scripts, renders the *signed difference* of each β's CLAHE output against
the β = 30 baseline:

    diff(β) = CLAHE(β) − CLAHE(30)

The pipeline up to CLAHE is: green channel → morphological background removal
→ CLAHE(β). Differences use a diverging colormap centred at zero, and all
diff panels share one symmetric scale so they are directly comparable across β.
The β = 30 panel instead shows the raw CLAHE output (the baseline itself).

Outputs (one bare panel each, written into this script's directory):
  viz_clahe_clip_b30.png         — raw CLAHE output for the β = 30 baseline
  viz_clahe_clip_b{β}_diff.png   — CLAHE(β) − CLAHE(30) for every other β
"""

import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# === Edit these ===========================================================
BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S222-2_a"
BASE_PATH = BASE_PATH / f"data_0331/{IMAGE_ID}"

# Shared crop region: same as every other viz script.
CROP_Y0, CROP_X0, CROP_H, CROP_W = 666, 4700, 200, 200
# CROP_Y0, CROP_X0, CROP_H, CROP_W = 800,6525,200 , 200

# Background-removal and CLAHE tile size: pipeline defaults (see viz_preprocessing).
BG_KERNEL_SIZE = 5
CLAHE_TILE = 768

# Clip limits β to sweep; 30 is the baseline every panel is differenced against.
CLIP_LIMITS = [10, 20, 30, 40, 50]
BETA_REF = 30

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

    If ``colorbar`` is set, a large colour bar is drawn with numeric ticks
    spanning the symmetric scale (no title, just the tick numbers).
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

    # ── CLAHE output crop per β ───────────────────────────────────────────────
    if BETA_REF not in CLIP_LIMITS:
        raise ValueError(f"BETA_REF={BETA_REF} must be in CLIP_LIMITS={CLIP_LIMITS}")

    clahe_crops: dict[int, np.ndarray] = {}
    for beta in CLIP_LIMITS:
        clahe = cv2.createCLAHE(
            clipLimit=float(beta), tileGridSize=(CLAHE_TILE, CLAHE_TILE)
        )
        # int16 so the signed difference below cannot under/overflow.
        clahe_crops[beta] = _crop(clahe.apply(bg_removed)).astype(np.int16)

    # ── signed differences vs baseline, shared symmetric scale ────────────────
    # The β = BETA_REF panel shows the raw CLAHE output (the baseline itself);
    # every other β shows CLAHE(β) − CLAHE(BETA_REF). The baseline's own diff is
    # zero (kept in the dict only so the stats table below is complete).
    baseline = clahe_crops[BETA_REF]
    diffs = {beta: crop - baseline for beta, crop in clahe_crops.items()}
    vlim = max(1, max(int(np.abs(diffs[b]).max()) for b in CLIP_LIMITS if b != BETA_REF))

    for beta in CLIP_LIMITS:
        if beta == BETA_REF:
            _save_panel(
                baseline,
                f"CLAHE β={beta} (baseline, raw output)",
                f"viz_clahe_clip_b{beta}.png",
                cmap="gray",
                vmin=0,
                vmax=255,
            )
        else:
            _save_panel(
                diffs[beta],
                f"CLAHE β={beta} − β={BETA_REF}",
                f"viz_clahe_clip_b{beta}_diff.png",
                cmap=DIFF_CMAP,
                vmin=-vlim,
                vmax=vlim,
                colorbar=True,
            )
    print(f"Shared symmetric diff scale: ±{vlim} (grey 8-bit levels)")

    _report_diff_stats(diffs, baseline)


def _report_diff_stats(diffs: dict[int, np.ndarray], baseline: np.ndarray) -> None:
    """Print (and CSV-dump) per-β difference statistics, split fiber vs background.

    Fibre / background pixels are separated by Otsu on the β = BETA_REF CLAHE
    output, so the stats answer the question the diff *image* cannot: does each
    β change fibres more than background? Per region we report the signed mean
    (direction: darker/brighter than baseline) and the mean magnitude |Δ|; the
    fibre/background |Δ| ratio summarises "how concentrated on fibres" the
    change is (>1 = mostly on fibres).
    """
    base_u8 = np.clip(baseline, 0, 255).astype(np.uint8)
    _thr, fiber = cv2.threshold(
        base_u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    fiber = fiber.astype(bool)
    bg = ~fiber
    frac = float(fiber.mean())

    header = (
        f"{'beta':>5} {'mean|d|':>8} {'fib_mean':>9} {'bg_mean':>8} "
        f"{'fib|d|':>7} {'bg|d|':>7} {'fib/bg':>7}"
    )
    print(
        f"\nDiff statistics vs baseline β={BETA_REF} "
        f"(Otsu fibre mask: {frac*100:.1f}% of crop is fibre)"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for beta in CLIP_LIMITS:
        d = diffs[beta].astype(np.float64)
        ad = np.abs(d)
        g = float(ad.mean())
        fib_m = float(d[fiber].mean())
        bg_m = float(d[bg].mean())
        fib_a = float(ad[fiber].mean())
        bg_a = float(ad[bg].mean())
        ratio = fib_a / bg_a if bg_a > 0 else float("inf")
        rows.append((beta, g, fib_m, bg_m, fib_a, bg_a, ratio))
        print(
            f"{beta:>5} {g:>8.3f} {fib_m:>+9.3f} {bg_m:>+8.3f} "
            f"{fib_a:>7.3f} {bg_a:>7.3f} {ratio:>7.2f}"
        )

    csv_path = OUT_DIR / "viz_clahe_clip_diff_stats.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["beta", "mean_abs_diff", "fiber_mean_signed", "bg_mean_signed",
             "fiber_mean_abs", "bg_mean_abs", "fiber_bg_abs_ratio", "fiber_fraction"]
        )
        for beta, g, fib_m, bg_m, fib_a, bg_a, ratio in rows:
            w.writerow([beta, g, fib_m, bg_m, fib_a, bg_a, ratio, frac])
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()

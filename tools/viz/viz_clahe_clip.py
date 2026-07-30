"""
CLAHE clip-limit (β) visualisation against the background-removed input.

For several CLAHE clip limits β on the same fixed crop used by the other paper
viz scripts, renders both the raw CLAHE output and its *signed difference*
against the CLAHE input — the background-removed green channel (bg_removed, NO
CLAHE):

    diff(β) = CLAHE(β) − bg_removed

The CLAHE input pipeline is: green channel → morphological background removal
→ CLAHE(β). The diff baseline is bg_removed (the direct CLAHE input), so each
diff isolates the net effect of CLAHE itself. The diff is non-negative, so it
uses a discrete sequential colormap on a 0 → max scale (flat colour bands, not a
gradient), shared across all β so they are directly comparable.

Outputs (one bare panel each, written into this script's directory):
  viz_clahe_clip_b{β}.png            — raw CLAHE output for each β
  viz_clahe_clip_b{β}_diff.png       — CLAHE(β) − bg_removed for each β
The diff panels use a non-linear (gamma / PowerNorm) colour mapping so the
high-value fibre pixels get most of the colour resolution while the lower
background offset is compressed toward the floor (DIFF_GAMMA controls how hard).
"""

import csv
from pathlib import Path

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# === Edit these ===========================================================
BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S1494-2_a"
BASE_PATH = BASE_PATH / f"data_0331/{IMAGE_ID}"

# Crop on annotated fibre where beta 35/40/45 land in distinct CLAHE clip_count
# buckets (P=60 -> clip_count 8/9/10), so adjacent betas differ *on the fibre*.
# Bucket boundaries shift with image size (see clip-count quantisation note).
CROP_Y0, CROP_X0, CROP_H, CROP_W = 2396, 935, 200, 200
# CROP_Y0, CROP_X0, CROP_H, CROP_W = 681, 28, 200, 200      # max-diff (noisy edge)
# CROP_Y0, CROP_X0, CROP_H, CROP_W = 666, 4700, 200, 200    # old S222-2_a crop

# Background-removal and CLAHE tile size: pipeline defaults (see viz_preprocessing).
BG_KERNEL_SIZE = 5
CLAHE_TILE = 768

# Clip limits β to sweep. Every diff is against the raw green channel, not a β.
CLIP_LIMITS = [20, 30, 35, 40,45, 50, 55]

# Sequential colormap for the (non-negative) differences, quantised into
# DIFF_LEVELS discrete bands so each magnitude step is a flat block of colour
# (easier to read off than a smooth gradient). Range is 0 → max diff.
DIFF_CMAP = "magma"
DIFF_LEVELS = 10
# Non-linear mapping (PowerNorm gamma) for the diff panels: gamma > 1 compresses
# the low background values toward the floor and spends most of the colour bands
# on the high-value fibre pixels, so fibres stand out. 1.0 = linear.
DIFF_GAMMA = 2.5
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
    ``colorbar`` is set, a large colour bar is drawn with numeric ticks at the
    band boundaries (which sit at non-linear value positions when ``gamma`` is
    used).
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

    # ── CLAHE output crop per β ───────────────────────────────────────────────
    clahe_crops: dict[int, np.ndarray] = {}
    for beta in CLIP_LIMITS:
        clahe = cv2.createCLAHE(
            clipLimit=float(beta), tileGridSize=(CLAHE_TILE, CLAHE_TILE)
        )
        # int16 so the signed difference below cannot under/overflow.
        clahe_crops[beta] = _crop(clahe.apply(bg_removed)).astype(np.int16)

    # ── raw output + diff vs bg_removed, shared 0 → max scale ─────────────────
    diffs = {beta: crop - base_crop for beta, crop in clahe_crops.items()}
    # CLAHE only brightens bg_removed here, so the diff is non-negative; the
    # scale runs 0 → max (any stray negative pixel clips to the lowest band).
    vmax = max(1, max(int(diffs[b].max()) for b in CLIP_LIMITS))

    for beta in CLIP_LIMITS:
        _save_panel(
            clahe_crops[beta],
            f"CLAHE β={beta} (raw output)",
            f"viz_clahe_clip_b{beta}.png",
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        _save_panel(
            diffs[beta],
            f"CLAHE β={beta} − bg_removed",
            f"viz_clahe_clip_b{beta}_diff.png",
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
    """Print (and CSV-dump) per-β difference statistics, split fiber vs background.

    Fibre / background pixels are separated by Otsu on the bg_removed crop
    (the diff baseline), so the stats answer the question the diff *image*
    cannot: does each β change fibres more than background? Per region we report
    the signed mean (direction: darker/brighter than bg_removed) and the mean
    magnitude |Δ|; the fibre/background |Δ| ratio summarises "how concentrated
    on fibres" the change is (>1 = mostly on fibres).
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
        f"\nDiff statistics vs bg_removed "
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

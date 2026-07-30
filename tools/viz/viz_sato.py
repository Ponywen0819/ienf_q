"""
Sato vesselness contribution visualisation.

Sato vesselness runs over a scale range sigmas = range(start, stop) on top of
the CLAHE-enhanced image. For each range this renders the raw enhanced output
crop and its *signed difference* against the post-CLAHE image — i.e. the Sato
input, the pipeline image up to and including CLAHE but before Sato:

    diff(a,b) = enhanced(a,b) − post_CLAHE

so each diff isolates what the Sato stage adds (red) or removes (blue) relative
to the plain CLAHE intensity, for that scale range. The sigma range is swept as
two 1-D arms (NOT the full 2-D grid):

    start arm (stop fixed = 4):  (1,4)  (2,4)  (3,4)
    stop  arm (start fixed = 1):  (1,4)  (1,2)  (1,3)  (1,5)  (1,6)
    (arms are configured via START_SWEEP / STOP_SWEEP)

The enhanced image is produced by the real pipeline function
``cost_map.build_enhanced_image`` (green → bg removal → CLAHE → Sato →
min-max → uint8) and the post-CLAHE baseline replicates that same pipeline up
to the CLAHE step, so panels reflect exactly what feeds the cost map.
The Sato diff is genuinely signed (Sato brightens *and* darkens relative to
plain CLAHE), so the diff panels use a diverging colormap centred at zero on a
symmetric −v → +v scale (red = Sato brighter, blue = Sato darker), shared
across all ranges so they are directly comparable.

Per range we also report, over the labelled fibre region vs background, the
fibre contrast (fibre_mean − bg_mean) and background noise (bg_std) so the
fibre-loss / noise-amplification trade-off can be read as numbers.

Outputs (into this script's directory):
  viz_sato_label.png              — label (GT) crop, spatial reference
  viz_sato_post_clahe.png         — post-CLAHE image (the diff baseline, Sato input)
  viz_sato_s{a}_{b}.png           — raw enhanced (Sato) output per range
  viz_sato_s{a}_{b}_diff.png      — enhanced(a,b) − post_CLAHE per range
  viz_sato_diff_stats.csv         — per-range contrast/noise + mean|Δ| stats
"""

import csv
from pathlib import Path

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from neural_reconstruction.algorithms.annotation_grow.cost_map import (
    DEFAULT_STRIP_WIDTH,
    _morph_open_subtract,
    apply_within_mask_strips,
    build_enhanced_image,
)
from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically

# === Edit these ===========================================================
BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S1768-2_a"
BASE_PATH = BASE_PATH / f"data_0510/{IMAGE_ID}"

# Crop region: the dataset-wide strongest sigma_max "merge" location, found by
# tools/viz/find_sato_merge.py (threshold-band throat search over all 77
# samples + Sato gap-fill ranking). Here the large scales {4,5} added by
# sigma_max=5 thicken adjacent near-parallel fibres until the narrow gaps
# between them fill in — i.e. distinct fibres bleed into one wider band. The
# crop is centred on the strongest throat (abs ~(1050, 1821)).
CROP_Y0, CROP_X0, CROP_H, CROP_W = 950, 1721, 200, 200

# Pipeline defaults held fixed while the Sato range is swept.
OFFSET_PX = 50
BG_KERNEL_SIZE = 5
CLAHE_CLIP = 30.0
CLAHE_TILE = 768

# Sato sigma range sweep: two 1-D arms through the (1, 4) baseline.
SIGMA_REF = (1, 4)
START_SWEEP = [1, 2, 3]   # vary start, stop fixed at SIGMA_REF[1]
STOP_SWEEP = [2,3,4, 5, 6]    # vary stop, start fixed at SIGMA_REF[0]

# Background mask = pixels at least this far (px) from any labelled fibre, so
# the "noise" measurement is not contaminated by peri-fibre transitions.
BG_FIBRE_MARGIN = 9

# Diff colormap — diverging, centred at zero. The Sato diff is signed (Sato
# both brightens and darkens vs plain CLAHE), so we plot the *signed* difference
# on a symmetric −v → +v scale: red = Sato brighter, blue = Sato darker, white =
# unchanged. Quantised into DIFF_LEVELS discrete bands (even count -> a band
# boundary sits exactly at 0). No gamma here: PowerNorm needs a non-negative
# range, and a diverging map should stay linear about its centre.
DIFF_CMAP = "RdBu_r"
DIFF_LEVELS = 10
# ==========================================================================

OUT_DIR = Path(__file__).parent


def _sigma_configs() -> list[tuple[int, int]]:
    """Baseline first, then the two arms (deduplicated, order preserved)."""
    configs: list[tuple[int, int]] = [SIGMA_REF]
    for start in START_SWEEP:
        cfg = (start, SIGMA_REF[1])
        if cfg not in configs and start < cfg[1]:
            configs.append(cfg)
    for stop in STOP_SWEEP:
        cfg = (SIGMA_REF[0], stop)
        if cfg not in configs and cfg[0] < stop:
            configs.append(cfg)
    return configs


def _crop(arr: np.ndarray) -> np.ndarray:
    """Return the fixed crop used by the visualisation."""
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


def _enhanced(green: np.ndarray, roi_mask: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Pipeline enhanced image for a Sato sigma range (full frame, uint8)."""
    return build_enhanced_image(
        green,
        roi_mask,
        bg_kernel_size=BG_KERNEL_SIZE,
        clahe_clip=CLAHE_CLIP,
        clahe_grid=(CLAHE_TILE, CLAHE_TILE),
        sato_sigmas=range(start, stop),
    )


def _post_clahe(green: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """Pipeline image up to (and including) CLAHE — the Sato input (uint8).

    Mirrors ``build_enhanced_image`` exactly through the CLAHE step (masked
    strip-based background removal → ROI mask → CLAHE), stopping before Sato, so
    the diff baseline matches what the Sato stage actually receives.
    """
    if BG_KERNEL_SIZE > 0:
        corrected = apply_within_mask_strips(
            green,
            roi_mask,
            lambda patch: _morph_open_subtract(patch, BG_KERNEL_SIZE),
            pad=BG_KERNEL_SIZE,
            strip_w=DEFAULT_STRIP_WIDTH,
        )
    else:
        corrected = green
    roi_image = cv2.bitwise_and(corrected, corrected, mask=roi_mask)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    return clahe.apply(roi_image)


def main() -> None:
    image_rgb = cv2.imread(f"{BASE_PATH}/image.png", cv2.IMREAD_COLOR_RGB)
    if image_rgb is None:
        raise FileNotFoundError(f"image.png not found under {BASE_PATH}")
    green = image_rgb[:, :, 1]
    mask = cv2.imread(f"{BASE_PATH}/mask.png", cv2.IMREAD_GRAYSCALE)
    roi_mask = dilate_epidermis_vertically(mask, offset_px=OFFSET_PX)

    # ── label (ground-truth) crop, for spatial reference ─────────────────────
    label = cv2.imread(f"{BASE_PATH}/label.png", cv2.IMREAD_GRAYSCALE)
    if label is None:
        print(f"label.png not found under {BASE_PATH} — skipping label panel")
        fiber = None
    else:
        _save_panel(
            _crop(label), "label crop (ground truth)", "viz_sato_label.png",
            cmap="gray", vmin=0, vmax=255,
        )
        fiber = _crop(label) > 127

    # ── post-CLAHE baseline (the Sato input) ──────────────────────────────────
    base_crop = _crop(_post_clahe(green, roi_mask)).astype(np.int16)
    _save_panel(
        base_crop, "post-CLAHE (Sato input / diff baseline)",
        "viz_sato_post_clahe.png", cmap="gray", vmin=0, vmax=255,
    )

    # ── enhanced-image crop per sigma range ──────────────────────────────────
    configs = _sigma_configs()
    crops: dict[tuple[int, int], np.ndarray] = {
        cfg: _crop(_enhanced(green, roi_mask, *cfg)).astype(np.int16)
        for cfg in configs
    }

    # ── differences vs the post-CLAHE baseline, shared symmetric scale ────────
    # Each range gets a raw enhanced panel plus diff(a,b) = enhanced(a,b) −
    # post_CLAHE, isolating what the Sato stage contributes over plain CLAHE.
    # The diff is signed and plotted on a diverging map about zero (red = Sato
    # brighter, blue = darker), so the −v → +v scale is symmetric.
    diffs = {cfg: crop - base_crop for cfg, crop in crops.items()}
    vlim = max(1, max(int(np.abs(diffs[cfg]).max()) for cfg in configs))

    for cfg in configs:
        a, b = cfg
        _save_panel(
            crops[cfg], f"sigmas=({a},{b}) raw enhanced (Sato)",
            f"viz_sato_s{a}_{b}.png", cmap="gray", vmin=0, vmax=255,
        )
        _save_panel(
            diffs[cfg], f"sigmas=({a},{b}) enhanced − post-CLAHE",
            f"viz_sato_s{a}_{b}_diff.png", cmap=DIFF_CMAP,
            vmin=-vlim, vmax=vlim, colorbar=True, levels=DIFF_LEVELS,
        )
    print(f"Shared diff scale: ±{vlim} (grey 8-bit levels, {DIFF_LEVELS} "
          f"diverging bands centred at 0)")

    _report_stats(configs, crops, diffs, fiber)


def _report_stats(
    configs: list[tuple[int, int]],
    crops: dict[tuple[int, int], np.ndarray],
    diffs: dict[tuple[int, int], np.ndarray],
    fiber: np.ndarray | None,
) -> None:
    """Print + CSV-dump fibre contrast / background noise and diff magnitude.

    Fibre vs background is taken from the GT label crop; background excludes a
    ``BG_FIBRE_MARGIN``-px halo around fibres so ``bg_std`` reflects flat-region
    noise rather than peri-fibre transitions.
    """
    if fiber is None or not fiber.any():
        print("No label fibre pixels in crop — skipping contrast/noise stats")
        return

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * BG_FIBRE_MARGIN + 1, 2 * BG_FIBRE_MARGIN + 1)
    )
    bg = ~cv2.dilate(fiber.astype(np.uint8), k).astype(bool)

    header = (
        f"{'sigmas':>8} {'fib_mean':>8} {'bg_mean':>8} {'contrast':>9} "
        f"{'bg_std':>7} {'mean|d|':>8}"
    )
    print(
        f"\nFibre contrast & background noise per Sato range "
        f"(fibre={int(fiber.sum())}px, background={int(bg.sum())}px)"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for cfg in configs:
        c = crops[cfg].astype(np.float64)
        fm = float(c[fiber].mean())
        bm = float(c[bg].mean())
        contrast = fm - bm
        bg_std = float(c[bg].std())
        mad = float(np.abs(diffs[cfg]).mean())
        rows.append((cfg, fm, bm, contrast, bg_std, mad))
        print(
            f"{f'({cfg[0]},{cfg[1]})':>8} {fm:>8.2f} {bm:>8.2f} {contrast:>9.2f} "
            f"{bg_std:>7.2f} {mad:>8.3f}"
        )

    csv_path = OUT_DIR / "viz_sato_diff_stats.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["sigma_start", "sigma_stop", "fiber_mean", "bg_mean",
             "contrast", "bg_std", "mean_abs_diff"]
        )
        for (start, stop), fm, bm, contrast, bg_std, mad in rows:
            w.writerow([start, stop, fm, bm, contrast, bg_std, mad])
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()

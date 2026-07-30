"""Visualise the effect of adding a fine Sato scale (sigma=0.5) on a fixed crop.

Panels match tools/viz/viz_sato.py: bare matplotlib panels, the diff rendered
with the same diverging colormap + symmetric colorbar.

Two scale sets are compared on the SAME CLAHE input
(green -> morphological bg removal -> ROI mask -> CLAHE; the Sato input):

    LO  sigmas = {1, 2, 3}
    HI  sigmas = {0.5, 1, 2, 3}

The only difference is the extra fine scale 0.5 in HI. Sato returns the
per-pixel MAX vesselness over its scale set, so HI >= LO everywhere and the
diff is NON-NEGATIVE. With the diverging RdBu_r colormap centred at 0 (same as
viz_sato.py) only the positive (red) half is ever used; 0 maps to white.

IMPORTANT — this works on the RAW (pre-normalization) Sato response on purpose.
The production pipeline (build_enhanced_image) min-max stretches Sato to uint8;
the 0.5 contribution here is a tiny fraction of the response max, so it
quantizes to ~0 in the final enhanced image. The honest headline is "adding
sigma=0.5 changes the enhanced image essentially not at all" — this script
shows the raw diff to make that small effect visible at all.

NOTE on the default crop: (950, 1721) was selected by find_sato_merge.py for
the *coarse-scale* gap-bridging mechanism ({4,5}), NOT for thin fibres. The
0.5 scale acts on thin fibres, so point CROP_* at a thin-fibre region for a
representative result.

Edit IMAGE_ID / CROP_* below, then:
    uv run python tools/viz/viz_sato_sigma_diff.py

Outputs (in this directory):
  viz_sigma_diff_label.png        — label (GT) crop, spatial reference
  viz_sigma_diff_lo.png           — raw Sato response, sigmas {1,2,3}
  viz_sigma_diff_hi.png           — raw Sato response, sigmas {0.5,1,2,3}
  viz_sigma_diff_diff.png         — diff HI - LO (RdBu_r, symmetric colorbar)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically

# === Edit these ===========================================================
BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S1768-2_a"
BASE_PATH = BASE_PATH / f"data_0510/{IMAGE_ID}"

# Crop region (top-left + size). See NOTE in the module docstring: the default
# is inherited from the coarse-scale merge search, not a thin-fibre region.
CROP_Y0, CROP_X0, CROP_H, CROP_W = 950, 1721, 200, 200

# Pipeline defaults held fixed while only the Sato sigma set changes.
OFFSET_PX = 50
BG_KERNEL_SIZE = 5
CLAHE_CLIP = 30.0
CLAHE_TILE = 768

# The two scale sets compared. HI = LO plus the fine scale 0.5.
SIGMAS_LO: Sequence[float] = [1, 2, 3]
SIGMAS_HI: Sequence[float] = [0.5, 1, 2, 3]

# Diverging colormap for the difference (same as viz_sato.py, centred at 0).
DIFF_CMAP = "RdBu_r"
# ==========================================================================

OUT_DIR = Path(__file__).parent
# Sato finite-difference halo so the crop's response is not edge-contaminated.
SATO_PAD = int(np.ceil(4 * max(max(SIGMAS_LO), max(SIGMAS_HI)))) + 6


def _fmt(sigmas: Sequence[float]) -> str:
    """'{1,2,3}' style label for a sigma set."""
    return "{" + ",".join(f"{s:g}" for s in sigmas) + "}"


def _green_clahe_input(image_rgb: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """green -> morphological bg removal -> ROI mask -> CLAHE (the Sato input)."""
    green = image_rgb[:, :, 1]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BG_KERNEL_SIZE, BG_KERNEL_SIZE))
    bg = cv2.morphologyEx(green, cv2.MORPH_OPEN, k)
    corrected = cv2.subtract(green, bg)
    corrected = cv2.bitwise_and(corrected, corrected, mask=roi_mask)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    return clahe.apply(corrected)


def _sato_crop(base: np.ndarray, sigmas: Sequence[float]) -> np.ndarray:
    """Raw (pre-normalization) Sato response over the fixed crop, at crop size.

    Computed on a padded patch so the crop interior is free of the filter's
    edge halo, then sliced back to the CROP_* window.
    """
    H, W = base.shape
    py0, px0 = max(0, CROP_Y0 - SATO_PAD), max(0, CROP_X0 - SATO_PAD)
    py1 = min(H, CROP_Y0 + CROP_H + SATO_PAD)
    px1 = min(W, CROP_X0 + CROP_W + SATO_PAD)
    patch = base[py0:py1, px0:px1]
    sato = ski.filters.sato(patch, sigmas=list(sigmas), black_ridges=False)  # type: ignore[arg-type]
    oy, ox = CROP_Y0 - py0, CROP_X0 - px0
    return sato[oy : oy + CROP_H, ox : ox + CROP_W]


def _crop(arr: np.ndarray) -> np.ndarray:
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

    Mirrors viz_sato.py's _save_panel: optional colorbar with numeric ticks
    spanning the (symmetric) scale.
    """
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ticks = np.linspace(vmin, vmax, 5).tolist()
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.3g}" for t in ticks])
        cbar.ax.tick_params(labelsize=11, length=3, width=1.0)
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved: {out_path}  ({label})")


def main() -> None:
    image_rgb = cv2.imread(f"{BASE_PATH}/image.png", cv2.IMREAD_COLOR_RGB)
    if image_rgb is None:
        raise FileNotFoundError(f"image.png not found under {BASE_PATH}")
    mask = cv2.imread(f"{BASE_PATH}/mask.png", cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"mask.png not found under {BASE_PATH}")
    roi_mask = dilate_epidermis_vertically(mask, offset_px=OFFSET_PX)
    base = _green_clahe_input(image_rgb, roi_mask)

    lo_label, hi_label = _fmt(SIGMAS_LO), _fmt(SIGMAS_HI)

    # ── label (ground-truth) crop, for spatial reference ─────────────────────
    label = cv2.imread(f"{BASE_PATH}/label.png", cv2.IMREAD_GRAYSCALE)
    fiber = None
    if label is None:
        print(f"label.png not found under {BASE_PATH} — skipping label panel")
    else:
        _save_panel(
            _crop(label), "label crop (ground truth)", "viz_sigma_diff_label.png",
            cmap="gray", vmin=0, vmax=255,
        )
        fiber = _crop(label) > 127

    # ── raw Sato response per sigma set, shared display scale ────────────────
    lo = _sato_crop(base, SIGMAS_LO)
    hi = _sato_crop(base, SIGMAS_HI)
    smax = max(float(lo.max()), float(hi.max()), 1e-9)
    _save_panel(
        lo, f"raw Sato sigmas={lo_label}", "viz_sigma_diff_lo.png",
        cmap="gray", vmin=0, vmax=smax,
    )
    _save_panel(
        hi, f"raw Sato sigmas={hi_label}", "viz_sigma_diff_hi.png",
        cmap="gray", vmin=0, vmax=smax,
    )

    # ── difference HI - LO, diverging colormap centred at 0 (as viz_sato) ────
    # The diff is non-negative (Sato is a per-pixel max), so only the positive
    # half of the symmetric scale is used; 0 maps to white.
    diff = hi - lo
    vlim = max(float(np.abs(diff).max()), 1e-9)
    _save_panel(
        diff, f"sigmas={hi_label} − {lo_label}", "viz_sigma_diff_diff.png",
        cmap=DIFF_CMAP, vmin=-vlim, vmax=vlim, colorbar=True,
    )
    print(f"Symmetric diff scale: ±{vlim:.4g} (raw Sato response units)")

    # ── honest magnitude summary ─────────────────────────────────────────────
    dmax = float(diff.max())
    print(f"\nraw Sato diff  {hi_label} − {lo_label}  (crop {CROP_W}x{CROP_H})")
    print(f"  Sato response max ≈ {smax:.4g}")
    print(f"  max Δ = {dmax:.4g}  ({100 * dmax / smax:.3f}% of response max)")
    print(f"  mean|Δ| = {np.abs(diff).mean():.4g}   pixels with Δ>1e-6: "
          f"{int((diff > 1e-6).sum())} / {diff.size}")
    if fiber is not None and fiber.any():
        print(f"  on fibre pixels: mean Δ = {diff[fiber].mean():.4g}")
    print("  -> The 0.5 scale's contribution is a tiny fraction of the response; "
          "after the pipeline's uint8 min-max stretch it rounds to ~0, i.e. "
          "adding sigma=0.5 barely changes the final enhanced image.")


if __name__ == "__main__":
    main()

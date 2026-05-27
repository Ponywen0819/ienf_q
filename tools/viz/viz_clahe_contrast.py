"""
CLAHE clip-limit (β) vs contrast chart.

Instead of comparing CLAHE output panels by eye (the differences are subtle),
this quantifies the contrast of the CLAHE output as a function of β and plots
it as a curve. Processing is the same as viz_clahe_clip:

    green channel  ->  morphological background removal  ->  CLAHE(β)

evaluated on the same fixed crop used by the other paper viz scripts.

Two contrast metrics are computed on the crop:

  * Global RMS contrast — std. dev. of all CLAHE-output pixels.
  * Fibre-vs-background contrast — mean intensity of fibre pixels minus mean
    of background pixels, with fibre pixels taken from the skeleton GT
    (label.png). This directly measures how well fibres separate from
    background, which is what drives clDice connectivity.

Output:
  viz_clahe_contrast.png  — two-panel β-vs-contrast chart
"""

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

# Background-removal and CLAHE tile size: pipeline defaults.
BG_KERNEL_SIZE = 5
CLAHE_TILE = 768

# Fine β sweep for the curve, the discrete β values to mark, and the default.
BETA_SWEEP = list(range(2, 61, 2))
BETA_MARKS = [10, 20, 30, 40, 50]
BETA_REF = 30
# ==========================================================================

OUT_DIR = Path(__file__).parent


def _crop(arr: np.ndarray) -> np.ndarray:
    """Return the fixed crop used by the paper visualisations."""
    return arr[CROP_Y0 : CROP_Y0 + CROP_H, CROP_X0 : CROP_X0 + CROP_W]


def main() -> None:
    # ── green channel → morphological background removal ────────────────────
    image_rgb = cv2.imread(f"{BASE_PATH}/image.png", cv2.IMREAD_COLOR_RGB)
    if image_rgb is None:
        raise FileNotFoundError(f"image.png not found under {BASE_PATH}")
    green = image_rgb[:, :, 1]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (BG_KERNEL_SIZE, BG_KERNEL_SIZE)
    )
    background = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
    bg_removed = cv2.subtract(green, background)

    # Skeleton GT → fibre / background masks on the crop.
    label = cv2.imread(f"{BASE_PATH}/label.png", cv2.IMREAD_GRAYSCALE)
    fibre_mask = bg_mask = None
    if label is not None:
        fibre_mask = _crop(label) > 0
        bg_mask = ~fibre_mask
        if not fibre_mask.any() or not bg_mask.any():
            print("  warning: crop has no fibre or no background GT pixels; "
                  "fibre-vs-background curve will be skipped")
            fibre_mask = bg_mask = None
    else:
        print(f"  warning: {BASE_PATH}/label.png not found; "
              "fibre-vs-background curve will be skipped")

    # ── sweep β, measure contrast on the crop ───────────────────────────────
    rms_contrast: list[float] = []
    fb_contrast: list[float] = []
    for beta in BETA_SWEEP:
        clahe = cv2.createCLAHE(
            clipLimit=float(beta), tileGridSize=(CLAHE_TILE, CLAHE_TILE)
        )
        crop = _crop(clahe.apply(bg_removed)).astype(np.float32)
        rms_contrast.append(float(crop.std()))
        if fibre_mask is not None:
            fb_contrast.append(
                float(crop[fibre_mask].mean() - crop[bg_mask].mean())
            )

    betas = np.array(BETA_SWEEP, dtype=float)

    # ── plot ────────────────────────────────────────────────────────────────
    have_fb = fibre_mask is not None
    ncols = 2 if have_fb else 1
    fig, axes = plt.subplots(
        1, ncols, figsize=(6.0 * ncols, 4.6), constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    def _draw(ax, y, ylabel: str, color: str) -> None:
        ax.plot(betas, y, "-", color=color, linewidth=2.0, zorder=2)
        # Highlight the marked β values.
        marks = [b for b in BETA_MARKS if b in BETA_SWEEP]
        idx = [BETA_SWEEP.index(b) for b in marks]
        ax.plot([betas[i] for i in idx], [y[i] for i in idx],
                "o", color=color, markersize=7, zorder=3)
        # Reference β.
        ax.axvline(BETA_REF, color="0.5", linestyle="--", linewidth=1.5,
                   zorder=1, label=f"default β = {BETA_REF}")
        ax.set_xlabel("CLAHE clip limit β", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    _draw(axes[0], rms_contrast, "Global RMS contrast (std. dev.)", "#1f77b4")
    axes[0].set_title("Global contrast vs β", fontsize=13)
    if have_fb:
        _draw(axes[1], fb_contrast,
              "Fibre − background mean intensity", "#d62728")
        axes[1].set_title("Fibre-vs-background contrast vs β", fontsize=13)

    out_path = OUT_DIR / "viz_clahe_contrast.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Console summary at the marked β values.
    print("\n  β    RMS contrast" + ("   fibre-bg contrast" if have_fb else ""))
    for b in BETA_MARKS:
        if b not in BETA_SWEEP:
            continue
        i = BETA_SWEEP.index(b)
        line = f"  {b:>3}  {rms_contrast[i]:>11.3f}"
        if have_fb:
            line += f"   {fb_contrast[i]:>16.3f}"
        print(line)


if __name__ == "__main__":
    main()

"""
White top-hat structuring-element size (r_bg) sweep visualisation.

Renders the background-removal output for several structuring-element sizes
r_bg on the same fixed crop used by the other paper viz scripts. Processing
stops after the background-removal step, i.e.

    green channel  ->  white top-hat background removal (disk of size r_bg)

so the panels isolate the effect of r_bg alone. r_bg = 0 means no background
removal (the green channel passes through unchanged). For reference the raw
green channel is also saved.

As with the CLAHE sweep, the raw panels differ only subtly to the eye, so a
difference map is also rendered for every r_bg: result(r_bg) − result(r_ref),
shown with a diverging colormap centred at zero. Red = brighter than the
reference, blue = darker.

Outputs (one panel each):
  viz_bg_remove_green.png            — raw green channel (input)
  viz_bg_remove_r{r}.png             — bg-removed output for each r in KERNEL_SIZES
  viz_bg_remove_diff_r{r}.png        — result(r) − result(r_ref) difference map
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, SymLogNorm
import numpy as np

# === Edit these ===========================================================
BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S222-2_a"
BASE_PATH = BASE_PATH / f"data_0510/{IMAGE_ID}"

# Shared crop region: same as every other viz script.
CROP_Y0, CROP_X0, CROP_H, CROP_W = 666, 4700, 200, 200

# Structuring-element sizes r_bg to sweep, and the reference r the diff maps
# subtract. r_bg = 0 means no background removal.
KERNEL_SIZES = [0, 3, 5, 7, 9, 11]
R_REF = 5

# Panel colour scale. After background removal the fibre signal sits at low
# intensity, so a plain 0-255 grayscale renders the panels almost black. A
# colormap plus a gamma < 1 (PowerNorm) lifts the dark values so the fibres
# and the differences between r_bg values become visible. All sweep panels
# share the same fixed 0-255 scale so they stay directly comparable.
PANEL_CMAP = "magma"
PANEL_GAMMA = 0.45

# Difference-map colour scale. Most diff pixels sit very close to 0, so a
# linear scale washes everything out. A symmetric-log scale is linear within
# ±DIFF_LINTHRESH and logarithmic beyond, amplifying the small differences.
DIFF_CMAP = "seismic"
DIFF_LINTHRESH = 2.0
# ==========================================================================

OUT_DIR = Path(__file__).parent


def _crop(arr: np.ndarray) -> np.ndarray:
    """Return the fixed crop used by the paper visualisations."""
    return arr[CROP_Y0 : CROP_Y0 + CROP_H, CROP_X0 : CROP_X0 + CROP_W]


def _norm_u8(arr: np.ndarray) -> np.ndarray:
    """Min-max stretch to [0, 255] uint8 for display."""
    a = arr.astype(np.float32)
    vmin, vmax = float(a.min()), float(a.max())
    if vmax > vmin:
        a = (a - vmin) / (vmax - vmin) * 255.0
    return a.astype(np.uint8)


def _bg_remove(green: np.ndarray, r: int) -> np.ndarray:
    """White top-hat background removal with a disk of size r. r = 0 disables it."""
    if r <= 0:
        return green.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    background = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(green, background)


def _save_image(img: np.ndarray, norm, cmap: str, label: str, out_name: str) -> None:
    """Save the image with a colorbar but no text — no title, axes, ticks or labels."""
    fig, ax = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
    im = ax.imshow(img, cmap=cmap, norm=norm)
    ax.set_axis_off()
    # Keep the colorbar strip, strip every bit of text from it.
    # Colorbar keeps its ticks and numbers; only the axis label is omitted.
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.tick_params(labelsize=12)
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {out_path}  ({label})")


def _save_panel(img: np.ndarray, label: str, out_name: str) -> None:
    """Render a sweep panel with a gamma-lifted colormap on a fixed 0-255 scale."""
    norm = PowerNorm(gamma=PANEL_GAMMA, vmin=0, vmax=255)
    _save_image(img, norm, PANEL_CMAP, label, out_name)


def _save_diff(diff: np.ndarray, vlim: float, label: str, out_name: str) -> None:
    """Render a signed difference map with a symmetric-log diverging scale."""
    linthresh = min(DIFF_LINTHRESH, vlim)
    norm = SymLogNorm(linthresh=linthresh, vmin=-vlim, vmax=vlim, base=10)
    _save_image(diff, norm, DIFF_CMAP, label, out_name)


def main() -> None:
    # ── load green channel ──────────────────────────────────────────────────
    image_rgb = cv2.imread(f"{BASE_PATH}/image.png", cv2.IMREAD_COLOR_RGB)
    if image_rgb is None:
        raise FileNotFoundError(f"image.png not found under {BASE_PATH}")
    green = image_rgb[:, :, 1]

    # Reference panel: the raw green channel.
    _save_panel(_crop(green), "green channel (input)", "viz_bg_remove_green.png")

    # ── background removal for each r_bg ────────────────────────────────────
    bg_crops: dict[int, np.ndarray] = {}
    for r in KERNEL_SIZES:
        bg_removed = _bg_remove(green, r)
        crop = _crop(bg_removed)
        bg_crops[r] = crop.astype(np.int16)
        if r == 0:
            title = "r_bg = 0 (no removal)"
        elif r == R_REF:
            title = f"r_bg = {r} (default)"
        else:
            title = f"r_bg = {r}"
        _save_panel(crop, title, f"viz_bg_remove_r{r}.png")

    # ── Difference maps: result(r) − result(r_ref) ──────────────────────────
    if R_REF not in bg_crops:
        print(f"  skip diff maps: reference r_bg={R_REF} not in KERNEL_SIZES")
        return
    ref_crop = bg_crops[R_REF]
    diffs = {r: bg_crops[r] - ref_crop for r in KERNEL_SIZES if r != R_REF}
    # r_bg = 0 (no removal) is a different regime — its diff dwarfs the others,
    # so it gets its own scale. The remaining r_bg values share one symmetric
    # scale so those subtle panels stay directly comparable.
    small = [d for r, d in diffs.items() if r != 0]
    vlim_small = max((np.abs(d).max() for d in small), default=1.0)
    vlim_small = float(max(vlim_small, 1.0))
    for r, diff in diffs.items():
        mae = float(np.abs(diff).mean())
        title = f"r_bg = {r} − r_bg = {R_REF}   (mean |Δ| = {mae:.2f})"
        vlim = float(max(np.abs(diff).max(), 1.0)) if r == 0 else vlim_small
        _save_diff(diff, vlim, title, f"viz_bg_remove_diff_r{r}.png")


if __name__ == "__main__":
    main()

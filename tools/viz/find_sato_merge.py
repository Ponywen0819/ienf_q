"""Search the dataset for a crop where a larger Sato sigma_max visibly merges
two nearby fibres ("bleeding").

Mechanism: skimage's Sato takes the max vesselness over the scale set. The
baseline range(1,4) uses scales {1,2,3}; range(1,6) adds {4,5}. A scale-s
Hessian sees two fibres with an edge-to-edge gap <~2s as ONE wide ridge, so it
brightens the gap between them. The merge therefore only becomes *newly*
visible in a threshold band: gaps the max-scale-3 filter leaves dark but the
max-scale-5 filter fills. Minimum-gap (2-4px) fibres are already bridged at the
baseline and show no change — that was the earlier failed search.

Two passes:
  1. Label-only (cheap, all samples): background distance transform + medial
     axis -> "throat" pixels (gap midlines) with DT in [DT_LO, DT_HI]. Score
     every 200x200 window by throat count via an integral image.
  2. On the top windows: compute the actual Sato gap-fill,
     sato(range(1,STOP_HI)) - sato(range(1,STOP_LO)) at the throat pixels,
     on the real CLAHE input. The window where dark throats light up is the
     merge crop.

This is a *mechanism illustration* search: the e2e table shows sigma_max=6 vs
=4 is ~a wash, so a found crop illustrates the bridging mechanism, it does not
justify sigma_max=4 over 6. Caption accordingly.

Run:
    uv run python tools/viz/find_sato_merge.py --data-dir data_0510 --top 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import skimage as ski
from skimage.morphology import skeletonize

from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically

# Fixed pipeline params (match viz_sato / ablation defaults).
OFFSET_PX = 50
BG_KERNEL_SIZE = 5
CLAHE_CLIP = 30.0
CLAHE_TILE = 768

# Sato scale ranges compared: baseline range(1, STOP_LO) vs range(1, STOP_HI).
STOP_LO = 4   # scales {1,2,3}  (sigma_max = 3)
STOP_HI = 6   # scales {1,2,3,4,5}  (sigma_max = 5)

# Throat band: background distance-transform value at the gap midline (px).
# DT = half the edge-to-edge gap, so [2,6] targets gaps ~4-12px wide.
DT_LO = 2.0
DT_HI = 6.0

WIN = 200          # crop size
STRIDE = 50        # window scan stride for pass 1
SATO_PAD = int(np.ceil(4 * (STOP_HI - 1))) + 6   # filter halo for crop Sato


def _green_clahe_input(image_rgb: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """green -> morphological bg removal -> ROI mask -> CLAHE (the Sato input)."""
    green = image_rgb[:, :, 1]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BG_KERNEL_SIZE, BG_KERNEL_SIZE))
    bg = cv2.morphologyEx(green, cv2.MORPH_OPEN, k)
    corrected = cv2.subtract(green, bg)
    corrected = cv2.bitwise_and(corrected, corrected, mask=roi_mask)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    return clahe.apply(corrected)


def _throat_mask(label: np.ndarray, roi_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (throat_mask, dt): gap-midline pixels with DT in [DT_LO, DT_HI]."""
    fiber = label > 127
    roi = roi_mask > 0
    bg = roi & ~fiber
    dt = cv2.distanceTransform(bg.astype(np.uint8), cv2.DIST_L2, 5)
    # Medial axis of the background = ridge midlines of narrow channels = the
    # throats between two fibres. Restricting DT to a small band keeps only
    # narrow gaps (open background has large DT and is excluded).
    medial = skeletonize(bg)
    throat = medial & (dt >= DT_LO) & (dt <= DT_HI)
    return throat, dt


def _best_window(throat: np.ndarray) -> tuple[int, int, int]:
    """Best (y0, x0, count) 200x200 window by throat-pixel count (integral image)."""
    integ = cv2.integral(throat.astype(np.uint8))  # (H+1, W+1)
    H, W = throat.shape
    best = (0, 0, -1)
    for y0 in range(0, max(1, H - WIN + 1), STRIDE):
        y1 = y0 + WIN
        for x0 in range(0, max(1, W - WIN + 1), STRIDE):
            x1 = x0 + WIN
            c = int(
                integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
            )
            if c > best[2]:
                best = (y0, x0, c)
    return best


def _sato_crop(base: np.ndarray, y0: int, x0: int, stop: int) -> np.ndarray:
    """Raw (pre-normalization) Sato response on a padded crop, returned at WIN size."""
    H, W = base.shape
    py0, px0 = max(0, y0 - SATO_PAD), max(0, x0 - SATO_PAD)
    py1, px1 = min(H, y0 + WIN + SATO_PAD), min(W, x0 + WIN + SATO_PAD)
    patch = base[py0:py1, px0:px1]
    sato = ski.filters.sato(patch, sigmas=range(1, stop), black_ridges=False)
    return sato[y0 - py0 : y0 - py0 + WIN, x0 - px0 : x0 - px0 + WIN]


def _gapfill_score(
    base: np.ndarray, throat: np.ndarray, y0: int, x0: int
) -> tuple[float, np.ndarray, np.ndarray]:
    """Mean newly-lit response at throat pixels going STOP_LO -> STOP_HI.

    Both Sato crops are scaled by the SAME factor (the high-sigma crop max) so
    "newly lit" = throat pixels dark in sato_lo but bright in sato_hi.
    """
    s_lo = _sato_crop(base, y0, x0, STOP_LO)
    s_hi = _sato_crop(base, y0, x0, STOP_HI)
    scale = float(s_hi.max()) or 1.0
    lo_n = s_lo / scale
    hi_n = s_hi / scale
    tw = throat[y0 : y0 + WIN, x0 : x0 + WIN]
    if not tw.any():
        return 0.0, s_lo, s_hi
    newly_lit = np.clip(hi_n - lo_n, 0, None)
    return float(newly_lit[tw].mean()), s_lo, s_hi


def _norm_u8(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    lo, hi = a.min(), a.max()
    if hi <= lo:
        return np.zeros_like(a, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data_0510"))
    ap.add_argument("--top", type=int, default=12,
                    help="how many top-throat windows to evaluate with Sato")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "sato_merge_search")
    args = ap.parse_args()

    samples = sorted(
        p for p in args.data_dir.iterdir()
        if p.is_dir() and (p / "label.png").exists()
    )
    print(f"Pass 1: throat scan over {len(samples)} labelled samples")

    # ── pass 1: label-only throat scoring ────────────────────────────────────
    cands: list[tuple[int, str, int, int]] = []  # (count, sid, y0, x0)
    for p in samples:
        label = cv2.imread(str(p / "label.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(p / "mask.png"), cv2.IMREAD_GRAYSCALE)
        if label is None or mask is None:
            continue
        roi = dilate_epidermis_vertically(mask, offset_px=OFFSET_PX)
        throat, _ = _throat_mask(label, roi)
        y0, x0, c = _best_window(throat)
        cands.append((c, p.name, y0, x0))
        print(f"  {p.name:<14} best throat window=({y0},{x0}) count={c}")

    cands.sort(reverse=True)
    top = cands[: args.top]

    # ── pass 2: Sato gap-fill on top windows ─────────────────────────────────
    print(f"\nPass 2: Sato gap-fill on top {len(top)} windows "
          f"(scales 1..{STOP_LO-1} vs 1..{STOP_HI-1})")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scored: list[tuple[float, str, int, int, int]] = []
    cache: dict[str, np.ndarray] = {}
    for c, sid, y0, x0 in top:
        p = args.data_dir / sid
        if sid not in cache:
            img = cv2.imread(str(p / "image.png"), cv2.IMREAD_COLOR_RGB)
            mask = cv2.imread(str(p / "mask.png"), cv2.IMREAD_GRAYSCALE)
            roi = dilate_epidermis_vertically(mask, offset_px=OFFSET_PX)
            cache[sid] = _green_clahe_input(img, roi)
        base = cache[sid]
        label = cv2.imread(str(p / "label.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(p / "mask.png"), cv2.IMREAD_GRAYSCALE)
        roi = dilate_epidermis_vertically(mask, offset_px=OFFSET_PX)
        throat, _ = _throat_mask(label, roi)
        score, s_lo, s_hi = _gapfill_score(base, throat, y0, x0)
        scored.append((score, sid, y0, x0, c))
        print(f"  {sid:<14} window=({y0},{x0}) throats={c} gapfill={score:.4f}")

    scored.sort(reverse=True)
    print("\n=== ranked by Sato gap-fill (higher = more visible merge) ===")
    for rank, (score, sid, y0, x0, c) in enumerate(scored, 1):
        print(f"  #{rank}  {sid:<14} crop=({y0},{x0},{WIN},{WIN}) "
              f"throats={c} gapfill={score:.4f}")

    # ── dump panels for the top 4 so they can be eyeballed ───────────────────
    for rank, (score, sid, y0, x0, c) in enumerate(scored[:4], 1):
        p = args.data_dir / sid
        base = cache[sid]
        label = cv2.imread(str(p / "label.png"), cv2.IMREAD_GRAYSCALE)
        s_lo = _sato_crop(base, y0, x0, STOP_LO)
        s_hi = _sato_crop(base, y0, x0, STOP_HI)
        lab_c = label[y0 : y0 + WIN, x0 : x0 + WIN]
        diff = s_hi.astype(np.float64) - s_lo.astype(np.float64)
        strip = np.hstack([
            lab_c,
            _norm_u8(s_lo),
            _norm_u8(s_hi),
            _norm_u8(diff),
        ])
        out = args.out_dir / f"rank{rank}_{sid}_y{y0}_x{x0}.png"
        cv2.imwrite(str(out), strip)
        print(f"  saved {out}  [label | sato(1..{STOP_LO-1}) | sato(1..{STOP_HI-1}) | diff]")


if __name__ == "__main__":
    main()

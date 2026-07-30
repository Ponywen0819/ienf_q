"""Find crop windows that best discriminate an ablation from ``full``.

The crop showcase (tools/viz/viz_ablation_grow.py) only tells a story where the
reconstructions actually diverge. For the connection-step ablations that happens
all over the image, but for the image-enhancement ablations the topology barely
moves except at faint / low-contrast fibres — those spots are what this tool
hunts for.

For a sample and a set of comparison configs it runs ``full`` plus each config,
rasterises the predicted graphs, and builds a per-config DISAGREEMENT map versus
full (pixels where one reconstruction has a fibre and the other does not, beyond
a tolerance so 1px skeleton jitter is ignored). It then slides a window over the
combined disagreement (max across configs, restricted to the ROI) and reports the
top non-overlapping windows as ready-to-paste (x, y, size) CROPS entries.

Run:
    uv run python tools/viz/find_ablation_crops.py --sample S3091-2_b \\
        --configs no no_clahe no_sato no_wth --size 180 --top 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.ablation_annotation_grow import ABLATIONS, run_ablated_inference  # noqa: E402
from tools.evaluate_dataset import _rasterize_graph  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data_0510"
TOL_UM = 1.28

_SWITCHES_BY_NAME = dict(ABLATIONS)


def load_inputs(sample_id: str):
    p = DATA_DIR / sample_id
    image = np.array(Image.open(p / "image.png"))
    mask = np.array(Image.open(p / "mask.png"))
    annotation = np.array(Image.open(p / "weka.png"))
    green = image[:, :, 1] if image.ndim == 3 else image
    return image, green, mask, annotation


def tolerance_px(sample_id: str) -> int:
    path = DATA_DIR / "px_um.json"
    ratio = json.load(path.open()).get(sample_id) if path.exists() else None
    return int(TOL_UM / (ratio if ratio is not None else TOL_UM))


def pred_mask(image, mask, annotation, name) -> np.ndarray:
    graph, _roi = run_ablated_inference(
        image, mask, annotation, switches=_SWITCHES_BY_NAME[name]
    )
    return _rasterize_graph(image.shape, graph, thickness=1)


def disagreement(ref: np.ndarray, other: np.ndarray, tol: int) -> np.ndarray:
    """Pixels where exactly one of the two reconstructions has a fibre (>tol)."""
    if tol > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
        ref_d = cv2.dilate(ref.astype(np.uint8), k) > 0
        other_d = cv2.dilate(other.astype(np.uint8), k) > 0
    else:
        ref_d, other_d = ref, other
    return (ref & ~other_d) | (other & ~ref_d)


def window_scores(diff: np.ndarray, size: int, stride: int):
    """Slide a `size` window with `stride`; return (score, y0, x0) descending."""
    integral = cv2.integral(diff.astype(np.uint8))  # (H+1, W+1)
    H, W = diff.shape
    cands = []
    for y0 in range(0, H - size + 1, stride):
        for x0 in range(0, W - size + 1, stride):
            y1, x1 = y0 + size, x0 + size
            s = (integral[y1, x1] - integral[y0, x1]
                 - integral[y1, x0] + integral[y0, x0])
            if s > 0:
                cands.append((int(s), y0, x0))
    cands.sort(reverse=True)
    return cands


def nms(cands, size: int, top: int):
    """Greedy: keep highest-scoring windows that don't overlap earlier picks."""
    kept = []
    for s, y0, x0 in cands:
        if all(abs(y0 - ky) >= size or abs(x0 - kx) >= size
               for _, ky, kx in kept):
            kept.append((s, y0, x0))
        if len(kept) >= top:
            break
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", default="S3091-2_b")
    ap.add_argument("--configs", nargs="+", default=["no", "no_clahe", "no_sato", "no_wth"],
                    help="ablation names to discriminate against full")
    ap.add_argument("--size", type=int, default=180, help="square crop side (px)")
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    unknown = [c for c in args.configs if c not in _SWITCHES_BY_NAME]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; valid: {sorted(_SWITCHES_BY_NAME)}")

    image, green, mask, annotation = load_inputs(args.sample)
    tol = tolerance_px(args.sample)
    roi = mask[:, :, 0] if mask.ndim == 3 else mask
    roi_bool = roi > 127
    print(f"[load] {args.sample} shape={green.shape} tol={tol}px configs={args.configs}")

    ref = pred_mask(image, mask, annotation, "full")
    combined = np.zeros(green.shape, dtype=bool)
    per_config = {}
    for name in args.configs:
        other = pred_mask(image, mask, annotation, name)
        d = disagreement(ref, other, tol) & roi_bool
        per_config[name] = d
        combined |= d
        print(f"[diff] full vs {name:<10} disagreement px={int(d.sum())}")

    cands = window_scores(combined, args.size, args.stride)
    kept = nms(cands, args.size, args.top)

    print(f"\nTop {len(kept)} discriminative windows (size={args.size}):")
    print("  paste into CROPS as (x, y, size):")
    for s, y0, x0 in kept:
        breakdown = ", ".join(
            f"{n}:{int(per_config[n][y0:y0+args.size, x0:x0+args.size].sum())}"
            for n in args.configs
        )
        print(f"    ({x0}, {y0}, {args.size}),   # score={s}  [{breakdown}]")


if __name__ == "__main__":
    main()

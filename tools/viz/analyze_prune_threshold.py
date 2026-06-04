"""Decisive pre-check for the prune_threshold figure (run BEFORE rendering).

prune_edges drops component-graph bridges whose Dijkstra cost (`weight`)
exceeds the threshold. The e2e sweep already shows prune_threshold=20 is the
optimum (clDice peak; sensitivity rises but precision falls with threshold).
This script verifies the *mechanism* at the bridge level and finds an honest
crop:

  1. Rebuild the component graph PRE-prune for every sample (pipeline defaults).
  2. Classify each candidate bridge by GT overlap: dilate the GT label by a few
     px (bridges run *alongside* fibres, not on the 1px centreline) and call an
     edge CORRECT if most of its `path` lies in that band, else SPURIOUS.
  3. Report the weight distribution of each class and, per threshold T in
     {10,20,30,40,50}, the fraction of correct/spurious bridges kept. If
     correct bridges sit below ~20 and spurious above, the "20 is the knee"
     story holds at the bridge level too.
  4. Crop finder: rank 200x200 windows that contain BOTH a low-cost correct
     bridge (wrongly dropped at T=10) AND a high-cost spurious bridge (wrongly
     admitted at T=30), so the figure shows both failure modes honestly.

Run:
    uv run python tools/viz/analyze_prune_threshold.py --data-dir data_0510
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Allow `from tools.ablation_annotation_grow import ...` when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_reconstruction.algorithms.annotation_grow.dijkstra import (
    get_components,
    multi_source_dijkstra,
)
from neural_reconstruction.algorithms.annotation_grow.graph_builder import (
    build_component_graph,
    find_meeting_points,
)
from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically

# Reuse the exact ablation building blocks so enhancement/cost match the table.
from tools.ablation_annotation_grow import (
    FIXED_PARAMS,
    build_cost_map_ablated,
    build_enhanced_image_ablated,
)

THRESHOLDS = [10.0, 20.0, 30.0, 40.0, 50.0]
GT_DILATE = 5          # px: half-width of the "alongside a real fibre" band
CORRECT_FRAC = 0.6     # path fraction inside the GT band to call a bridge correct
WIN = 500              # overlay region size (spurious bridges are long-range)


def _component_graph(image, mask, annotation):
    """Pipeline up to build_component_graph (PRE-prune), pipeline defaults."""
    green = image[:, :, 1] if image.ndim == 3 else image
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if annotation.ndim == 3:
        annotation = annotation[:, :, 0]
    roi_mask = dilate_epidermis_vertically(mask, offset_px=FIXED_PARAMS["offset_px"])
    roi_image = build_enhanced_image_ablated(
        green=green, roi_mask=roi_mask,
        use_wth=True, use_clahe=True, use_sato=True,
        bg_kernel_size=FIXED_PARAMS["bg_kernel_size"],
        clahe_clip=FIXED_PARAMS["clahe_clip"],
        clahe_grid=FIXED_PARAMS["clahe_grid"],
        sato_sigmas=range(
            FIXED_PARAMS["sato_sigmas_start"], FIXED_PARAMS["sato_sigmas_stop"]
        ),
    )
    roi_annotation = cv2.bitwise_and(annotation, annotation, mask=roi_mask)
    cost_map = build_cost_map_ablated(roi_image, mode="exp")
    annotation_bin = (roi_annotation > 127).astype(np.uint8)
    annot_labeled = get_components(annotation_bin)
    n_components = int(annot_labeled.max())
    owner, dist, py, px = multi_source_dijkstra(
        cost_map, annot_labeled,
        connectivity=FIXED_PARAMS["connectivity"], roi_mask=(roi_mask > 127),
    )
    connections = find_meeting_points(owner, dist, py, px)
    g = build_component_graph(connections, n_components)
    return g, roi_mask


def _gt_band(label: np.ndarray) -> np.ndarray:
    fiber = (label[:, :, 0] if label.ndim == 3 else label) > 127
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * GT_DILATE + 1, 2 * GT_DILATE + 1)
    )
    return cv2.dilate(fiber.astype(np.uint8), k).astype(bool)


def _classify(g, band: np.ndarray) -> list[dict]:
    """One record per bridge: weight, correctness, path, bbox centre."""
    H, W = band.shape
    out = []
    for a, b, d in g.edges(data=True):
        path = d.get("path", [])
        if not path:
            continue
        ys = np.array([p[0] for p in path])
        xs = np.array([p[1] for p in path])
        inside = band[np.clip(ys, 0, H - 1), np.clip(xs, 0, W - 1)]
        frac = float(inside.mean())
        out.append({
            "weight": float(d["weight"]),
            "correct": frac >= CORRECT_FRAC,
            "frac": frac,
            "cy": int(ys.mean()), "cx": int(xs.mean()),
            "y0": int(ys.min()), "x0": int(xs.min()),
            "y1": int(ys.max()), "x1": int(xs.max()),
            "len": len(path),
        })
    return out


def _best_region(edges: list[dict], win: int) -> tuple[int, int, int]:
    """Best (y0,x0,score) win x win window for the overlay: a region rich in
    both droppable-correct (weight 10-20) and admittable-spurious (20-35)
    bridges, scored on path midpoints so both failure modes appear together.
    """
    cor = [(e["cy"], e["cx"]) for e in edges if e["correct"] and 10 <= e["weight"] <= 20]
    spu = [(e["cy"], e["cx"]) for e in edges if (not e["correct"]) and 20 < e["weight"] <= 35]
    if not cor or not spu:
        return -1, -1, -1
    pts = [(y, x, 1) for y, x in cor] + [(y, x, 1) for y, x in spu]
    best = (-1, -1, -1)
    ys = sorted({p[0] for p in pts})
    xs = sorted({p[1] for p in pts})
    for y0 in ys:
        for x0 in xs:
            nc = sum(1 for y, x in cor if y0 <= y < y0 + win and x0 <= x < x0 + win)
            ns = sum(1 for y, x in spu if y0 <= y < y0 + win and x0 <= x < x0 + win)
            score = min(nc, ns) * 100 + nc + ns  # require BOTH classes present
            if nc and ns and score > best[2]:
                best = (y0, x0, score)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data_0510"))
    ap.add_argument("--limit", type=int, default=0, help="cap #samples (0=all)")
    args = ap.parse_args()

    samples = sorted(
        p for p in args.data_dir.iterdir()
        if p.is_dir() and (p / "label.png").exists()
    )
    if args.limit:
        samples = samples[: args.limit]
    print(f"Building pre-prune component graphs for {len(samples)} samples")

    all_edges: list[dict] = []
    per_sample: dict[str, list[dict]] = {}
    for p in samples:
        try:
            image = np.array(Image.open(p / "image.png"))
            mask = np.array(Image.open(p / "mask.png"))
            annotation = np.array(Image.open(p / "weka.png"))  # loader's annotation_path
            label = np.array(Image.open(p / "label.png"))
        except FileNotFoundError as e:
            print(f"  {p.name}: missing file ({e}); skip")
            continue
        g, _ = _component_graph(image, mask, annotation)
        band = _gt_band(label)
        edges = _classify(g, band)
        per_sample[p.name] = edges
        all_edges.extend(edges)
        nc = sum(e["correct"] for e in edges)
        print(f"  {p.name:<14} edges={len(edges):>4} correct={nc:>4} spurious={len(edges)-nc:>4}")

    cor = np.array([e["weight"] for e in all_edges if e["correct"]])
    spu = np.array([e["weight"] for e in all_edges if not e["correct"]])
    print(f"\nCandidate bridges: {len(cor)} correct, {len(spu)} spurious "
          f"(GT band ±{GT_DILATE}px, correct if >={CORRECT_FRAC:.0%} of path inside)")

    def pct(a, q):
        return float(np.percentile(a, q)) if len(a) else float("nan")
    print(f"{'class':>8} {'n':>5} {'p25':>7} {'p50':>7} {'p75':>7} {'p90':>7} {'mean':>7}")
    for name, a in (("correct", cor), ("spurious", spu)):
        print(f"{name:>8} {len(a):>5} {pct(a,25):>7.2f} {pct(a,50):>7.2f} "
              f"{pct(a,75):>7.2f} {pct(a,90):>7.2f} "
              f"{(a.mean() if len(a) else float('nan')):>7.2f}")

    print(f"\nFraction kept (weight <= T) per threshold:")
    print(f"{'T':>5} {'correct_kept':>13} {'spurious_kept':>14}")
    for T in THRESHOLDS:
        ck = float((cor <= T).mean()) if len(cor) else float("nan")
        sk = float((spu <= T).mean()) if len(spu) else float("nan")
        print(f"{T:>5} {ck:>12.1%} {sk:>13.1%}")

    # ── save artifacts for the figure script (avoid recomputing 77 graphs) ────
    out_dir = Path(__file__).parent / "prune_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "weights.npz", correct=cor, spurious=spu,
             thresholds=np.array(THRESHOLDS))
    # drop paths (huge) before pickling; geometry summary is enough for ranking.
    with open(out_dir / "per_sample_edges.pkl", "wb") as f:
        pickle.dump(per_sample, f)
    print(f"\nSaved artifacts to {out_dir}/ (weights.npz, per_sample_edges.pkl)")

    # ── overlay region finder: a window holding BOTH failure modes ───────────
    print(f"\nOverlay region candidates ({WIN}px window with both 10-20 correct "
          f"and 20-35 spurious bridges):")
    ranked = []
    for sid, edges in per_sample.items():
        y0, x0, score = _best_region(edges, WIN)
        if score > 0:
            ranked.append((score, sid, y0, x0))
    ranked.sort(reverse=True)
    for score, sid, y0, x0 in ranked[:10]:
        print(f"  {sid:<14} region=({y0},{x0},{WIN},{WIN}) score={score}")
    if not ranked:
        print("  (none found at this window size)")


if __name__ == "__main__":
    main()

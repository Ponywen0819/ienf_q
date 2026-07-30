"""prune_threshold visualisation (two figures).

prune_edges drops component-graph bridges whose Dijkstra cost (`weight`) exceeds
the threshold. The e2e sweep makes prune_threshold=20 the clDice optimum, and
the bridge-level analysis (tools/viz/analyze_prune_threshold.py) shows why:
correct bridges (running alongside a real GT fibre) carry low cost, spurious
bridges (crossing background between unrelated fibres) carry high cost, and 20
sits in the valley between the two distributions.

Figure 1 — quantitative (from cached weights.npz):
    viz_prune_hist.png     histogram of correct vs spurious bridge weights with
                           the threshold marked (the separation that 20 exploits)
    viz_prune_kept.png     fraction of correct/spurious bridges kept vs threshold
                           (the precision/sensitivity knee at 20)

Figure 2 — overlay, two crops of ONE sample, each swept over the full tau set
(THRESHOLDS = 10, 15, 20, 25, 30, 40):
  Per-pixel prediction-vs-GT confusion map, three colours only (pred and GT are
  each dilated by COV_TOL before comparing):
    yellow = pred & GT      (overlap / true positive)
    green  = GT, no pred    (missing / false negative)
    red    = pred, no GT    (spurious / false positive)
  As tau rises more bridges are admitted: green shrinks, red grows.
    viz_prune_overlay_{frag,over}_label.png        GT label (all yellow).
    viz_prune_overlay_{frag,over}_t{10..40}.png    confusion map per tau.

Run (after analyze_prune_threshold.py has written prune_analysis/):
    uv run python tools/viz/viz_prune_threshold.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_reconstruction.algorithms.annotation_grow.dijkstra import (  # noqa: E402
    get_components,
)
from neural_reconstruction.algorithms.annotation_grow.graph_builder import (  # noqa: E402
    minimum_spanning_forest,
    prune_edges,
)
from tools.ablation_annotation_grow import (  # noqa: E402
    FIXED_PARAMS,
    build_enhanced_image_ablated,
)
from tools.viz.analyze_prune_threshold import (  # noqa: E402
    CORRECT_FRAC,
    _component_graph,
    _gt_band,
)

# === Edit these ===========================================================
DATA_DIR = Path("/home/pony/projects/ienf_q/data_0510")

# Overlay = a GT COVERAGE ERROR MAP shown for two crops of one sample, one per
# failure mode of the U. The GT label is coloured by whether the reconstruction
# (annotation bodies + the bridges kept at tau) reaches it — yellow=covered,
# red=uncovered — and the bridges that differ from the chosen tau are overlaid:
#   frag crop (tau < chosen): bridges LOST are drawn green. Under-pruning drops
#       real connections, so coverage breaks up (red gaps appear on GT) and the
#       missing links are the green segments. tau=20 is clean, tau=10 is broken.
#   over crop (tau > chosen): bridges ADMITTED are drawn red (same error colour
#       as uncovered GT). Over-pruning admits spurious links that run off-fibre
#       across background; coverage barely improves. tau=20 clean, tau=30 adds.
#
# (No algorithmic correct/spurious claim is made — coverage is measured against
# the GT label directly. The sample was found by a dataset-wide search for one
# with both a fragmentation-rich and an admission-rich window.)
OVERLAY_SID = "S3091-2_b"
OVERLAY_REGIONS = {            # name: (y0, x0, win, mode)  mode: fragment|overconn
    "frag": (297, 3986, 180, "fragment"),
    "over": (305, 1790, 140, "overconn"),
}

CHOSEN = 20.0
# One sweep applied to BOTH crops. Diff colour is driven by T vs CHOSEN:
# T < 20 -> bridges lost (green, missing); T > 20 -> bridges admitted (red).
THRESHOLDS = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0]

# A GT pixel is "covered" if the reconstruction (annotation bodies + kept
# bridges) passes within this many pixels of it.
COV_TOL = 3

# Colours (BGR). The figure is a coverage error map: GT coloured by whether the
# reconstruction reaches it, plus the bridges that differ from the chosen tau.
COLOR_GT = (0, 235, 235)         # yellow, pred & GT overlap (true positive)
COLOR_MISSING = (0, 230, 0)      # green, GT only / no prediction (false negative)
COLOR_ADDED = (60, 60, 255)      # red, prediction only / no GT (false positive)
BG_DIM = 0.45
# ==========================================================================

OUT_DIR = Path(__file__).parent
ART_DIR = OUT_DIR / "prune_analysis"


# ── Figure 1: distribution + kept-fraction curve ──────────────────────────────
def figure_quantitative() -> None:
    npz = np.load(ART_DIR / "weights.npz")
    cor, spu = npz["correct"], npz["spurious"]

    # histogram (clip the long spurious tail so the valley is visible)
    hi = float(np.percentile(spu, 95))
    bins = np.linspace(0, hi, 60).tolist()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(np.clip(cor, 0, hi), bins=bins, alpha=0.6, color="#3a8a3a",
            label=f"correct bridges (n={len(cor)})", density=True)
    ax.hist(np.clip(spu, 0, hi), bins=bins, alpha=0.6, color="#c23a3a",
            label=f"spurious bridges (n={len(spu)})", density=True)
    ax.axvline(CHOSEN, color="k", lw=2, ls="--", label=f"threshold = {CHOSEN:g}")
    ax.set_xlabel("bridge Dijkstra cost (weight)")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "viz_prune_hist.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'viz_prune_hist.png'}")

    # kept fraction vs threshold
    ts = npz["thresholds"]
    ck = [float((cor <= t).mean()) * 100 for t in ts]
    sk = [float((spu <= t).mean()) * 100 for t in ts]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(ts, ck, "-o", color="#3a8a3a", label="correct kept")
    ax.plot(ts, sk, "-o", color="#c23a3a", label="spurious kept")
    ax.axvline(CHOSEN, color="k", lw=2, ls="--", label=f"threshold = {CHOSEN:g}")
    for t, c, s in zip(ts, ck, sk):
        ax.annotate(f"{c:.0f}", (t, c), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8, color="#3a8a3a")
        ax.annotate(f"{s:.0f}", (t, s), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8, color="#c23a3a")
    ax.set_xlabel("prune_threshold")
    ax.set_ylabel("bridges kept (%)")
    ax.set_title("Bridges kept vs threshold")
    ax.set_ylim(-3, 103)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "viz_prune_kept.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'viz_prune_kept.png'}")


# ── Figure 2: per-threshold MST overlay ───────────────────────────────────────
def _edge_geom(g, band: np.ndarray) -> dict:
    """{(a,b): (correct, ys, xs)} for every bridge with a path.

    `correct` = most of the path lies in the dilated GT band (a real fibre).
    """
    H, W = band.shape
    geom: dict[tuple[int, int], tuple[bool, np.ndarray, np.ndarray]] = {}
    for a, b, d in g.edges(data=True):
        path = d.get("path", [])
        if not path:
            continue
        ys = np.array([p[0] for p in path])
        xs = np.array([p[1] for p in path])
        frac = float(band[np.clip(ys, 0, H - 1), np.clip(xs, 0, W - 1)].mean())
        geom[(a, b)] = (frac >= CORRECT_FRAC, ys, xs)
    return geom


def figure_overlay() -> None:
    if OVERLAY_SID == "PLACEHOLDER":
        print("OVERLAY_SID not set — run analyze_prune_threshold.py, then set "
              "OVERLAY_SID / OVERLAY_REGIONS from its region finder.")
        return
    p = DATA_DIR / OVERLAY_SID
    image = np.array(Image.open(p / "image.png"))
    mask = np.array(Image.open(p / "mask.png"))
    annotation = np.array(Image.open(p / "weka.png"))
    label = np.array(Image.open(p / "label.png"))

    g, roi_mask = _component_graph(image, mask, annotation)
    band = _gt_band(label)
    geom = _edge_geom(g, band)

    green = image[:, :, 1] if image.ndim == 3 else image
    enhanced = build_enhanced_image_ablated(
        green=green, roi_mask=roi_mask,
        use_wth=True, use_clahe=True, use_sato=True,
        bg_kernel_size=FIXED_PARAMS["bg_kernel_size"],
        clahe_clip=FIXED_PARAMS["clahe_clip"],
        clahe_grid=FIXED_PARAMS["clahe_grid"],
        sato_sigmas=range(FIXED_PARAMS["sato_sigmas_start"],
                          FIXED_PARAMS["sato_sigmas_stop"]),
    )
    gt_bin = (label[:, :, 0] if label.ndim == 3 else label) > 127

    # Annotation bodies (manual fibre segments). With the kept bridges these
    # form the reconstruction whose GT coverage we colour below.
    ann2d = annotation[:, :, 0] if annotation.ndim == 3 else annotation
    ann_roi = cv2.bitwise_and(ann2d, ann2d, mask=roi_mask)
    ann_lab = get_components((ann_roi > 127).astype(np.uint8))

    # MST graph per threshold (chosen tau only tagged for the filename).
    thresholds = set(THRESHOLDS) | {CHOSEN}
    msts = {T: minimum_spanning_forest(prune_edges(g, threshold=T))
            for T in thresholds}

    # Per-threshold prediction mask = annotation bodies + the kept bridge paths
    # (thin pixels — NOT dilated for display). Following evaluate_dataset, the
    # dilation by COV_TOL is used ONLY to decide overlap (a tolerance), so the
    # painted pixels stay the original thin GT / prediction; only their colour
    # class is decided against the dilated counterpart.
    ann_body = ann_lab > 0
    cov_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * COV_TOL + 1, 2 * COV_TOL + 1))
    gt_dil = cv2.dilate(gt_bin.astype(np.uint8), cov_kernel).astype(bool)

    pred_masks, pred_dils = {}, {}
    for T in thresholds:
        recon = ann_body.copy()
        for a, b in msts[T].edges():
            rec = geom.get(_key((a, b)))
            if rec is not None:
                recon[rec[1], rec[2]] = True
        pred_masks[T] = recon
        pred_dils[T] = cv2.dilate(recon.astype(np.uint8), cov_kernel).astype(bool)

    for name, (y0, x0, win, _mode) in OVERLAY_REGIONS.items():
        bg = (enhanced[y0:y0 + win, x0:x0 + win].astype(np.float64)
              * BG_DIM).astype(np.uint8)
        base = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        gt_w = gt_bin[y0:y0 + win, x0:x0 + win]

        # GT label panel (all yellow), the spatial reference.
        gt_panel = base.copy()
        gt_panel[gt_w] = COLOR_GT
        cv2.imwrite(str(OUT_DIR / f"viz_prune_overlay_{name}_label.png"), gt_panel)
        print(f"Saved: viz_prune_overlay_{name}_label.png  (GT label reference)")

        # Per-pixel prediction-vs-GT confusion map (three colours only), painting
        # the raw thin masks (no dilation on the displayed pixels):
        #   yellow = overlap within tolerance, green = GT only (missing),
        #   red    = prediction only (spurious). Background keeps the dim image.
        gt_dil_w = gt_dil[y0:y0 + win, x0:x0 + win]
        for T in THRESHOLDS:
            canvas = base.copy()
            pred_w = pred_masks[T][y0:y0 + win, x0:x0 + win]
            pred_dil_w = pred_dils[T][y0:y0 + win, x0:x0 + win]
            overlap = (gt_w & pred_dil_w) | (pred_w & gt_dil_w)
            gt_only = gt_w & ~pred_dil_w
            pred_only = pred_w & ~gt_dil_w
            canvas[gt_only] = COLOR_MISSING
            canvas[pred_only] = COLOR_ADDED
            canvas[overlap] = COLOR_GT
            out = OUT_DIR / f"viz_prune_overlay_{name}_t{int(T)}.png"
            cv2.imwrite(str(out), canvas)
            tag = " (chosen)" if T == CHOSEN else ""
            print(f"Saved: {out}  region={name} T={T:g}{tag}  "
                  f"overlap/missing/spurious: "
                  f"{int(overlap.sum())}/{int(gt_only.sum())}/{int(pred_only.sum())}")


def _key(ab: tuple[int, int]) -> tuple[int, int]:
    """Edge key in the (a<b) order build_component_graph stored geometry under."""
    a, b = ab
    return (a, b) if a < b else (b, a)


def main() -> None:
    figure_quantitative()
    figure_overlay()


if __name__ == "__main__":
    main()

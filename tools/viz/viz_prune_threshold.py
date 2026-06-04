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

Figure 2 — overlay, two crops of ONE sample, one per failure mode:
  frag crop (under-pruning):
    viz_prune_overlay_frag_label.png   GT label (yellow).
    viz_prune_overlay_frag_t10.png     MST coloured by connected component: a
    viz_prune_overlay_frag_t20.png     fibre is one colour at tau=20 but breaks
                           into several at tau=10 — topological fragmentation.
  over crop (over-pruning):
    viz_prune_overlay_over_label.png   GT label (yellow).
    viz_prune_overlay_over_t20.png     chosen-tau MST in grey (clean); tau=30
    viz_prune_overlay_over_t30.png     adds bridges in red, visibly off-fibre.

Run (after analyze_prune_threshold.py has written prune_analysis/):
    uv run python tools/viz/viz_prune_threshold.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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

# Overlay sample + zoom window. The figure shows BOTH failure modes of the U,
# each in the encoding that actually makes it visible (verified empirically —
# see below). Two crops of one sample carry the two modes separately.
#
# Under-pruning (tau < chosen): the harm is TOPOLOGICAL FRAGMENTATION, not a
# detour and not a visible gap. Pruning only removes edges (MST10 ⊆ MST20), so
# tau=10 leaves real fibres split into more disconnected trees (frame-wide the
# MST gains ~50% more components at tau=10) — but the pieces stay spatially
# adjacent, so plotting the MST as one colour looks unchanged. We instead
# colour each MST edge by its
# CONNECTED COMPONENT: a fibre that is one component (one colour) at tau=20
# breaks into several colours at tau=10. The colour boundaries are the breaks.
#
# Over-pruning (tau > chosen): extra bridges are ADMITTED. We draw the chosen-
# tau MST as a faint grey reference and highlight the admitted bridges in red;
# compared to the GT label they visibly run off-fibre across background. (No
# algorithmic correct/spurious claim is made — the reader compares to GT.)
#
# One sample carries both modes in two separate crops (found by a dataset-wide
# search for a sample with both a fragmentation-rich and an admission-rich
# window): the "frag" crop uses the component encoding, the "over" crop the
# red-admitted-bridge encoding. Each region declares its own mode.
OVERLAY_SID = "S3091-2_b"
OVERLAY_REGIONS = {            # name: (y0, x0, win, mode)  mode: fragment|overconn
    "frag": (297, 3986, 180, "fragment"),
    "over": (279, 1790, 180, "overconn"),
}

CHOSEN = 20.0
FRAG_THRESHOLDS = [10.0, 20.0]       # component-coloured (under-pruning)
OVERCONN_THRESHOLDS = [20.0, 30.0]   # grey ref (+red admitted) (over-pruning)

# Colours (BGR) and the dimmed-background factor.
COLOR_ADDED = (60, 60, 255)      # red, bridges admitted above the chosen tau
COLOR_GT = (0, 235, 235)         # bright yellow, the GT label panel
COLOR_GT_FAINT = (0, 90, 90)     # dark yellow, GT location shown under overlays
COLOR_REF = (120, 120, 120)      # grey, the chosen-tau MST (over-pruning ref)
BG_DIM = 0.45
W_REF = 1                        # reference skeleton (thin)
W_DELTA = 4                      # highlighted admitted bridges (thick)
W_COMP = 2                       # component-coloured MST edges
COMP_SEED = 7                    # palette RNG seed (reproducible colours)
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

    # MST graphs per threshold; the chosen-tau MST is the over-pruning reference.
    thresholds = set(FRAG_THRESHOLDS) | set(OVERCONN_THRESHOLDS) | {CHOSEN}
    msts = {T: minimum_spanning_forest(prune_edges(g, threshold=T))
            for T in thresholds}
    ref = {_key(ab) for ab in msts[CHOSEN].edges()}

    def _clip(ys, xs, y0, x0, win):
        m = (ys >= y0) & (ys < y0 + win) & (xs >= x0) & (xs < x0 + win)
        if not m.any():
            return None
        return np.stack([xs[m] - x0, ys[m] - y0], axis=1).astype(np.int32)

    def _draw(canvas, keys, y0, x0, win, color, width):
        """Stroke each edge's in-window path in `color`; return #drawn."""
        n = 0
        for k in keys:
            rec = geom.get(k)
            if rec is None:
                continue
            pts = _clip(rec[1], rec[2], y0, x0, win)
            if pts is None:
                continue
            cv2.polylines(canvas, [pts], False, color, width, cv2.LINE_AA)
            n += 1
        return n

    def _draw_components(canvas, mst, y0, x0, win):
        """Colour each MST edge by its connected component (one colour per tree).
        Returns the number of distinct components with an edge in view."""
        rng = np.random.default_rng(COMP_SEED)
        n_comp = 0
        for cc in nx.connected_components(mst):
            colour = tuple(int(v) for v in rng.integers(70, 256, 3))
            drew = False
            for a, b in mst.subgraph(cc).edges():
                rec = geom.get(_key((a, b)))
                if rec is None:
                    continue
                pts = _clip(rec[1], rec[2], y0, x0, win)
                if pts is None:
                    continue
                cv2.polylines(canvas, [pts], False, colour, W_COMP, cv2.LINE_AA)
                drew = True
            n_comp += drew
        return n_comp

    for name, (y0, x0, win, mode) in OVERLAY_REGIONS.items():
        bg = (enhanced[y0:y0 + win, x0:x0 + win].astype(np.float64)
              * BG_DIM).astype(np.uint8)
        base = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        gt_win = gt_bin[y0:y0 + win, x0:x0 + win]

        # GT label panel (bright yellow), the spatial reference.
        gt_panel = base.copy()
        gt_panel[gt_win] = COLOR_GT
        cv2.imwrite(str(OUT_DIR / f"viz_prune_overlay_{name}_label.png"), gt_panel)
        print(f"Saved: viz_prune_overlay_{name}_label.png  (GT label reference)")

        if mode == "fragment":
            # Component-coloured MST: a fibre that is one component (one colour)
            # at the chosen tau breaks into several colours at smaller tau — the
            # colour boundaries are the fragmentation points.
            for T in FRAG_THRESHOLDS:
                canvas = base.copy()
                canvas[gt_win] = COLOR_GT_FAINT
                n_comp = _draw_components(canvas, msts[T], y0, x0, win)
                out = OUT_DIR / f"viz_prune_overlay_{name}_t{int(T)}.png"
                cv2.imwrite(str(out), canvas)
                tag = " (chosen)" if T == CHOSEN else ""
                print(f"Saved: {out}  region={name} T={T:g}{tag}  "
                      f"components in view: {n_comp}")
        elif mode == "overconn":
            # Chosen-tau MST in grey, bridges admitted above it in red (visibly
            # off-fibre vs the GT label). tau=chosen has no admissions -> clean.
            for T in OVERCONN_THRESHOLDS:
                canvas = base.copy()
                canvas[gt_win] = COLOR_GT_FAINT
                _draw(canvas, ref, y0, x0, win, COLOR_REF, W_REF)
                added = {_key(ab) for ab in msts[T].edges()} - ref
                n_added = _draw(canvas, added, y0, x0, win, COLOR_ADDED, W_DELTA)
                out = OUT_DIR / f"viz_prune_overlay_{name}_t{int(T)}.png"
                cv2.imwrite(str(out), canvas)
                tag = " (chosen, reference)" if T == CHOSEN else ""
                print(f"Saved: {out}  region={name} T={T:g}{tag}  "
                      f"bridges admitted vs tau={CHOSEN:g} in view: {n_added}")
        else:
            raise ValueError(f"unknown region mode: {mode!r}")


def _key(ab: tuple[int, int]) -> tuple[int, int]:
    """Edge key in the (a<b) order build_component_graph stored geometry under."""
    a, b = ab
    return (a, b) if a < b else (b, a)


def main() -> None:
    figure_quantitative()
    figure_overlay()


if __name__ == "__main__":
    main()

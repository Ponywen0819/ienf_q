"""PSD-output variant of tools/viz/viz_meet_point.py — one layer per component.

viz_meet_point.py runs top-level (no ``main()`` guard) and its final figure
bakes everything into one flat PNG, so this re-derives the same A/B pair and
meeting points (component picking + Dijkstra territories copied verbatim,
the only trusted-and-imported parts are the real library functions
dilate_epidermis_vertically / get_components / multi_source_dijkstra) and
writes each visual component to its own PSD layer instead:

  * "background"          — territory-coloured crop (opaque)
  * "meeting_points_cost" — all A-B meeting points, coloured by cost
  * "label_a" / "label_b" — the plain "a" / "b" text beside each component
                             (hidden by default: the source script has this
                             call commented out, kept here as an optional
                             layer instead of deleted)
  * "point_xa_ya" / "point_xb_yb" — the minimum-cost meeting point's two
    endpoints, each its own dot + "(x_a, y_a)" / "(x_b, y_b)" label

Note: the source's draw_meet_pixels() has its ax.scatter(...) call
mis-indented outside the `for` loop, so only the *last* point (B) is ever
plotted. Splitting into two independently-drawn layers fixes that as a
side effect — both endpoints now render as their own scatter+label.

Each non-background layer is rendered by giving matplotlib a transparent
axes sized exactly to the crop's pixel grid (mathtext labels need real
matplotlib text rendering, not cv2.putText), then read back as an RGBA
array — see render_overlay().

Uses psd-tools (see tools/viz/viz_ablation_grow_psd for why: pytoshop's
output only opens in lenient readers, not real Photoshop).

Run:
    uv run python tools/viz/viz_meet_point_psd/viz_meet_point_psd.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from PIL import Image
from psd_tools import PSDImage
from skimage.measure import regionprops

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neural_reconstruction.algorithms.annotation_grow.dijkstra import (  # noqa: E402
    get_components,
    multi_source_dijkstra,
)
from neural_reconstruction.core.preprocessing import (  # noqa: E402
    dilate_epidermis_vertically,
)

BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S222-2_a"
BASE_PATH = BASE_PATH / f"data_0331/{IMAGE_ID}"
CROP_Y0, CROP_X0, CROP_H, CROP_W = 755, 4775, 140, 140

# viz_meet_point.py renders at figsize=(6,6)/dpi=200 (1200x1200 px) even
# though the crop itself is only 140x140 — matplotlib upsamples the
# background and draws markers/text at that resolution. Match it here, or
# every layer comes out pixelated with oversized text/dots relative to the
# canvas.
FIG_INCHES, DPI = 6.0, 200
OUT_W = round(FIG_INCHES * DPI)
OUT_H = round(FIG_INCHES * DPI * CROP_H / CROP_W)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def crop(arr: np.ndarray) -> np.ndarray:
    return arr[CROP_Y0 : CROP_Y0 + CROP_H, CROP_X0 : CROP_X0 + CROP_W]


def _component_palette(n_comp: int, seed: int = 0) -> np.ndarray:
    """Identical construction to viz_region_grow.py / viz_meet_point.py so a
    given component ID gets the same colour across all three figures."""
    rng = np.random.default_rng(seed)
    base = np.array(
        [plt.get_cmap("tab20")(i / 20.0)[:3] for i in range(20)]
        + [plt.get_cmap("tab20b")(i / 20.0)[:3] for i in range(20)]
        + [plt.get_cmap("tab20c")(i / 20.0)[:3] for i in range(20)],
        dtype=np.float32,
    )
    if n_comp > len(base):
        extra = rng.uniform(0.15, 0.95, size=(n_comp - len(base), 3)).astype(np.float32)
        base = np.vstack([base, extra])
    colours = base[:n_comp]
    rng.shuffle(colours)
    palette = np.zeros((n_comp + 1, 3), dtype=np.float32)
    palette[1:] = colours
    return palette


def all_meeting_points(owner: np.ndarray, dist: np.ndarray) -> dict:
    """{(A, B): ndarray (n, 5)} rows [ya, xa, yb, xb, cost], A < B."""
    shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
    H, W = owner.shape
    records: dict = {}
    for dy, dx in shifts:
        y0, y1 = max(0, -dy), H - max(0, dy)
        x0, x1 = max(0, -dx), W - max(0, dx)
        oa = owner[y0:y1, x0:x1]
        ob = owner[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        da = dist[y0:y1, x0:x1]
        db = dist[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        meet = (oa > 0) & (ob > 0) & (oa != ob)
        ys, xs = np.where(meet)
        if len(ys) == 0:
            continue
        ya, xa = ys + y0, xs + x0
        yb, xb = ya + dy, xa + dx
        a_ids = oa[ys, xs].astype(int)
        b_ids = ob[ys, xs].astype(int)
        costs = (da[ys, xs] + db[ys, xs]).astype(float)
        for ay, ax_, by, bx, a, b, c in zip(
            ya.tolist(), xa.tolist(), yb.tolist(), xb.tolist(),
            a_ids.tolist(), b_ids.tolist(), costs.tolist(),
        ):
            if a < b:
                key, row = (a, b), [ay, ax_, by, bx, c]
            else:
                key, row = (b, a), [by, bx, ay, ax_, c]
            records.setdefault(key, []).append(row)
    return {k: np.asarray(v, dtype=float) for k, v in records.items()}


def render_overlay(data_shape: tuple[int, int], draw_fn: Callable) -> np.ndarray:
    """Rasterise one matplotlib element onto a transparent OUT_H x OUT_W
    canvas, with axes data coordinates spanning `data_shape` (the crop's
    native pixel grid) in image (row, col) convention: xlim=[0,W], ylim=[H,0].
    Output resolution is decoupled from data_shape so markers/text render at
    viz_meet_point.py's actual on-screen size instead of the raw crop's.
    """
    dh, dw = data_shape
    fig = plt.figure(figsize=(OUT_W / DPI, OUT_H / DPI), dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, dw)
    ax.set_ylim(dh, 0)
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    draw_fn(ax)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    assert buf.shape[:2] == (OUT_H, OUT_W), f"canvas {buf.shape[:2]} != requested {(OUT_H, OUT_W)}"
    return buf


def main() -> None:
    # --- Preprocessing pipeline (identical to viz_meet_point.py) ----------
    green_raw = np.array(cv2.imread(f"{BASE_PATH}/image.png", cv2.IMREAD_COLOR_RGB))[:, :, 1]
    image = green_raw.copy()
    mask = cv2.imread(f"{BASE_PATH}/mask.png", cv2.IMREAD_GRAYSCALE)
    annotation = cv2.imread(f"{BASE_PATH}/weka.png", cv2.IMREAD_GRAYSCALE)

    roi_mask = dilate_epidermis_vertically(mask, offset_px=50)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.subtract(image, background)

    clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(768, 768))
    image = clahe.apply(image)

    image = cv2.bitwise_and(image, image, mask=roi_mask)
    image = ski.filters.sato(image, sigmas=range(3, 8), black_ridges=False)
    image = (image - image.min()) / (image.max() - image.min()) * 255
    image = image.astype(np.uint8)
    cost_map = np.exp(1.0 - (image.astype(np.float32) / 255.0)) - 1.0

    # --- Multi-source Dijkstra expansion -----------------------------------
    annotation_roi = cv2.bitwise_and(annotation, annotation, mask=roi_mask)
    annotation_bin = (annotation_roi > 127).astype(np.uint8)
    annot_labeled = get_components(annotation_bin)

    owner_map, dist_map, _, _ = multi_source_dijkstra(
        cost_map=cost_map, annot_labeled=annot_labeled,
        connectivity=8, roi_mask=(roi_mask > 127),
    )

    meeting = all_meeting_points(owner_map, dist_map)

    # --- Pick A, B: Voronoi-adjacent, both visible in crop, far apart -----
    centroids = {p.label: p.centroid for p in regionprops(annot_labeled)}
    labels_crop = crop(annot_labeled)
    crop_ids = set(int(v) for v in np.unique(labels_crop)) - {0}

    crop_cy, crop_cx = CROP_H / 2.0, CROP_W / 2.0
    candidates = []
    for (a, b), arr in meeting.items():
        if a not in crop_ids or b not in crop_ids:
            continue
        ya, xa = arr[:, 0], arr[:, 1]
        inside = (
            (ya >= CROP_Y0) & (ya < CROP_Y0 + CROP_H)
            & (xa >= CROP_X0) & (xa < CROP_X0 + CROP_W)
        )
        n_inside = int(inside.sum())
        if n_inside < 10:
            continue
        mcy = float(ya[inside].mean()) - CROP_Y0
        mcx = float(xa[inside].mean()) - CROP_X0
        dist_to_center = float(np.hypot(mcy - crop_cy, mcx - crop_cx))
        ca, cb = centroids[a], centroids[b]
        sep = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
        candidates.append((dist_to_center, n_inside, sep, a, b, mcy, mcx))

    if not candidates:
        raise RuntimeError("No Voronoi-adjacent component pair found inside the crop.")

    in_box = [
        c for c in candidates
        if 0.2 * CROP_H <= c[5] <= 0.8 * CROP_H and 0.2 * CROP_W <= c[6] <= 0.8 * CROP_W
    ]
    if in_box:
        in_box.sort(key=lambda c: -c[2])
        chosen = in_box[0]
    else:
        candidates.sort(key=lambda c: c[0])
        chosen = candidates[0]

    _, _, sep, A_ID, B_ID = chosen[:5]
    print(f"Picked A={A_ID}, B={B_ID}  (sep={sep:.1f}px)")

    # --- Territory-coloured background --------------------------------
    palette = _component_palette(int(annot_labeled.max()))
    A_COLOR = palette[A_ID]
    B_COLOR = palette[B_ID]
    OTHER_COLOR = np.array([0.55, 0.55, 0.55], dtype=np.float32)

    green_crop = crop(green_raw)
    owner_crop = crop(owner_map)
    roi_crop = crop(roi_mask)

    disp = np.stack([green_crop] * 3, axis=-1).astype(np.float32) / 255.0
    other_mask = (owner_crop > 0) & (owner_crop != A_ID) & (owner_crop != B_ID)
    disp[other_mask] = 0.55 * disp[other_mask] + 0.45 * OTHER_COLOR
    a_terr = owner_crop == A_ID
    disp[a_terr] = 0.75 * disp[a_terr] + 0.25 * A_COLOR
    b_terr = owner_crop == B_ID
    disp[b_terr] = 0.75 * disp[b_terr] + 0.25 * B_COLOR
    disp[labels_crop == A_ID] = A_COLOR
    disp[labels_crop == B_ID] = B_COLOR
    disp[roi_crop == 0] = 0.0

    # --- Meeting points for the chosen pair (local crop coords) -----------
    arr = meeting[(A_ID, B_ID)]
    mid_y = (arr[:, 0] + arr[:, 2]) / 2.0 - CROP_Y0
    mid_x = (arr[:, 1] + arr[:, 3]) / 2.0 - CROP_X0
    mp_cost = arr[:, 4]
    inside = (mid_y >= 0) & (mid_y < CROP_H) & (mid_x >= 0) & (mid_x < CROP_W)
    arr_inside = arr[inside]
    mid_y, mid_x, mp_cost = mid_y[inside], mid_x[inside], mp_cost[inside]
    imin = int(np.argmin(mp_cost))

    ya_a = arr_inside[imin, 0] - CROP_Y0
    xa_a = arr_inside[imin, 1] - CROP_X0
    yb_b = arr_inside[imin, 2] - CROP_Y0
    xb_b = arr_inside[imin, 3] - CROP_X0

    def crop_centroid(cid: int):
        ys, xs = np.where(labels_crop == cid)
        if len(ys) == 0:
            return None
        return float(ys.mean()), float(xs.mean())

    ca_local = crop_centroid(A_ID)
    cb_local = crop_centroid(B_ID)

    def _clip(ty: float, tx: float) -> tuple[float, float]:
        margin = 12.0
        return float(np.clip(ty, margin, CROP_H - margin)), float(np.clip(tx, margin, CROP_W - margin))

    def offset_away(py: float, px: float, other) -> tuple[float, float]:
        """Push a seed label (a/b) further from the *other* component -
        matches draw_seed_labels(): label = this_c + (this_c - other_c)."""
        label_offset = 22.0
        if other is not None:
            dy, dx = py - other[0], px - other[1]
            norm = float(np.hypot(dy, dx)) or 1.0
            return _clip(py + label_offset * dy / norm, px + label_offset * dx / norm)
        return _clip(py - label_offset, px)

    def offset_toward(py: float, px: float, target) -> tuple[float, float]:
        """Pull a meet-point coordinate label toward its *own* component's
        centroid - matches draw_meet_pixels(): label = py + (comp_c - py)."""
        label_offset = 22.0
        if target is not None:
            dy, dx = target[0] - py, target[1] - px
            norm = float(np.hypot(dy, dx)) or 1.0
            return _clip(py + label_offset * dy / norm, px + label_offset * dx / norm)
        return _clip(py - label_offset, px)

    # --- Build layers --------------------------------------------------
    layers: list[tuple[str, np.ndarray, bool]] = []  # (name, rgba, visible)

    background_rgb = (np.clip(disp, 0.0, 1.0) * 255).astype(np.uint8)
    background_rgb = cv2.resize(background_rgb, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST)
    layers.append(("background", np.dstack([background_rgb, np.full((OUT_H, OUT_W), 255, np.uint8)]), True))

    def draw_cost_scatter(ax):
        ax.scatter(mid_x, mid_y, c=mp_cost, cmap="cool", s=22, edgecolors="none", zorder=3)

    layers.append(("meeting_points_cost", render_overlay((CROP_H, CROP_W), draw_cost_scatter), True))

    for txt, col, this_c, other_c in (("a", A_COLOR, ca_local, cb_local), ("b", B_COLOR, cb_local, ca_local)):
        if this_c is None:
            continue
        ty, tx = offset_away(this_c[0], this_c[1], other_c)

        def draw_label(ax, tx=tx, ty=ty, txt=txt):
            ax.text(tx, ty, txt, color="yellow", fontsize=18, fontweight="bold",
                     ha="center", va="center", zorder=5)

        layers.append((f"label_{txt}", render_overlay((CROP_H, CROP_W), draw_label), False))

    for py, px, txt, col, comp_c, key in (
        (ya_a, xa_a, r"$(x_a,\, y_a)$", A_COLOR, ca_local, "point_xa_ya"),
        (yb_b, xb_b, r"$(x_b,\, y_b)$", B_COLOR, cb_local, "point_xb_yb"),
    ):
        ty, tx_ = offset_toward(py, px, comp_c)

        def draw_point(ax, py=py, px=px, tx_=tx_, ty=ty, txt=txt, col=col):
            ax.scatter(px, py, marker="o", s=80, c=[col], edgecolors="none", zorder=5)
            ax.annotate(
                txt, xy=(px, py), xytext=(tx_, ty),
                color="white", fontsize=14, ha="center", va="center", zorder=6,
                arrowprops=dict(arrowstyle="-", color="white", lw=0.9),
            )

        layers.append((key, render_overlay((CROP_H, CROP_W), draw_point), True))

    # --- Write PSD -----------------------------------------------------
    psd = PSDImage.new("RGBA", (OUT_W, OUT_H))
    for name, rgba, visible in layers:
        layer = psd.create_pixel_layer(Image.fromarray(rgba), name=name, top=0, left=0)
        layer.visible = visible

    out_path = OUTPUT_DIR / f"{IMAGE_ID}_meet_point.psd"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    psd.save(out_path)
    print(f"[save] {out_path}  layers={[n for n, _, _ in layers]}")


if __name__ == "__main__":
    main()

"""Visualise meeting points between two components (Voronoi-adjacent pair).

Reuses the exact crop and preprocessing pipeline of viz_region_grow.py.
Picks two components a, b that are Voronoi-adjacent, relatively far apart,
and both visible in the crop. Shows:
  - a / b territories in their palette colours, other components greyed out
  - All a-b meeting points coloured by cost w = D(a-side) + D(b-side)
  - The minimum-cost meeting point marked with a star
  - Lowercase text labels a / b on the two components
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import skimage as ski
from skimage.measure import regionprops
from mpl_toolkits.axes_grid1 import make_axes_locatable

from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically
from neural_reconstruction.algorithms.annotation_grow.dijkstra import (
    get_components,
    multi_source_dijkstra,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config / crop (shared with viz_region_grow.py)
# ─────────────────────────────────────────────────────────────────────────────
BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S222-2_a"
BASE_PATH = BASE_PATH / f"data_0331/{IMAGE_ID}"
CROP_Y0, CROP_X0, CROP_H, CROP_W = 666, 4700, 200, 200


def crop(arr):
    """Return the fixed crop used by the paper visualizations."""
    return arr[CROP_Y0 : CROP_Y0 + CROP_H, CROP_X0 : CROP_X0 + CROP_W]


def _component_palette(n_comp: int, seed: int = 0) -> np.ndarray:
    """Distinct qualitative colours for component IDs 1..n_comp.

    Identical construction to viz_region_grow.py so a given component ID
    receives the same colour across both figures.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing pipeline (identical to viz_region_grow.py)
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Multi-source Dijkstra expansion
# ─────────────────────────────────────────────────────────────────────────────
annotation_roi = cv2.bitwise_and(annotation, annotation, mask=roi_mask)
annotation_bin = (annotation_roi > 127).astype(np.uint8)
annot_labeled = get_components(annotation_bin)

owner_map, dist_map, _, _ = multi_source_dijkstra(
    cost_map=cost_map,
    annot_labeled=annot_labeled,
    connectivity=8,
    roi_mask=(roi_mask > 127),
)


# ─────────────────────────────────────────────────────────────────────────────
# All meeting-point pixel pairs for every Voronoi-adjacent component pair
# ─────────────────────────────────────────────────────────────────────────────
def all_meeting_points(owner: np.ndarray, dist: np.ndarray) -> dict:
    """
    Return {(A, B): ndarray (n, 5)} with rows [ya, xa, yb, xb, cost], A < B.
    A meeting point is a pair of 8-adjacent pixels owned by different components;
    cost = dist[A-side] + dist[B-side].
    """
    shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
    H, W = owner.shape
    records: dict = {}
    for dy, dx in shifts:
        y0 = max(0, -dy)
        y1 = H - max(0, dy)
        x0 = max(0, -dx)
        x1 = W - max(0, dx)
        oa = owner[y0:y1, x0:x1]
        ob = owner[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        da = dist[y0:y1, x0:x1]
        db = dist[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        meet = (oa > 0) & (ob > 0) & (oa != ob)
        ys, xs = np.where(meet)
        if len(ys) == 0:
            continue
        ya = ys + y0
        xa = xs + x0
        yb = ya + dy
        xb = xa + dx
        a_ids = oa[ys, xs].astype(int)
        b_ids = ob[ys, xs].astype(int)
        costs = (da[ys, xs] + db[ys, xs]).astype(float)
        for ay, ax_, by, bx, a, b, c in zip(
            ya.tolist(), xa.tolist(), yb.tolist(), xb.tolist(),
            a_ids.tolist(), b_ids.tolist(), costs.tolist(),
        ):
            if a < b:
                key = (a, b)
                row = [ay, ax_, by, bx, c]
            else:
                key = (b, a)
                row = [by, bx, ay, ax_, c]
            records.setdefault(key, []).append(row)
    return {k: np.asarray(v, dtype=float) for k, v in records.items()}


meeting = all_meeting_points(owner_map, dist_map)


# ─────────────────────────────────────────────────────────────────────────────
# Pick A, B: Voronoi-adjacent, both visible in crop, centroids far apart
# ─────────────────────────────────────────────────────────────────────────────
centroids = {p.label: p.centroid for p in regionprops(annot_labeled)}  # (row, col)

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
    # Meeting-point centroid in local crop coords
    mcy = float(ya[inside].mean()) - CROP_Y0
    mcx = float(xa[inside].mean()) - CROP_X0
    dist_to_center = float(np.hypot(mcy - crop_cy, mcx - crop_cx))
    ca, cb = centroids[a], centroids[b]
    sep = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
    candidates.append((dist_to_center, n_inside, sep, a, b, mcy, mcx))

if not candidates:
    raise RuntimeError("No Voronoi-adjacent component pair found inside the crop.")

# Prefer pairs whose meeting region sits in the central area of the crop;
# among those, pick the most widely separated component pair.
in_box = [
    c for c in candidates
    if 0.2 * CROP_H <= c[5] <= 0.8 * CROP_H
    and 0.2 * CROP_W <= c[6] <= 0.8 * CROP_W
]
if in_box:
    in_box.sort(key=lambda c: -c[2])  # max centroid separation
    chosen = in_box[0]
else:
    candidates.sort(key=lambda c: c[0])  # fallback: most central
    chosen = candidates[0]

DIST_TO_CENTER, N_INSIDE, SEP, A_ID, B_ID = chosen[:5]
print(
    f"Picked A={A_ID}, B={B_ID}  (sep={SEP:.1f}px, {N_INSIDE} meeting pts in crop, "
    f"meeting-centroid {DIST_TO_CENTER:.1f}px from crop center)"
)


# ─────────────────────────────────────────────────────────────────────────────
# Build the display image
# ─────────────────────────────────────────────────────────────────────────────
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
disp[a_terr] = 0.45 * disp[a_terr] + 0.55 * A_COLOR
b_terr = owner_crop == B_ID
disp[b_terr] = 0.45 * disp[b_terr] + 0.55 * B_COLOR

# Source (annotation) pixels solid
disp[labels_crop == A_ID] = A_COLOR
disp[labels_crop == B_ID] = B_COLOR

# Outside ROI → black
disp[roi_crop == 0] = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Meeting points for the chosen pair (local crop coords)
# ─────────────────────────────────────────────────────────────────────────────
arr = meeting[(A_ID, B_ID)]
mid_y = (arr[:, 0] + arr[:, 2]) / 2.0 - CROP_Y0
mid_x = (arr[:, 1] + arr[:, 3]) / 2.0 - CROP_X0
mp_cost = arr[:, 4]
inside = (mid_y >= 0) & (mid_y < CROP_H) & (mid_x >= 0) & (mid_x < CROP_W)
mid_y, mid_x, mp_cost = mid_y[inside], mid_x[inside], mp_cost[inside]
imin = int(np.argmin(mp_cost))


def crop_centroid(cid: int):
    ys, xs = np.where(labels_crop == cid)
    if len(ys) == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def draw_seed_labels(ax) -> None:
    """Draw plain text 'a' / 'b' beside the two components (away from each other)."""
    ca_local = crop_centroid(A_ID)
    cb_local = crop_centroid(B_ID)
    label_offset = 26.0  # px, placed beside the component
    margin = 16.0
    for txt, col, this_c, other_c in (
        ("a", A_COLOR, ca_local, cb_local),
        ("b", B_COLOR, cb_local, ca_local),
    ):
        if this_c is None:
            continue
        if other_c is not None:
            dy = this_c[0] - other_c[0]
            dx = this_c[1] - other_c[1]
            norm = float(np.hypot(dy, dx)) or 1.0
            ty = this_c[0] + label_offset * dy / norm
            tx = this_c[1] + label_offset * dx / norm
        else:
            ty, tx = this_c[0] - label_offset, this_c[1]
        ty = float(np.clip(ty, margin, CROP_H - margin))
        tx = float(np.clip(tx, margin, CROP_W - margin))
        lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
        stroke = "black" if lum > 0.6 else "white"
        ax.text(
            tx, ty, txt, color=tuple(col), fontsize=18, fontweight="bold",
            ha="center", va="center", zorder=5,
            path_effects=[pe.withStroke(linewidth=3.0, foreground=stroke)],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Output 1: all meeting points coloured by cost
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.0, 6.0), constrained_layout=True)
ax.imshow(disp)
ax.axis("off")

sc = ax.scatter(
    mid_x, mid_y, c=mp_cost, cmap="viridis",
    s=22, edgecolors="white", linewidths=0.3, zorder=3,
)
ax.scatter(
    mid_x[imin], mid_y[imin], marker="*", s=360,
    c="yellow", edgecolors="black", linewidths=1.0, zorder=4,
)
draw_seed_labels(ax)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.04)
fig.colorbar(sc, cax=cax, label=r"meeting cost $w = D_A + D_B$")

out_path = Path(__file__).parent / "viz_meet_point.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

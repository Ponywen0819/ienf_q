"""Render a 10x10 teaching diagram for one region-grow decision.

The figure is intentionally small enough that every cell can carry an
accumulated-cost label, while still showing 2x2 seeds and a low-cost branching
fiber path.
It is a synthetic explanation of the annotation-grow / multi-source Dijkstra
step, not a visualization of one dataset sample.
"""

from __future__ import annotations

import argparse
import heapq
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


GRID_SIZE = 10
OUTPUT_DIR = Path("./output")

SEED_CELLS = {
    1: ((1, 1), (1, 2), (2, 1), (2, 2)),
    2: ((7, 7), (7, 8), (8, 7), (8, 8)),
}

REGION_COLORS = {
    1: "#e53935",
    2: "#2eae53",
}

FRONTIER_COLOR = "#ffca3a"
SELECTED_COLOR = "#00a6fb"
MEETING_COLOR = "#ff1744"
GRID_COLOR = "#171717"

NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def build_cost_map() -> np.ndarray:
    """Build a synthetic traversal-cost grid.

    Lower values are drawn brighter in the final figure.  The low-cost cells
    form a main fiber plus two small branches so that expansion has visible
    choices rather than a single straight tunnel.
    """
    yy, xx = np.mgrid[:GRID_SIZE, :GRID_SIZE]
    cost = 3.35 + 0.28 * np.sin(0.8 * xx) + 0.22 * np.cos(0.7 * yy)
    cost = cost.astype(np.float32)

    fiber_costs = {
        # Red-side trunk.
        (1, 1): 0.35,
        (1, 2): 0.35,
        (2, 1): 0.35,
        (2, 2): 0.35,
        (1, 3): 0.60,
        (2, 3): 0.60,
        (3, 3): 0.60,
        (4, 3): 0.50,
        (4, 4): 0.50,
        (4, 5): 0.60,
        (5, 5): 0.50,
        (6, 5): 0.60,
        (6, 6): 0.60,
        (7, 6): 0.60,
        (7, 7): 0.60,
        (7, 8): 0.35,
        (8, 7): 0.35,
        (8, 8): 0.35,
        # Red branch.
        (3, 1): 0.90,
        (4, 2): 0.80,
        (5, 2): 0.90,
        # Upper branch.
        (2, 4): 1.00,
        (2, 5): 0.85,
        (1, 5): 0.90,
        (1, 6): 0.90,
        # Green-side branch.
        (6, 8): 0.95,
        (6, 7): 0.60,
        (7, 5): 0.95,
    }
    for (y, x), value in fiber_costs.items():
        cost[y, x] = value

    return cost


def all_seed_cells() -> set[tuple[int, int]]:
    """Return all seed cells as a flat set for fast membership checks."""
    return {cell for cells in SEED_CELLS.values() for cell in cells}


def init_search_state(cost_map: np.ndarray) -> dict:
    """Create a Dijkstra-like state with seeds already settled."""
    dist = np.full(cost_map.shape, np.inf, dtype=np.float64)
    owner = np.zeros(cost_map.shape, dtype=np.int32)
    settled = np.zeros(cost_map.shape, dtype=bool)
    heap: list[tuple[float, int, int, int]] = []

    for cid, cells in SEED_CELLS.items():
        for y, x in cells:
            dist[y, x] = 0.0
            owner[y, x] = cid
            settled[y, x] = True

    state = {"dist": dist, "owner": owner, "settled": settled, "heap": heap}
    for cid, cells in SEED_CELLS.items():
        for y, x in cells:
            relax_from_cell(state, cost_map, y, x, cid)
    return state


def relax_from_cell(
    state: dict,
    cost_map: np.ndarray,
    y: int,
    x: int,
    cid: int,
) -> None:
    """Relax 4-neighbor costs from one settled cell."""
    for dy, dx in NEIGHBORS_4:
        ny, nx = y + dy, x + dx
        if not (0 <= ny < GRID_SIZE and 0 <= nx < GRID_SIZE):
            continue
        if state["settled"][ny, nx]:
            continue

        new_dist = float(state["dist"][y, x] + cost_map[ny, nx])
        if new_dist < float(state["dist"][ny, nx]):
            state["dist"][ny, nx] = new_dist
            state["owner"][ny, nx] = cid
            heapq.heappush(state["heap"], (new_dist, ny, nx, cid))


def frontier_mask(state: dict) -> np.ndarray:
    """Return cells that have a finite tentative cost and are not settled."""
    return np.isfinite(state["dist"]) & ~state["settled"]


def peek_next_frontier(state: dict) -> tuple[float, int, int, int]:
    """Find the current minimum-cost frontier cell without mutating the heap."""
    mask = frontier_mask(state)
    if not mask.any():
        raise RuntimeError("No frontier cells remain.")

    candidates = np.argwhere(mask)
    scores = [float(state["dist"][y, x]) for y, x in candidates]
    best_i = int(np.argmin(scores))
    y, x = candidates[best_i]
    cid = int(state["owner"][y, x])
    return float(state["dist"][y, x]), int(y), int(x), cid


def absorb_next_cell(state: dict, cost_map: np.ndarray) -> tuple[float, int, int, int]:
    """Settle the current minimum frontier cell and update its neighbors."""
    d, y, x, cid = peek_next_frontier(state)
    state["settled"][y, x] = True
    relax_from_cell(state, cost_map, y, x, cid)
    return d, y, x, cid


def capture_one_step(
    cost_map: np.ndarray,
    target_cost: float = 2.3,
) -> tuple[dict, dict, tuple]:
    """Return states immediately before and after one highlighted decision."""
    state = init_search_state(cost_map)

    for _ in range(GRID_SIZE * GRID_SIZE):
        next_item = peek_next_frontier(state)
        if round(next_item[0], 1) >= target_cost:
            before = deepcopy(state)
            chosen = absorb_next_cell(state, cost_map)
            after = deepcopy(state)
            return before, after, chosen
        absorb_next_cell(state, cost_map)

    raise RuntimeError("Could not capture a Dijkstra decision step.")


def capture_first_expansion(cost_map: np.ndarray) -> tuple[dict, tuple]:
    """Return the state after the first frontier cell has been absorbed."""
    state = init_search_state(cost_map)
    absorbed = absorb_next_cell(state, cost_map)
    return deepcopy(state), absorbed


def capture_half_expansion(
    cost_map: np.ndarray,
    target_state: dict,
) -> tuple[dict, tuple[float, int, int, int] | None]:
    """Return a Dijkstra state about halfway between seeds and the target state."""
    state = init_search_state(cost_map)
    seed_count = len(all_seed_cells())
    final_count = int(target_state["settled"].sum())
    target_count = max(seed_count + 1, int(round((seed_count + final_count) / 2)))

    while int(state["settled"].sum()) < target_count and frontier_mask(state).any():
        absorb_next_cell(state, cost_map)

    next_choice = peek_next_frontier(state) if frontier_mask(state).any() else None
    return deepcopy(state), next_choice


def capture_complete_expansion(cost_map: np.ndarray) -> dict:
    """Return the state after the multi-source expansion covers the whole grid."""
    state = init_search_state(cost_map)
    while frontier_mask(state).any():
        absorb_next_cell(state, cost_map)
    return deepcopy(state)


def single_source_dist(
    cost_map: np.ndarray,
    seed_cells: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Compute a 4-connected accumulated-cost map from one seed component."""
    dist = np.full(cost_map.shape, np.inf, dtype=np.float64)
    heap = []
    for sy, sx in seed_cells:
        dist[sy, sx] = 0.0
        heapq.heappush(heap, (0.0, sy, sx))

    while heap:
        d, y, x = heapq.heappop(heap)
        if d > float(dist[y, x]):
            continue
        for dy, dx in NEIGHBORS_4:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < GRID_SIZE and 0 <= nx < GRID_SIZE):
                continue
            new_dist = d + float(cost_map[ny, nx])
            if new_dist < float(dist[ny, nx]):
                dist[ny, nx] = new_dist
                heapq.heappush(heap, (new_dist, ny, nx))

    return dist


def build_meeting_state(
    cost_map: np.ndarray,
) -> tuple[dict, tuple[int, int], float, float]:
    """Create a collision snapshot using two independent accumulated costs."""
    red_dist = single_source_dist(cost_map, SEED_CELLS[1])
    green_dist = single_source_dist(cost_map, SEED_CELLS[2])
    meeting = (6, 6)
    red_meet = float(red_dist[meeting])
    green_meet = float(green_dist[meeting])

    owner = np.zeros(cost_map.shape, dtype=np.int32)
    settled = np.zeros(cost_map.shape, dtype=bool)
    shown_dist = np.full(cost_map.shape, np.inf, dtype=np.float64)

    active = (red_dist < red_meet) | (green_dist < green_meet)
    red_side = active & (red_dist <= green_dist)
    green_side = active & ~red_side

    owner[red_side] = 1
    owner[green_side] = 2
    settled[active] = True
    shown_dist[red_side] = red_dist[red_side]
    shown_dist[green_side] = green_dist[green_side]

    my, mx = meeting
    owner[my, mx] = 0
    settled[my, mx] = False
    shown_dist[my, mx] = red_meet + green_meet

    state = {
        "dist": shown_dist,
        "owner": owner,
        "settled": settled,
        "heap": [],
    }
    return state, meeting, red_meet, green_meet


def draw_cost_background(ax: plt.Axes, cost_map: np.ndarray) -> None:
    """Draw cost as grayscale with bright cells for low traversal cost."""
    lo = float(cost_map.min())
    hi = float(cost_map.max())
    brightness = 1.0 - np.clip((cost_map - lo) / (hi - lo), 0.0, 1.0)
    ax.imshow(brightness, cmap="gray", vmin=0, vmax=1, origin="upper")

    for line in np.arange(-0.5, GRID_SIZE, 1.0):
        ax.axhline(line, color=GRID_COLOR, linewidth=0.65, alpha=0.62)
        ax.axvline(line, color=GRID_COLOR, linewidth=0.65, alpha=0.62)

    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(GRID_SIZE - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def add_cell_rect(
    ax: plt.Axes,
    y: int,
    x: int,
    color: str,
    *,
    alpha: float,
    linewidth: float = 1.5,
    fill: bool = True,
    linestyle: str = "-",
) -> None:
    ax.add_patch(
        Rectangle(
            (x - 0.5, y - 0.5),
            1,
            1,
            facecolor=color if fill else "none",
            edgecolor=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            joinstyle="miter",
        )
    )


def label_cell(
    ax: plt.Axes,
    y: int,
    x: int,
    text: str,
    *,
    color: str = "#111111",
    fontsize: float = 8.8,
    weight: str = "regular",
) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        family="DejaVu Sans",
        bbox={
            "boxstyle": "round,pad=0.08",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.72,
        },
    )


def draw_state(
    ax: plt.Axes,
    cost_map: np.ndarray,
    state: dict,
    *,
    title: str,
    show_frontier: bool = True,
    settled_alpha: float = 0.36,
    chosen: tuple[float, int, int, int] | None = None,
    show_chosen_annotation: bool = True,
    just_absorbed: tuple[float, int, int, int] | None = None,
    meeting: tuple[int, int] | None = None,
    meeting_label: str | None = None,
) -> None:
    draw_cost_background(ax, cost_map)

    owner = state["owner"]
    settled = state["settled"]
    dist = state["dist"]
    frontier = frontier_mask(state)

    for cid, color in REGION_COLORS.items():
        ys, xs = np.where(settled & (owner == cid))
        for y, x in zip(ys.tolist(), xs.tolist()):
            add_cell_rect(ax, y, x, color, alpha=settled_alpha, linewidth=1.0)

    for cid, cells in SEED_CELLS.items():
        for y, x in cells:
            add_cell_rect(ax, y, x, REGION_COLORS[cid], alpha=0.86, linewidth=1.8)
            label_cell(ax, y, x, "0", color="white", fontsize=10.2, weight="bold")

    if show_frontier:
        fy, fx = np.where(frontier)
        for y, x in zip(fy.tolist(), fx.tolist()):
            cid = int(owner[y, x])
            edge = REGION_COLORS.get(cid, FRONTIER_COLOR)
            add_cell_rect(
                ax,
                y,
                x,
                FRONTIER_COLOR,
                alpha=0.18,
                linewidth=1.7,
                fill=True,
            )
            add_cell_rect(
                ax,
                y,
                x,
                edge,
                alpha=0.95,
                linewidth=1.9,
                fill=False,
                linestyle="--",
            )

    if just_absorbed is not None:
        _, y, x, cid = just_absorbed
        add_cell_rect(
            ax,
            y,
            x,
            REGION_COLORS[cid],
            alpha=0.70,
            linewidth=2.4,
            fill=True,
        )
        add_cell_rect(
            ax,
            y,
            x,
            SELECTED_COLOR,
            alpha=1.0,
            linewidth=2.5,
            fill=False,
        )

    if chosen is not None:
        d, y, x, _ = chosen
        add_cell_rect(
            ax,
            y,
            x,
            SELECTED_COLOR,
            alpha=1.0,
            linewidth=2.8,
            fill=False,
        )
        if show_chosen_annotation:
            arrow = FancyArrowPatch(
                (x + 2.0, y - 1.4),
                (x + 0.42, y - 0.36),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.9,
                color=SELECTED_COLOR,
                clip_on=False,
            )
            ax.add_patch(arrow)
            ax.text(
                x + 2.15,
                y - 1.56,
                f"min cost on frontier = {d:.1f}",
                ha="left",
                va="center",
                fontsize=8.2,
                color=SELECTED_COLOR,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": SELECTED_COLOR,
                    "linewidth": 0.9,
                    "alpha": 0.9,
                },
                clip_on=False,
            )

    if meeting is not None:
        my, mx = meeting
        add_cell_rect(ax, my, mx, MEETING_COLOR, alpha=0.84, linewidth=2.8)
        label_cell(
            ax,
            my,
            mx,
            f"{float(dist[my, mx]):.1f}",
            color="white",
            fontsize=9.8,
            weight="bold",
        )
        if meeting_label is not None:
            ax.text(
                0.02,
                0.98,
                meeting_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.0,
                color=MEETING_COLOR,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": MEETING_COLOR,
                    "linewidth": 0.9,
                    "alpha": 0.92,
                },
            )

    visible_frontier = frontier if show_frontier else np.zeros_like(frontier)
    label_mask = np.isfinite(dist) & (visible_frontier | settled)
    if meeting is not None:
        label_mask[meeting] = False

    ys, xs = np.where(label_mask)
    seed_cells = all_seed_cells()
    for y, x in zip(ys.tolist(), xs.tolist()):
        if (y, x) in seed_cells:
            continue
        cid = int(owner[y, x])
        text_color = REGION_COLORS.get(cid, "#111111") if frontier[y, x] else "#111111"
        label_cell(ax, y, x, f"{float(dist[y, x]):.1f}", color=text_color)

    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=9)


def make_initial_state(cost_map: np.ndarray) -> dict:
    """Panel 1 state: only seeds have accumulated cost 0."""
    state = init_search_state(cost_map)
    frontier = frontier_mask(state)
    state["dist"][frontier] = np.inf
    state["owner"][frontier] = 0
    state["heap"] = []
    return state


def add_legend(fig: plt.Figure, bounds: list[float]) -> None:
    """Add a compact legend to a figure."""
    legend_ax = fig.add_axes([0.16, 0.025, 0.68, 0.05])
    legend_ax.set_position(bounds)
    legend_ax.axis("off")
    legend_items = [
        ("red seed/region", REGION_COLORS[1]),
        ("green seed/region", REGION_COLORS[2]),
        ("frontier", FRONTIER_COLOR),
        ("selected next pixel", SELECTED_COLOR),
    ]
    x = 0.0
    for label, color in legend_items:
        legend_ax.add_patch(
            Rectangle(
                (x, 0.31),
                0.035,
                0.38,
                transform=legend_ax.transAxes,
                facecolor=color,
                edgecolor=color,
                alpha=0.75,
            )
        )
        legend_ax.text(
            x + 0.045,
            0.50,
            label,
            transform=legend_ax.transAxes,
            va="center",
            ha="left",
            fontsize=8.8,
        )
        x += 0.20


def build_panel_specs(cost_map: np.ndarray) -> list[dict]:
    """Build the panel states and drawing options."""
    initial_state = make_initial_state(cost_map)
    initial_frontier_state = init_search_state(cost_map)
    first_choice = peek_next_frontier(initial_frontier_state)
    first_expansion_state, first_absorbed = capture_first_expansion(cost_map)
    complete_expansion_state = capture_complete_expansion(cost_map)
    half_expansion_state, half_choice = capture_half_expansion(
        cost_map,
        complete_expansion_state,
    )

    return [
        {
            "slug": "panel_1_initial",
            "state": initial_state,
            "title": "Panel 1 - Initial",
        },
        {
            "slug": "panel_2_initial_frontier",
            "state": initial_frontier_state,
            "title": "Panel 2 - Initial Frontier",
            "chosen": first_choice,
            "show_chosen_annotation": False,
        },
        {
            "slug": "panel_3_first_expansion",
            "state": first_expansion_state,
            "title": "Panel 3 - First Expansion",
            "just_absorbed": first_absorbed,
            "show_frontier": False,
        },
        {
            "slug": "panel_4_half_expanded",
            "state": half_expansion_state,
            "title": "Panel 4 - Half Expanded",
            "chosen": half_choice,
            "show_chosen_annotation": False,
            "settled_alpha": 1.0,
        },
        {
            "slug": "panel_5_fully_expanded",
            "state": complete_expansion_state,
            "title": "Panel 5 - Fully Expanded",
            "show_frontier": False,
            "settled_alpha": 1.0,
        },
    ]


def draw_panel_from_spec(ax: plt.Axes, cost_map: np.ndarray, spec: dict) -> None:
    """Draw one panel from a panel spec."""
    draw_state(
        ax,
        cost_map,
        spec["state"],
        title=spec["title"],
        show_frontier=spec.get("show_frontier", True),
        settled_alpha=spec.get("settled_alpha", 0.36),
        chosen=spec.get("chosen"),
        show_chosen_annotation=spec.get("show_chosen_annotation", True),
        just_absorbed=spec.get("just_absorbed"),
        meeting=spec.get("meeting"),
        meeting_label=spec.get("meeting_label"),
    )


def save_individual_panels(
    output_path: Path,
    cost_map: np.ndarray,
    panel_specs: list[dict],
) -> list[Path]:
    """Save each panel as its own PNG beside the combined output."""
    saved_paths = []
    for spec in panel_specs:
        panel_path = output_path.with_name(f"{output_path.stem}_{spec['slug']}.png")
        fig, ax = plt.subplots(1, 1, figsize=(5.2, 5.55), constrained_layout=True)
        fig.patch.set_facecolor("white")
        draw_panel_from_spec(ax, cost_map, spec)
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(panel_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(panel_path)
    return saved_paths


def render(output_path: Path) -> list[Path]:
    cost_map = build_cost_map()
    panel_specs = build_panel_specs(cost_map)

    fig, axes = plt.subplots(
        1,
        len(panel_specs),
        figsize=(4.5 * len(panel_specs), 5.25),
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "10x10 multi-source Dijkstra decision grid (bright = low traversal cost / fiber)",
        fontsize=13.5,
        fontweight="bold",
    )

    for ax, spec in zip(axes, panel_specs):
        draw_panel_from_spec(ax, cost_map, spec)

    add_legend(fig, [0.16, 0.025, 0.68, 0.05])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    panel_paths = save_individual_panels(output_path, cost_map, panel_specs)
    return [output_path, *panel_paths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a 10x10 multi-source Dijkstra step diagram."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=OUTPUT_DIR / "viz_region_grow_step_grid.png",
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_paths = render(args.output)
    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()

"""
Generate topology linking dataset for MLP connection classifier.

For each sample in the data directory:
1. Apply SkinAnalysisPipeline preprocessing (uses full-image mask when mask.png is absent)
2. Build skeleton/seed graph from processed annotation
3. Compute A* paths and Hausdorff sum for all cross-component connection pairs
4. Save results as CSV (features) and pkl (paths)

Usage:
    uv run python tools/generate_linking_dataset.py \\
        --data-dir data_0320 \\
        --output-dir data/datasets

    # Process specific samples only
    uv run python tools/generate_linking_dataset.py \\
        --data-dir data_0320 \\
        --output-dir data/datasets \\
        --sample-ids S1585-2_a S1585-2_b
"""

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import pandas as pd
import skimage as ski
from scipy.spatial import KDTree
from skimage import morphology
from skimage.measure import label, regionprops
from skan import Skeleton, summarize
from skan.csr import skeleton_to_nx

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from neural_reconstruction.core.preprocessing.pipeline import SkinAnalysisPipeline

# ---------------------------------------------------------------------------
# Default parameters (match notebook defaults)
# ---------------------------------------------------------------------------
SEGMENT_LENGTH = 3
SEARCH_RADIUS = 50
PATH_FINDING_BBOX_PADDING = 10
HAUSDORFF_INF_SENTINEL = 10**10
HAUSDORFF_IN_LABEL_TOLERANCE = (
    5  # if partially outside GT but hausdorff <= this, keep real value
)


# ---------------------------------------------------------------------------
# Skeleton / seed graph helpers (ported directly from notebook)
# ---------------------------------------------------------------------------


def build_skeleton_graph(binary: np.ndarray) -> nx.MultiGraph:
    """Skeletonize binary annotation and return a cleaned skeleton graph."""
    skeleton = morphology.skeletonize(binary).astype(np.uint8)
    skel_obj = Skeleton(skeleton, keep_images=False)
    summary = summarize(skel_obj, separator="-")
    skeleton_graph = skeleton_to_nx(skel_obj, summary=summary)

    # Filter short terminal edges
    filtered = nx.MultiGraph()
    for u, v, data in skeleton_graph.edges(data=True):
        path = data["path"]
        if len(path) > 2 or (
            skeleton_graph.degree(u) != 1 and skeleton_graph.degree(v) != 1
        ):
            filtered.add_edge(u, v, **data)

    # Relabel integer nodes → (y, x) coordinate tuples
    mapping = {i: tuple(skel_obj.coordinates[i].astype(int)) for i in filtered.nodes()}
    filtered = nx.relabel_nodes(filtered, mapping)

    # Merge degree-2 intermediate nodes
    middle_points = [
        p for p in filtered.nodes() if len(list(filtered.neighbors(p))) == 2
    ]
    for mp in middle_points:
        neighbors = list(filtered.neighbors(mp))
        if len(neighbors) != 2:
            continue
        u, v = neighbors
        if not filtered.has_edge(u, mp) or not filtered.has_edge(mp, v):
            continue
        path1 = filtered[u][mp][0]["path"]
        path2 = filtered[mp][v][0]["path"]
        u_y, u_x = u
        mp_y, mp_x = mp

        result_path = []
        if tuple(path1[-1]) == (mp_y, mp_x) and tuple(path1[0]) == (u_y, u_x):
            result_path.extend(path1)
        else:
            result_path.extend(path1[::-1])

        if tuple(path2[0]) == (mp_y, mp_x) and tuple(path2[-1]) == v:
            result_path.extend(path2[1:])
        else:
            result_path.extend(path2[-2::-1])

        filtered.remove_node(mp)
        filtered.add_edge(u, v, path=result_path)

    return filtered


def fill_missing_components(
    filtered_graph: nx.MultiGraph,
    binary: np.ndarray,
    orig_green: np.ndarray,
) -> nx.MultiGraph:
    """Add representative nodes for connected components missing from the graph."""
    components = label(binary, connectivity=2)
    regions = regionprops(components)

    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        bbox_nodes = [
            n
            for n in filtered_graph.nodes()
            if min_row <= n[0] < max_row and min_col <= n[1] < max_col
        ]
        if bbox_nodes:
            continue

        # Pick brightest pixel in the component bounding box
        brightest_pixel = None
        brightest_value = -1
        for r in range(min_row, max_row):
            for c in range(min_col, max_col):
                if orig_green[r, c] > brightest_value:
                    brightest_value = orig_green[r, c]
                    brightest_pixel = (r, c)
        if brightest_pixel is not None:
            filtered_graph.add_node(brightest_pixel)

    return filtered_graph


def build_seed_graph(
    filtered_graph: nx.MultiGraph, segment_length: int
) -> nx.MultiGraph:
    """Split skeleton edges into segments of ~segment_length pixels."""
    seed_graph = nx.MultiGraph()

    for u in filtered_graph.nodes():
        seed_graph.add_node(u)

    for u, v, data in filtered_graph.edges(data=True):
        path = data["path"]
        corrected = (
            path[:] if tuple(path[0]) == u and tuple(path[-1]) == v else path[::-1]
        )
        path_arr = np.array(corrected)
        diffs = np.diff(path_arr, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        cumulative = np.concatenate(([0], np.cumsum(distances)))
        path_length = cumulative[-1]
        num_segments = int(path_length // segment_length)

        if num_segments <= 0:
            seed_graph.add_edge(u, v, path=path)
            continue

        last_index = 0
        for i in range(num_segments):
            target_dist = (i + 1) * path_length / num_segments
            seg_end = 0
            for idx, cd in enumerate(cumulative[last_index:]):
                if cd >= target_dist:
                    seg_end = idx + last_index
                    break
            seg_path = corrected[last_index : seg_end + 1]
            if len(seg_path) > 0:
                seed_graph.add_edge(
                    tuple(seg_path[0]), tuple(seg_path[-1]), path=seg_path
                )
            last_index = seg_end

        if last_index < len(corrected) - 1:
            final_seg = corrected[last_index:]
            seed_graph.add_edge(
                tuple(final_seg[0]), tuple(final_seg[-1]), path=final_seg
            )

    return seed_graph


# ---------------------------------------------------------------------------
# Hausdorff / path helpers
# ---------------------------------------------------------------------------


def compute_path_hausdorff_sum(path: list, gt_kdtree: KDTree) -> float:
    """Sum of min distances from each path point to the nearest GT pixel."""
    path_pts = np.array(path)
    dists, _ = gt_kdtree.query(path_pts)
    return float(np.sum(dists))


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------


def process_sample(
    sample_id: str,
    sample_dir: Path,
    segment_length: int,
    search_radius: int,
    bbox_padding: int,
    verbose: bool,
) -> list[dict]:
    """Process a single sample and return list of connection pair records."""
    if verbose:
        print(f"  Loading images from {sample_dir}")

    # Load images
    orig_bgr = cv2.imread(str(sample_dir / "image.png"), cv2.IMREAD_COLOR)
    if orig_bgr is None:
        raise FileNotFoundError(f"Cannot read image.png in {sample_dir}")

    # Extract green channel (strongest nerve fiber signal); SkinAnalysisPipeline expects grayscale
    orig_green = cv2.split(orig_bgr)[1]  # BGR → green channel (index 1)
    annotation_img = cv2.imread(
        str(sample_dir / "annotation.png"), cv2.IMREAD_GRAYSCALE
    )
    label_img = cv2.imread(str(sample_dir / "label.png"), cv2.IMREAD_GRAYSCALE)

    if annotation_img is None:
        raise FileNotFoundError(f"Cannot read annotation.png in {sample_dir}")
    if label_img is None:
        raise FileNotFoundError(f"Cannot read label.png in {sample_dir}")

    h, w = annotation_img.shape

    # Build full-image mask (no mask.png in this dataset)
    mask_path = sample_dir / "mask.png"
    if mask_path.exists():
        epidermis_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        use_full_roi = False
        if verbose:
            print("  Using mask.png")
    else:
        epidermis_mask = np.full((h, w), 255, dtype=np.uint8)
        # With a full-image mask the dermis ROI mask is empty; use full ROI for thresholding
        use_full_roi = True
        if verbose:
            print("  No mask.png found — using full-image mask (use_full_roi=True)")

    # Apply SkinAnalysisPipeline preprocessing
    if verbose:
        print("  Running SkinAnalysisPipeline...")
    sample_pipeline = SkinAnalysisPipeline()
    sample_pipeline.config.threshold.use_full_roi = use_full_roi
    final_label, roi_image = sample_pipeline.run(
        annotation_img, epidermis_mask, orig_green
    )

    # Green channel is already extracted above (orig_green)

    # Build skeleton from processed annotation
    binary = (final_label > 0).astype(np.uint8)
    if binary.sum() == 0:
        if verbose:
            print(
                f"  WARNING: empty annotation after preprocessing — skipping {sample_id}"
            )
        return []

    if verbose:
        print("  Building skeleton graph...")
    skeleton_graph = build_skeleton_graph(binary)
    skeleton_graph = fill_missing_components(skeleton_graph, binary, roi_image)
    seed_graph = build_seed_graph(skeleton_graph, segment_length)

    topology_points = np.array(list(seed_graph.nodes()))
    kdtree = KDTree(topology_points)

    # Build cost map (dark pixels = high cost)
    cost_map = ((255 - roi_image.astype(np.float64)) / 255.0) ** 8
    cost_h, cost_w = cost_map.shape

    seed_map = np.zeros((cost_h, cost_w), dtype=np.uint8)
    for p in topology_points:
        seed_map[p[0], p[1]] = 1

    # Build GT KDTree for Hausdorff
    gt_pixels = np.column_stack(np.where(label_img > 0))
    if len(gt_pixels) == 0:
        if verbose:
            print(f"  WARNING: empty GT label — skipping {sample_id}")
        return []
    gt_kdtree = KDTree(gt_pixels)
    label_binary = label_img > 0

    # Connected-component map for cross-component filtering
    annotation_components = label(binary, connectivity=2)

    if verbose:
        print(
            f"  Seeds: {len(topology_points)}, "
            f"GT pixels: {len(gt_pixels)}, "
            f"running A*..."
        )

    # A* path finding
    all_samples: list[dict] = []
    pair_id = 0

    for u_idx in range(len(topology_points)):
        u = topology_points[u_idx]

        neighbor_indices = kdtree.query_ball_point(u, r=search_radius)
        targets = [
            topology_points[v_idx]
            for v_idx in neighbor_indices
            if tuple(topology_points[v_idx]) != tuple(u)
        ]

        current_comp = annotation_components[u[0], u[1]]
        targets = [
            t for t in targets if annotation_components[t[0], t[1]] != current_comp
        ]

        if not targets:
            continue

        all_points = [u] + targets
        all_y = [p[0] for p in all_points]
        all_x = [p[1] for p in all_points]

        min_y = max(0, min(all_y) - bbox_padding)
        max_y = min(cost_h - 1, max(all_y) + bbox_padding)
        min_x = max(0, min(all_x) - bbox_padding)
        max_x = min(cost_w - 1, max(all_x) + bbox_padding)

        cropped_cost = cost_map[min_y : max_y + 1, min_x : max_x + 1]
        local_pts = [(p[0] - min_y, p[1] - min_x) for p in all_points]

        mcp = ski.graph.MCP_Geometric(cropped_cost, fully_connected=True)
        cumulative_costs, _ = mcp.find_costs(starts=local_pts[:1], ends=local_pts[1:])

        for target_local in local_pts[1:]:
            if np.isinf(cumulative_costs[target_local]):
                continue

            path = mcp.traceback(target_local)
            cost = cumulative_costs[target_local]

            global_path = [(p[0] + min_y, p[1] + min_x) for p in path]
            global_start = (int(u[0]), int(u[1]))
            global_target = (target_local[0] + min_y, target_local[1] + min_x)

            # Skip if path passes through another seed point
            middle_pts = np.array(path[1:-1])
            if len(middle_pts) > 0:
                mid_global = middle_pts + np.array([min_y, min_x])
                if np.any(seed_map[mid_global[:, 0], mid_global[:, 1]]):
                    continue

            distance = float(
                np.linalg.norm(np.array(global_start) - np.array(global_target))
            )
            path_length = len(global_path)
            normalized_cost = cost / path_length if path_length > 0 else cost

            # Hausdorff sum; sentinel if path strays outside GT
            path_pts = np.array(global_path)
            path_in_label = label_binary[path_pts[:, 0], path_pts[:, 1]]
            hausdorff_sum = compute_path_hausdorff_sum(global_path, gt_kdtree)

            if (
                not np.all(path_in_label)
                and hausdorff_sum > HAUSDORFF_IN_LABEL_TOLERANCE
            ):
                hausdorff_sum = HAUSDORFF_INF_SENTINEL

            all_samples.append(
                {
                    "pair_id": pair_id,
                    "image_id": sample_id,
                    "seed1_y": global_start[0],
                    "seed1_x": global_start[1],
                    "seed2_y": global_target[0],
                    "seed2_x": global_target[1],
                    "distance": distance,
                    "path_length": path_length,
                    "path_cost": cost,
                    "normalized_cost": normalized_cost,
                    "hausdorff_sum": hausdorff_sum,
                    "path": global_path,
                }
            )
            pair_id += 1

        if verbose and (u_idx + 1) % 200 == 0:
            print(
                f"    {u_idx + 1}/{len(topology_points)} seeds, "
                f"{len(all_samples)} pairs so far"
            )

    return all_samples


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def discover_samples(data_dir: Path) -> list[tuple[str, Path]]:
    """Return (sample_id, path) for directories containing required files."""
    required = {"image.png", "annotation.png", "label.png"}
    samples = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        files = {f.name for f in d.iterdir()}
        if required.issubset(files):
            samples.append((d.name, d))
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate topology linking dataset with SkinAnalysisPipeline preprocessing"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root data directory containing sample subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/datasets"),
        help="Output directory for CSV and pkl files (default: data/datasets)",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="Process only these sample IDs (default: all)",
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=SEGMENT_LENGTH,
        help=f"Seed spacing along skeleton in pixels (default: {SEGMENT_LENGTH})",
    )
    parser.add_argument(
        "--search-radius",
        type=int,
        default=SEARCH_RADIUS,
        help=f"A* search radius in pixels (default: {SEARCH_RADIUS})",
    )
    parser.add_argument(
        "--bbox-padding",
        type=int,
        default=PATH_FINDING_BBOX_PADDING,
        help=f"Bounding box padding for A* crop (default: {PATH_FINDING_BBOX_PADDING})",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose CSV already exists in output-dir",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-sample progress"
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover samples
    all_found = discover_samples(data_dir)
    if not all_found:
        print(f"No valid samples found in {data_dir}")
        sys.exit(1)

    if args.sample_ids:
        samples = [(sid, p) for sid, p in all_found if sid in args.sample_ids]
        missing = set(args.sample_ids) - {sid for sid, _ in samples}
        if missing:
            print(f"WARNING: sample IDs not found: {sorted(missing)}")
    else:
        samples = all_found

    print(f"Found {len(samples)} sample(s) to process")
    print(f"Output directory: {output_dir}")
    print(
        f"Parameters: segment_length={args.segment_length}, "
        f"search_radius={args.search_radius}, bbox_padding={args.bbox_padding}"
    )
    print()

    total_pairs = 0
    processed = 0
    skipped = 0

    for sample_id, sample_dir in samples:
        csv_path = output_dir / f"dataset_{sample_id}.csv"
        pkl_path = output_dir / f"dataset_{sample_id}_paths.pkl"

        if args.skip_existing and csv_path.exists():
            print(f"[SKIP] {sample_id} (CSV already exists)")
            skipped += 1
            continue

        print(f"[{processed + 1}/{len(samples)}] Processing {sample_id} ...")

        try:
            records = process_sample(
                sample_id=sample_id,
                sample_dir=sample_dir,
                segment_length=args.segment_length,
                search_radius=args.search_radius,
                bbox_padding=args.bbox_padding,
                verbose=args.verbose,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        if not records:
            print(f"  No pairs generated — skipping output for {sample_id}")
            processed += 1
            continue

        # Save CSV (without path column)
        csv_rows = [{k: v for k, v in r.items() if k != "path"} for r in records]
        df = pd.DataFrame(csv_rows)
        df.to_csv(csv_path, index=False)

        # Save pkl (pair_id → path)
        paths_dict = {r["pair_id"]: r["path"] for r in records}
        with open(pkl_path, "wb") as f:
            pickle.dump(paths_dict, f)

        n = len(records)
        total_pairs += n
        n_pos = int((df["hausdorff_sum"] <= 10).sum())
        n_neg = n - n_pos
        print(
            f"  {n} pairs saved  (hausdorff<=10: {n_pos}, >10: {n_neg})  "
            f"-> {csv_path.name}"
        )
        processed += 1

    print()
    print("=" * 60)
    print(f"Done. Processed: {processed}, Skipped: {skipped}")
    print(f"Total connection pairs generated: {total_pairs}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()

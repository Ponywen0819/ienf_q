"""
GT Count Evaluation Script

Uses the current crossing detection pipeline on GT graphs (built from label.png)
and compares the predicted valid_count against count.json ground truth.

This isolates the counting logic from the reconstruction: if this script gives
poor results, the problem is in the crossing detection method itself, not the
reconstruction quality.

Usage:
    python tools/evaluate_gt_count.py \
        --data-dir data/ \
        --output-dir output/gt_count_eval

    python tools/evaluate_gt_count.py \
        --data-dir data/ \
        --output-dir output/gt_count_eval \
        --sample-ids S1585-2_a S1585-2_b \
        --verbose
"""

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from neural_reconstruction.dataset import DatasetLoader, SampleFiles
from neural_reconstruction.core.topology import TopologyBuilder
from neural_reconstruction.core.crosses_detection.segment_detector import (
    SegmentDetector,
)
from neural_reconstruction.core.crosses_detection.region_labeler import RegionLabeler
from neural_reconstruction.core.crosses_detection.crossing_counter import (
    CrossingCounter,
)


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class SampleCountResult:
    sample_id: str
    status: str  # success | skipped | failed
    gt_count: Optional[int] = None
    pred_count: Optional[int] = None
    count_error: Optional[float] = None  # |pred - gt|
    num_nodes: Optional[int] = None
    num_edges: Optional[int] = None
    total_segments: Optional[int] = None
    segments_with_crossing: Optional[int] = None
    error_message: Optional[str] = None


# ============================================================================
# Crossing analysis (mirrors linker._run_crossing_analysis)
# ============================================================================


def _path_length(data: dict, u, v) -> int:
    """Number of pixel steps along an edge's path."""
    path = data.get("path", [u, v])
    return len(path) - 1


def run_crossing_analysis(
    graph: nx.Graph,
    mask: np.ndarray,
    min_stub_length: int = 5,
    min_region_length: float = 5.0,
) -> Dict:
    """
    Run the same crossing analysis pipeline as PureMstLinker._run_crossing_analysis.

    Steps:
      1. Detect segments
      2. Prune short dangling stubs (total path length < min_stub_length px)
      3. Re-detect segments on pruned graph
      4. Label regions (epidermis / dermis) per node and crossing per edge
      5. Count effective crossings

    Args:
        graph: Topology graph (nodes are (y, x) tuples, edges have ``path``).
        mask: Epidermis mask (H, W), values >127 = epidermis.
        min_stub_length: Remove stub segments shorter than this (pixels).
        min_region_length: Minimum path length in each region for a valid crossing.

    Returns:
        Dict from CrossingCounter.count_effective_crossings.
    """
    segment_detector = SegmentDetector()
    region_labeler = RegionLabeler()
    crossing_counter = CrossingCounter()

    # Step 1: detect segments
    segmented_graph = segment_detector.detect_segments(graph)

    # Step 2: prune short stubs (identical logic to linker)
    seg_edges: Dict[int, list] = defaultdict(list)
    for u, v, data in segmented_graph.edges(data=True):
        seg_id = data.get("segment_id")
        if seg_id is not None:
            seg_edges[seg_id].append((u, v, data))

    edges_to_remove = []
    for seg_id, edges in seg_edges.items():
        boundary_nodes = set()
        for u, v, _ in edges:
            if segmented_graph.nodes[u].get("node_type") in ("endpoint", "branchpoint"):
                boundary_nodes.add(u)
            if segmented_graph.nodes[v].get("node_type") in ("endpoint", "branchpoint"):
                boundary_nodes.add(v)

        has_endpoint = any(
            segmented_graph.nodes[n].get("node_type") == "endpoint"
            for n in boundary_nodes
        )
        if not has_endpoint:
            continue

        total_length = sum(_path_length(d, u, v) for u, v, d in edges)
        if total_length < min_stub_length:
            edges_to_remove.extend((u, v) for u, v, _ in edges)

    segmented_graph.remove_edges_from(edges_to_remove)
    segmented_graph.remove_nodes_from(list(nx.isolates(segmented_graph)))

    # Step 3: re-detect segments after pruning
    segmented_graph = segment_detector.detect_segments(segmented_graph)

    # Step 4: label regions
    labeled_graph, _ = region_labeler.label_topology(segmented_graph, mask)

    # Step 5: count
    result = crossing_counter.count_effective_crossings(
        labeled_graph,
        epidermis_mask=mask,
        min_region_length=min_region_length,
    )

    return result


# ============================================================================
# Per-sample evaluation
# ============================================================================


def evaluate_sample(
    sample: SampleFiles,
    gt_counts: Dict[str, int],
    topology_builder: TopologyBuilder,
    logger: logging.Logger,
) -> SampleCountResult:
    """Evaluate a single sample's GT count vs annotated count."""
    has_label = sample.label_path is not None and sample.label_path.exists()
    has_count = sample.sample_id in gt_counts

    if not has_label:
        return SampleCountResult(
            sample_id=sample.sample_id,
            status="skipped",
            error_message="missing_label",
        )
    if not has_count:
        return SampleCountResult(
            sample_id=sample.sample_id,
            status="skipped",
            error_message="missing_count_json_entry",
        )

    is_complete, reason = sample.is_complete()
    if not is_complete:
        return SampleCountResult(
            sample_id=sample.sample_id,
            status="skipped",
            error_message=reason,
        )

    try:
        gt_label = np.array(Image.open(sample.label_path).convert("L"))
        mask = np.array(Image.open(sample.mask_path).convert("L"))

        # Build GT topology graph
        gt_graph = topology_builder.build_seed_graph(gt_label)

        if gt_graph.number_of_nodes() == 0:
            logger.warning(f"{sample.sample_id}: GT graph is empty after skeletonization")
            return SampleCountResult(
                sample_id=sample.sample_id,
                status="failed",
                error_message="empty_gt_graph",
            )

        # Run crossing analysis
        crossing_result = run_crossing_analysis(gt_graph, mask)

        pred_count = crossing_result["effective_crossing_count"]
        gt_count = gt_counts[sample.sample_id]
        count_error = float(abs(pred_count - gt_count))

        logger.debug(
            f"{sample.sample_id}: pred={pred_count}, gt={gt_count}, "
            f"error={count_error:.0f}"
        )

        return SampleCountResult(
            sample_id=sample.sample_id,
            status="success",
            gt_count=gt_count,
            pred_count=pred_count,
            count_error=count_error,
            num_nodes=gt_graph.number_of_nodes(),
            num_edges=gt_graph.number_of_edges(),
            total_segments=crossing_result["total_segments"],
            segments_with_crossing=crossing_result["segments_with_crossing"],
        )

    except Exception as e:
        logger.error(f"{sample.sample_id}: failed — {e}", exc_info=True)
        return SampleCountResult(
            sample_id=sample.sample_id,
            status="failed",
            error_message=str(e),
        )


# ============================================================================
# Summary + report
# ============================================================================


def compute_summary(results: List[SampleCountResult]) -> Dict:
    """Compute aggregate statistics from all sample results."""
    success = [r for r in results if r.status == "success"]
    pairs = [
        (r.pred_count, r.gt_count)
        for r in success
        if r.pred_count is not None and r.gt_count is not None
    ]

    summary = {
        "total": len(results),
        "success": len(success),
        "skipped": sum(1 for r in results if r.status == "skipped"),
        "failed": sum(1 for r in results if r.status == "failed"),
        "count_n": len(pairs),
    }

    if not pairs:
        return summary

    preds, gts = zip(*pairs)
    errors = np.abs(np.array(preds, float) - np.array(gts, float))

    summary.update(
        {
            "mae_mean": float(np.mean(errors)),
            "mae_median": float(np.median(errors)),
            "mae_std": float(np.std(errors)),
            "mae_min": float(np.min(errors)),
            "mae_max": float(np.max(errors)),
        }
    )

    if len(pairs) >= 2:
        pr, pp = pearsonr(preds, gts)
        sr, sp = spearmanr(preds, gts)
        summary.update(
            {
                "pearson_r": float(pr),
                "pearson_p": float(pp),
                "spearman_r": float(sr),
                "spearman_p": float(sp),
            }
        )

    return summary


def save_csv(results: List[SampleCountResult], path: Path):
    fieldnames = [
        "sample_id",
        "status",
        "gt_count",
        "pred_count",
        "count_error",
        "num_nodes",
        "num_edges",
        "total_segments",
        "segments_with_crossing",
        "error_message",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def print_summary(summary: Dict):
    print("\n" + "=" * 60)
    print("GT Count Evaluation Summary")
    print("=" * 60)
    print(f"Total samples : {summary['total']}")
    print(f"Success       : {summary['success']}")
    print(f"Skipped       : {summary['skipped']}")
    print(f"Failed        : {summary['failed']}")

    n = summary.get("count_n", 0)
    if n == 0:
        print("Count metrics : no valid pairs")
        print("=" * 60 + "\n")
        return

    print(f"\nCount Metrics  (n={n}):")
    print(f"  MAE mean    : {summary['mae_mean']:.3f}")
    print(f"  MAE median  : {summary['mae_median']:.3f}")
    print(f"  MAE std     : {summary['mae_std']:.3f}")
    print(f"  MAE min/max : {summary['mae_min']:.1f} / {summary['mae_max']:.1f}")

    if "pearson_r" in summary:
        print(
            f"\n  Pearson  r  : {summary['pearson_r']:.4f}"
            f"  (p={summary['pearson_p']:.4f})"
        )
        print(
            f"  Spearman r  : {summary['spearman_r']:.4f}"
            f"  (p={summary['spearman_p']:.4f})"
        )

    print("=" * 60 + "\n")


# ============================================================================
# CLI
# ============================================================================


def setup_logging(output_dir: Path, verbose: bool):
    log_path = output_dir / "gt_count_eval.log"
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate crossing detection on GT graphs and compare with count.json. "
            "Useful for diagnosing whether errors come from reconstruction or counting."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset root directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for reports")
    parser.add_argument(
        "--sample-ids", nargs="+", help="Specific sample IDs to evaluate (default: all)"
    )
    parser.add_argument(
        "--segment-length",
        type=float,
        default=5.0,
        help="Seed spacing for TopologyBuilder (default: 5.0)",
    )
    parser.add_argument(
        "--min-region-length",
        type=float,
        default=5.0,
        help="Min path length in each region for a valid crossing (default: 5.0)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(args.output_dir, args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("GT Count Evaluation")
    logger.info("=" * 60)
    logger.info(f"Data dir        : {args.data_dir}")
    logger.info(f"Output dir      : {args.output_dir}")
    logger.info(f"segment_length  : {args.segment_length}")
    logger.info(f"min_region_len  : {args.min_region_length}")

    # Load count.json
    count_path = args.data_dir / "count.json"
    if not count_path.exists():
        logger.error(f"count.json not found at {count_path}")
        return
    with open(count_path, encoding="utf-8") as f:
        gt_counts: Dict[str, int] = json.load(f)
    logger.info(f"Loaded {len(gt_counts)} entries from count.json")

    # Load samples
    loader = DatasetLoader(args.data_dir)
    samples = loader.load_samples(args.sample_ids)

    topology_builder = TopologyBuilder(segment_length=args.segment_length)

    # Evaluate
    results: List[SampleCountResult] = []
    for sample in tqdm(samples, desc="Evaluating"):
        result = evaluate_sample(sample, gt_counts, topology_builder, logger)
        results.append(result)

    # Summary
    summary = compute_summary(results)

    # Save reports
    json_path = args.output_dir / "gt_count_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "config": {
                    "segment_length": args.segment_length,
                    "min_region_length": args.min_region_length,
                },
                "samples": {r.sample_id: asdict(r) for r in results},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"JSON report saved: {json_path}")

    csv_path = args.output_dir / "gt_count_results.csv"
    save_csv(results, csv_path)
    logger.info(f"CSV report saved: {csv_path}")

    print_summary(summary)


if __name__ == "__main__":
    main()

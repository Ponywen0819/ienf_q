#!/usr/bin/env python3
"""
Pipeline Visualization Tool

visualizes the full neural reconstruction pipeline by overlaying the reconstructed
topology on the green channel of the input image.

Usage:
    python tools/visualize_pipeline.py -i <input_image> -l <label_image> -m <mask_image> -o <output_image>
"""

import sys
import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
import networkx as nx

# Add src to python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from neural_reconstruction.ui.main_pipeline import NeuralReconstructionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def draw_topology(
    image: np.ndarray, graph: nx.Graph, color: tuple = (0, 0, 255), thickness: int = 1
):
    """Draw the topology graph on the image."""

    # Check if image is colored, if grayscale convert to RGB
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()

    # Draw edges
    for u, v, data in graph.edges(data=True):
        # Nodes are expected to be (y, x) tuples from global topology
        y1, x1 = u
        y2, x2 = v

        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))

        # If edge data has 'path', draw the full path
        if "path" in data and data["path"]:
            path_points = data["path"]
            # Draw path segments
            for i in range(len(path_points) - 1):
                py1, px1 = path_points[i]
                py2, px2 = path_points[i + 1]
                p1 = (int(px1), int(py1))
                p2 = (int(px2), int(py2))
                cv2.line(vis_image, p1, p2, color, thickness)
        else:
            # Draw straight line
            cv2.line(vis_image, pt1, pt2, color, thickness)

    # Draw nodes
    for node in graph.nodes():
        y, x = node
        pt = (int(x), int(y))
        cv2.circle(vis_image, pt, 1, (255, 0, 0), -1)  # Draw nodes in blue

    return vis_image


def main():
    parser = argparse.ArgumentParser(
        description="Visualize neural reconstruction pipeline."
    )
    parser.add_argument("-i", "--image", required=True, help="Input raw image path")
    parser.add_argument("-l", "--label", required=True, help="Input label image path")
    parser.add_argument("-m", "--mask", required=True, help="Input mask image path")
    parser.add_argument("-o", "--output", required=True, help="Output image path")

    args = parser.parse_args()

    # Check inputs
    if not Path(args.image).exists():
        logger.error(f"Image not found: {args.image}")
        return
    if not Path(args.label).exists():
        logger.error(f"Label not found: {args.label}")
        return
    if not Path(args.mask).exists():
        logger.error(f"Mask not found: {args.mask}")
        return

    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = NeuralReconstructionPipeline(
        preprocessing_config={
            "morphology": {
                "closing_kernel": 0,
                "opening_kernel": 3,
            },
            "mask": {
                "dilate_offset": 50,
            },
            "background": {
                "method": "rolling_ball",
                "radius": 2,
                "light_background": True,
            },
            "threshold": {"method": "binary", "use_full_roi": False},
            "normalization": {
                "enabled": True,
            },
        },
        reconstruction_config={
            "connectivity": 4,
            "min_area": 0,
            "segment_length": 5.0,
            "min_edge_length": 0,
            "prune_threshold": 5.0,
            "spacing": 0,
            "search_radius": 20.0,
            "max_cost_threshold": 0.98,
            "intensity_weight": 1,
            "shape_weight": 0,
        },
    )

    # Run pipeline
    logger.info("Running pipeline...")
    try:
        result = pipeline.run_from_files(
            label_path=args.label, mask_path=args.mask, image_path=args.image
        )
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        import traceback

        traceback.print_exc()
        return

    logger.info(
        f"Reconstruction finished. Nodes: {result.num_nodes}, Edges: {result.num_edges}"
    )

    # Load original image and extract green channel for background
    orig_img = cv2.imread(args.image)
    if orig_img is None:
        logger.error("Failed to re-read original image for visualization.")
        return

    if len(orig_img.shape) == 3:
        green_channel = orig_img[:, :, 1]
    else:
        # Assuming grayscale is the channel we want if only one exists
        green_channel = orig_img

    # Create visualization
    logger.info("Creating visualization...")
    vis_result = draw_topology(green_channel, result.mst_forest)

    # Save result
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_result)
    logger.info(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    main()

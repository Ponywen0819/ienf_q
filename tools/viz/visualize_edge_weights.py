#!/usr/bin/env python3
"""
Edge Weight Visualization Tool

Visualizes the reconstructed neural network by displaying edges color-coded
by their weight values using the viridis colormap. Does not display nodes.

Usage:
    python tools/visualize_edge_weights.py -i <input_image> -l <label_image> -m <mask_image> -o <output_image>
"""

import sys
import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

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


def draw_edges_by_weight(
    image: np.ndarray,
    graph: nx.Graph,
    thickness: int = 2,
    colormap: str = "viridis",
):
    """
    Draw edges on the image, color-coded by their weight values.

    Args:
        image: Background image (grayscale or BGR)
        graph: NetworkX graph with edges containing 'weight' attribute
        thickness: Line thickness for drawing edges
        colormap: Matplotlib colormap name (default: 'viridis')

    Returns:
        vis_image: Image with color-coded edges overlaid
    """
    # Check if image is colored, if grayscale convert to BGR
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()

    # Check if graph has edges
    if graph.number_of_edges() == 0:
        logger.warning("Graph has no edges to visualize.")
        return vis_image

    # Extract all edge weights
    weights = []
    for u, v, data in graph.edges(data=True):
        weight = data.get("weight", None)
        if weight is not None:
            weights.append(weight)

    if not weights:
        logger.warning("No edges have 'weight' attribute. Using default color.")
        default_color = (255, 255, 0)  # Yellow
        # Draw edges with default color
        for u, v, data in graph.edges(data=True):
            _draw_single_edge(vis_image, u, v, data, default_color, thickness)
        return vis_image

    # Normalize weights to [0, 1]
    min_weight = min(weights)
    max_weight = max(weights)

    logger.info(
        f"Edge weights - Min: {min_weight:.2f}, Max: {max_weight:.2f}, Mean: {np.mean(weights):.2f}"
    )

    # Get colormap
    cmap = plt.get_cmap(colormap)

    # Handle case where all weights are the same
    if max_weight == min_weight:
        logger.info("All edge weights are identical. Using middle colormap color.")
        norm_func = lambda x: 0.5  # Use middle of colormap
    else:
        # Invert normalization so lower weights (better quality) get higher values (brighter colors)
        normalizer = Normalize(vmin=min_weight, vmax=max_weight)
        # Create inverted mapper
        norm_func = lambda x: 1.0 - normalizer(x)

    # Draw edges with color mapping
    for u, v, data in graph.edges(data=True):
        weight = data.get("weight", None)

        if weight is None:
            # Fallback to cyan if weight is missing
            color_bgr = (255, 255, 0)
        else:
            # Map weight to color
            normalized_value = norm_func(weight)
            color_rgba = cmap(normalized_value)  # Returns (r, g, b, a) in [0, 1]

            # Convert to BGR uint8 for OpenCV
            color_bgr = (
                int(color_rgba[2] * 255),  # B
                int(color_rgba[1] * 255),  # G
                int(color_rgba[0] * 255),  # R
            )

        _draw_single_edge(vis_image, u, v, data, color_bgr, thickness)

    return vis_image


def _draw_single_edge(
    image: np.ndarray,
    u: tuple,
    v: tuple,
    data: dict,
    color: tuple,
    thickness: int,
):
    """
    Helper function to draw a single edge on the image.

    Args:
        image: Image to draw on (modified in-place)
        u: Start node (y, x)
        v: End node (y, x)
        data: Edge data dictionary
        color: BGR color tuple
        thickness: Line thickness
    """
    y1, x1 = u
    y2, x2 = v

    # If edge data has 'path', draw the full path
    if "path" in data and data["path"]:
        path_points = data["path"]
        # Draw path segments
        for i in range(len(path_points) - 1):
            py1, px1 = path_points[i]
            py2, px2 = path_points[i + 1]
            p1 = (int(px1), int(py1))
            p2 = (int(px2), int(py2))
            cv2.line(image, p1, p2, color, thickness)
    else:
        # Draw straight line between nodes
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.line(image, pt1, pt2, color, thickness)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize neural network edges color-coded by weight."
    )
    parser.add_argument("-i", "--image", required=True, help="Input raw image path")
    parser.add_argument("-l", "--label", required=True, help="Input label image path")
    parser.add_argument("-m", "--mask", required=True, help="Input mask image path")
    parser.add_argument("-o", "--output", required=True, help="Output image path")
    parser.add_argument(
        "--thickness", type=int, default=2, help="Line thickness (default: 2)"
    )
    parser.add_argument(
        "--colormap",
        default="viridis",
        help="Matplotlib colormap name (default: viridis)",
    )

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
                "closing_kernel": 5,
                "opening_kernel": 3,
            },
            "mask": {
                "dilate_offset": 100,
            },
            "background": {
                "method": "rolling_ball",
                "radius": 20,
                "light_background": True,
            },
            "threshold": {"method": "binary"},
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
            "spacing": 1.0,
            "search_radius": 20.0,
            "max_cost_threshold": 0.98,
            "intensity_weight": 10,
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
    logger.info("Creating edge weight visualization...")
    vis_result = draw_edges_by_weight(
        green_channel,
        result.mst_forest,
        thickness=args.thickness,
        colormap=args.colormap,
    )

    # Save result
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_result)
    logger.info(f"Visualization saved to: {output_path}")
    logger.info(
        f"Color legend: {args.colormap} colormap - brighter = lower cost (better quality)"
    )


if __name__ == "__main__":
    main()

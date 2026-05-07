#!/usr/bin/env python3
"""
Cost Map Visualization Tool

Visualizes the cost map components (inverted intensity, Sato response, combined)
used in neural network reconstruction pathfinding under different parameter configurations.

Usage:
    python tools/visualize_costmap.py -i <image> -l <label> -m <mask> -o <output>
"""

import sys
import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
from skimage.filters import sato
import matplotlib.pyplot as plt

# Add src to python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from neural_reconstruction.core.preprocessing.pipeline import SkinAnalysisPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_cost_map(image, intensity_weight, shape_weight):
    """
    Generate cost map with specified weights.

    Matches the logic in path_finder.py:46-71 exactly.

    Args:
        image: Grayscale uint8 image
        intensity_weight: Weight for inverted intensity component
        shape_weight: Weight for Sato tubular filter component

    Returns:
        tuple: (cost_map, inverted_intensity, inverted_sato)
    """
    # Step 1: Inverted intensity
    inverted_intensity = 255 - image.astype(np.float32)

    # Step 2: Sato filter for tubular structure detection
    logger.info(f"Computing Sato filter (sigmas 1-5)...")
    sato_response = sato(image, sigmas=range(1, 6, 1), mode="reflect")

    # Step 3: Normalize & invert Sato
    if sato_response.max() > 0:
        sato_response = (sato_response / sato_response.max()) * 255.0
    else:
        logger.warning("Sato response is all zeros - no tubular structures detected")
    inverted_sato = 255 - sato_response

    # Step 4: Weighted combination
    cost_map = (
        inverted_intensity * intensity_weight + inverted_sato * shape_weight
    ) + 1e-5

    # Log statistics
    logger.info(f"Cost map statistics:")
    logger.info(f"  Min: {cost_map.min():.2f}, Max: {cost_map.max():.2f}")
    logger.info(f"  Mean: {cost_map.mean():.2f}, Std: {cost_map.std():.2f}")

    return cost_map, inverted_intensity, inverted_sato


def visualize_single_component(
    component, title, colormap="viridis", invert=True, font_scale=1.0
):
    """
    Visualize a single cost map component with colormap.

    Args:
        component: 2D numpy array to visualize
        title: Title text to display
        colormap: Matplotlib colormap name
        invert: If True, invert colormap so low values are bright
        font_scale: Font size scale for title

    Returns:
        vis_image: BGR image with colormap applied
    """
    # Normalize to [0, 1]
    normalized = (component - component.min()) / (
        component.max() - component.min() + 1e-10
    )

    # Invert for intuitive visualization (low cost = bright)
    if invert:
        normalized = 1.0 - normalized

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(normalized)

    # Convert to BGR uint8
    vis_image = (colored[:, :, :3] * 255).astype(np.uint8)
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

    # Add title with background for better visibility
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = max(1, int(2 * font_scale))
    (text_width, text_height), _ = cv2.getTextSize(
        title, font, font_scale, font_thickness
    )

    # Draw black background rectangle
    cv2.rectangle(vis_image, (5, 5), (text_width + 15, text_height + 15), (0, 0, 0), -1)

    # Draw white text
    cv2.putText(
        vis_image,
        title,
        (10, text_height + 10),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
    )

    return vis_image


def create_component_view(image, intensity_weight, shape_weight, colormap="viridis"):
    """
    Create 3-panel horizontal component decomposition view.

    Args:
        image: Input grayscale image (ROI after preprocessing)
        intensity_weight: Weight for intensity component
        shape_weight: Weight for Sato component
        colormap: Matplotlib colormap name

    Returns:
        combined: 3-panel horizontal BGR image
    """
    logger.info(
        f"Generating cost map with weights: I={intensity_weight:.1f}, S={shape_weight:.1f}"
    )

    # Generate cost map and components
    cost_map, inv_intensity, inv_sato = generate_cost_map(
        image, intensity_weight, shape_weight
    )

    # Calculate font scale based on image height
    font_scale = min(2.0, image.shape[0] / 500.0)

    # Visualize each component
    intensity_vis = visualize_single_component(
        inv_intensity,
        "Inverted Intensity",
        colormap,
        invert=True,
        font_scale=font_scale,
    )
    sato_vis = visualize_single_component(
        inv_sato, "Inverted Sato", colormap, invert=True, font_scale=font_scale
    )
    combined_vis = visualize_single_component(
        cost_map,
        f"Cost Map (I:{intensity_weight:.1f} S:{shape_weight:.1f})",
        colormap,
        invert=True,
        font_scale=font_scale,
    )

    # Combine horizontally
    combined = np.hstack([intensity_vis, sato_vis, combined_vis])

    return combined


def create_comparison_grid(image, colormap="viridis"):
    """
    Create grid comparison of multiple weight configurations.

    Args:
        image: Input grayscale image
        colormap: Matplotlib colormap name

    Returns:
        grid_image: Vertically stacked component views
    """
    # Comparison configurations
    configs = [
        {"intensity_weight": 0.0, "shape_weight": 1.0},  # Only shape
        {"intensity_weight": 0.5, "shape_weight": 0.5},  # Balanced
        {"intensity_weight": 0.6, "shape_weight": 0.4},  # Default
        {"intensity_weight": 1.0, "shape_weight": 0.0},  # Only intensity
    ]

    logger.info(f"Creating comparison grid with {len(configs)} configurations")

    # Generate all component views
    views = []
    for config in configs:
        view = create_component_view(
            image, config["intensity_weight"], config["shape_weight"], colormap
        )
        views.append(view)

    # Stack vertically with spacing
    spacing = 20
    h, w = views[0].shape[:2]

    # Create grid canvas
    grid_h = len(views) * h + (len(views) - 1) * spacing
    grid = np.zeros((grid_h, w, 3), dtype=np.uint8)

    # Place views
    for idx, view in enumerate(views):
        y_offset = idx * (h + spacing)
        grid[y_offset : y_offset + h, :] = view

    return grid


def main():
    parser = argparse.ArgumentParser(
        description="Visualize cost maps with component decomposition."
    )
    parser.add_argument("-i", "--image", required=True, help="Input raw image path")
    parser.add_argument("-l", "--label", required=True, help="Input label image path")
    parser.add_argument("-m", "--mask", required=True, help="Input mask image path")
    parser.add_argument("-o", "--output", required=True, help="Output image path")
    parser.add_argument(
        "--intensity-weight",
        type=float,
        default=0.6,
        help="Intensity weight (default: 0.6)",
    )
    parser.add_argument(
        "--shape-weight", type=float, default=0.4, help="Shape weight (default: 0.4)"
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Create comparison grid with multiple configurations",
    )
    parser.add_argument(
        "--colormap", default="viridis", help="Matplotlib colormap (default: viridis)"
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

    # Load images
    logger.info("Loading images...")
    orig_img = cv2.imread(args.image)
    label_img = cv2.imread(args.label, cv2.IMREAD_GRAYSCALE)
    mask_img = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

    if orig_img is None:
        logger.error(f"Failed to load image: {args.image}")
        return
    if label_img is None:
        logger.error(f"Failed to load label: {args.label}")
        return
    if mask_img is None:
        logger.error(f"Failed to load mask: {args.mask}")
        return

    # Extract green channel
    if len(orig_img.shape) == 3:
        green_channel = orig_img[:, :, 1]
        logger.info(f"Extracted green channel from RGB image")
    else:
        green_channel = orig_img
        logger.info(f"Using grayscale image directly")

    logger.info(f"Image shape: {green_channel.shape}")

    # Run preprocessing (same config as other tools)
    logger.info("Running preprocessing pipeline...")
    preprocessing_config = {
        "morphology": {"closing_kernel": 5, "opening_kernel": 3},
        "mask": {"dilate_offset": 100},
        "background": {
            "method": "rolling_ball",
            "radius": 6,
            "light_background": True,
        },
        "threshold": {"method": "binary"},
        "normalization": {"enabled": True},
    }

    pipeline = SkinAnalysisPipeline(preprocessing_config)
    final_label, roi_image = pipeline.run(
        label_img, mask_img, green_channel, debug=False
    )

    logger.info(f"Preprocessing complete. ROI image shape: {roi_image.shape}")

    # Generate cost map visualization
    if args.comparison:
        logger.info("Creating comparison grid...")
        vis_result = create_comparison_grid(roi_image, args.colormap)
    else:
        logger.info("Creating single component view...")
        vis_result = create_component_view(
            roi_image, args.intensity_weight, args.shape_weight, args.colormap
        )

    # Save result
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_result)

    logger.info(f"Visualization saved to: {output_path}")
    logger.info(
        f"Color interpretation: bright ({args.colormap} high) = low cost = good paths"
    )


if __name__ == "__main__":
    main()

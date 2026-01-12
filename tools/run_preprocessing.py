#!/usr/bin/env python3
"""
Command line tool for running the skin analysis preprocessing pipeline.
"""

import sys
import os
import argparse
import cv2
import numpy as np
import yaml
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


from neural_reconstruction.core.preprocessing.pipeline import SkinAnalysisPipeline


def load_image(path: str, grayscale: bool = True) -> np.ndarray:
    """Load image from path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(path, flags)

    if image is None:
        raise ValueError(f"Failed to load image: {path}")

    return image


def save_image(path: str, image: np.ndarray):
    """Save image to path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    success = cv2.imwrite(path, image)
    if not success:
        logger.error(f"Failed to save image to {path}")


def get_default_config():
    """Return default configuration."""
    return {
        "morphology": {"closing_kernel": 3, "opening_kernel": 3},
        "mask": {
            "dilate_offset": 100  # 真皮區域向下延伸的像素數
        },
        "background": {
            "method": "rolling_ball",
            "radius": 2,
            "light_background": False,
        },
        "threshold": {"method": "binary"},
        "normalization": {
            "enabled": True  # 是否啟用區域正規化
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run skin analysis preprocessing pipeline"
    )

    parser.add_argument(
        "--label", "-l", required=True, help="Path to label image (neural fibers)"
    )
    parser.add_argument("--mask", "-m", required=True, help="Path to epidermis mask")
    parser.add_argument("--image", "-i", required=True, help="Path to original image")
    parser.add_argument(
        "--output-dir", "-o", required=True, help="Directory to save outputs"
    )
    parser.add_argument("--config", "-c", help="Path to YAML config file")
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Enable debug output"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = get_default_config()
        if args.config:
            if os.path.exists(args.config):
                with open(args.config, "r") as f:
                    user_config = yaml.safe_load(f)
                    # Simple recursive update could be added here if needed
                    # For now just updating top level keys
                    config.update(user_config)
            else:
                logger.warning(f"Config file not found: {args.config}, using defaults")

        # Initialize pipeline
        pipeline = SkinAnalysisPipeline(config)

        # Load images
        logger.info("Loading images...")
        # label_img = load_image(args.label)
        # mask_img = load_image(args.mask)
        # orig_img = load_image(args.image)
        label_img = cv2.imread(args.label, cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        original_image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
        original_green = original_image[:, :, 1]

        # Run pipeline
        logger.info("Running pipeline...")

        final_label, roi_image = pipeline.run(
            label_img, mask_img, original_green, debug=False
        )
        debug_output = None

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving results to {output_dir}")
        save_image(str(output_dir / "final_label.png"), final_label)
        save_image(str(output_dir / "roi_image.png"), roi_image)

        if args.debug and debug_output:
            debug_dir = output_dir / "debug"
            debug_dir.mkdir(exist_ok=True)

            if debug_output.processed_label is not None:
                save_image(
                    str(debug_dir / "01_processed_label.png"),
                    debug_output.processed_label,
                )
            if debug_output.dilated_mask is not None:
                save_image(
                    str(debug_dir / "02_dilated_mask.png"), debug_output.dilated_mask
                )
            if debug_output.background_corrected is not None:
                save_image(
                    str(debug_dir / "03_background_corrected.png"),
                    debug_output.background_corrected,
                )
            if debug_output.dermis_roi_mask is not None:
                save_image(
                    str(debug_dir / "04_dermis_roi_mask.png"),
                    debug_output.dermis_roi_mask,
                )
            if debug_output.pseudo_label is not None:
                save_image(
                    str(debug_dir / "05_pseudo_label.png"), debug_output.pseudo_label
                )

        logger.info("Processing complete.")

    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

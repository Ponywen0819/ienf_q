#!/usr/bin/env python3
"""
Tool to draw mask edges on target image using yellow dashed lines.

Usage:
    python tools/draw_mask_edges.py --mask <mask_path> --target <target_path> --output <output_path>
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


def draw_dashed_contour(img, contour, color, thickness=2, dash_length=10, gap_length=5):
    """
    Draw a dashed contour on the image.

    Args:
        img: Target image
        contour: Contour points
        color: Line color (B, G, R)
        thickness: Line thickness
        dash_length: Length of each dash
        gap_length: Length of gap between dashes
    """
    # Calculate the total perimeter to traverse
    perimeter = cv2.arcLength(contour, closed=True)

    # Resample contour points for smoother dashed lines
    num_points = int(perimeter)
    if num_points < 2:
        return

    # Get evenly spaced points along the contour
    points = []
    for i in range(num_points):
        t = i / num_points
        idx = int(t * len(contour))
        points.append(contour[idx][0])

    # Draw dashed line
    distance = 0
    drawing = True

    for i in range(len(points) - 1):
        pt1 = tuple(points[i])
        pt2 = tuple(points[i + 1])

        segment_length = np.linalg.norm(np.array(pt2) - np.array(pt1))

        if drawing:
            cv2.line(img, pt1, pt2, color, thickness)
            distance += segment_length
            if distance >= dash_length:
                drawing = False
                distance = 0
        else:
            distance += segment_length
            if distance >= gap_length:
                drawing = True
                distance = 0


def draw_mask_edges(mask_path, target_path, output_path, color=(0, 255, 255), thickness=2):
    """
    Draw mask edges on target image using dashed lines.

    Args:
        mask_path: Path to mask image (binary or grayscale)
        target_path: Path to target image
        output_path: Path to save output image
        color: Line color in BGR format (default: yellow = (0, 255, 255))
        thickness: Line thickness (default: 2)
    """
    # Read images
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(str(target_path))

    if mask is None:
        raise ValueError(f"Cannot read mask image: {mask_path}")
    if target is None:
        raise ValueError(f"Cannot read target image: {target_path}")

    # Ensure mask is binary
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create output image
    output = target.copy()

    # Draw each contour with dashed lines
    for contour in contours:
        if len(contour) > 2:  # Need at least 3 points for a contour
            draw_dashed_contour(output, contour, color, thickness)

    # Save output
    cv2.imwrite(str(output_path), output)
    print(f"Saved result to: {output_path}")
    print(f"Found {len(contours)} contours")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Draw mask edges on target image using yellow dashed lines"
    )
    parser.add_argument(
        "--mask", "-m",
        type=str,
        required=True,
        help="Path to mask image"
    )
    parser.add_argument(
        "--target", "-t",
        type=str,
        required=True,
        help="Path to target image"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to save output image"
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Line thickness (default: 2)"
    )
    parser.add_argument(
        "--dash-length",
        type=int,
        default=10,
        help="Length of each dash (default: 10)"
    )
    parser.add_argument(
        "--gap-length",
        type=int,
        default=5,
        help="Length of gap between dashes (default: 5)"
    )

    args = parser.parse_args()

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Draw mask edges
    draw_mask_edges(
        mask_path=args.mask,
        target_path=args.target,
        output_path=args.output,
        thickness=args.thickness
    )


if __name__ == "__main__":
    main()

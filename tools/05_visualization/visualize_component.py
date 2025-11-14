#!/usr/bin/env python3
"""Script to visualize component details with skeleton and seeds."""

import argparse
import sys
from pathlib import Path
import os
# Add src to path
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from component_detail_viz import ComponentDetailVisualizer


def main():
    """Main entry point for component visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize a component with its skeleton and seeds"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="data/Label/S163-2_a.tif",
        help="Path to input TIF image (default: data/Label/S163-2_a.tif)",
    )
    parser.add_argument(
        "--components",
        type=str,
        default="output/components/components.json",
        help="Path to components.json (default: output/components/components.json)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="output/seeds/seeds.json",
        help="Path to seeds.json (default: output/seeds/seeds.json)",
    )
    parser.add_argument(
        "--skeletons",
        type=str,
        default="output/skeletons/skeletons.json",
        help="Path to skeletons.json (default: output/skeletons/skeletons.json)",
    )
    parser.add_argument(
        "--labeled-skeletons",
        type=str,
        default="output/skeletons/labeled_skeletons.png",
        help="Path to labeled_skeletons.png (default: output/skeletons/labeled_skeletons.png)",
    )
    parser.add_argument(
        "--component-id",
        type=int,
        default=None,
        help="Specific component ID to visualize (default: random selection)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=20,
        help="Minimum area for random component selection (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for visualization (default: auto-generated in output/visualization/)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Padding around component bounding box (default: 20)",
    )

    args = parser.parse_args()

    # Validate input files exist
    image_path = Path(args.image)
    components_path = Path(args.components)
    seeds_path = Path(args.seeds)
    skeletons_path = Path(args.skeletons)
    labeled_skeletons_path = Path(args.labeled_skeletons)

    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    if not components_path.exists():
        print(f"Error: Components file not found: {components_path}")
        sys.exit(1)
    if not seeds_path.exists():
        print(f"Error: Seeds file not found: {seeds_path}")
        sys.exit(1)
    if not skeletons_path.exists():
        print(f"Error: Skeletons file not found: {skeletons_path}")
        sys.exit(1)

    print("Initializing Component Detail Visualizer...")
    print(f"Image: {image_path}")
    print(f"Components: {components_path}")
    print(f"Seeds: {seeds_path}")
    print(f"Skeletons: {skeletons_path}")
    print(f"Labeled Skeletons: {labeled_skeletons_path}")
    print()

    # Create visualizer
    visualizer = ComponentDetailVisualizer(
        image_path=image_path,
        components_path=components_path,
        seeds_path=seeds_path,
        skeletons_path=skeletons_path,
        labeled_skeletons_path=labeled_skeletons_path,
    )

    # Generate visualization
    print("Generating visualization...")
    visualizer.visualize_component(
        component_id=args.component_id,
        min_area=args.min_area,
        output_path=args.output,
        padding=args.padding,
    )

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

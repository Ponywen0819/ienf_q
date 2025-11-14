#!/usr/bin/env python3
"""
Epidermis-Dermis Boundary Crossing Analysis

Main script for detecting and analyzing nerve fibers crossing the
epidermis-dermis boundary.

Usage:
    python main_crossing_analysis.py [--config CONFIG_PATH]
"""

import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
import cv2
import networkx as nx
import numpy as np

from src.boundary_crossing import (
    CROSSING_CONFIG,
    EpidermisStatisticsBuilder,
    BoundaryDetector,
    CrossingAnalyzer,
    CrossingVisualizer
)


def load_network(network_path: Path) -> nx.Graph:
    """Load nerve network from GraphML file"""
    print(f"Loading network from: {network_path}")
    network = nx.read_graphml(network_path)

    # Convert string attributes back to appropriate types
    for node in network.nodes():
        network.nodes[node]['x'] = int(network.nodes[node]['x'])
        network.nodes[node]['y'] = int(network.nodes[node]['y'])

    # Convert edge path strings to list of tuples
    for u, v in network.edges():
        edge_data = network[u][v]
        if 'path' in edge_data and isinstance(edge_data['path'], str):
            # Parse string representation of list of tuples
            # e.g., "[(298, 3513), (297, 3514)]" -> [(298, 3513), (297, 3514)]
            try:
                import ast
                path_list = ast.literal_eval(edge_data['path'])
                network[u][v]['path'] = path_list
            except:
                # If parsing fails, set to None
                network[u][v]['path'] = None

    print(f"  Network loaded: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
    return network


def process_single_image(
    image_name: str,
    image_path: Path,
    mask_path: Path,
    network_path: Path,
    epidermis_stats: dict,
    config: dict,
    output_dir: Path
) -> dict:
    """
    Process a single image for boundary crossing analysis

    Args:
        image_name: Name identifier for the image
        image_path: Path to the original image
        mask_path: Path to the epidermis mask
        network_path: Path to the nerve network GraphML file
        epidermis_stats: Epidermis statistics dictionary
        config: Configuration dictionary
        output_dir: Output directory for results

    Returns:
        Dictionary containing analysis results
    """
    print(f"\n{'=' * 80}")
    print(f"Processing: {image_name}")
    print(f"{'=' * 80}")

    # Load data
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return None

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"ERROR: Could not load mask: {mask_path}")
        return None

    try:
        network = load_network(network_path)
    except Exception as e:
        print(f"ERROR: Could not load network: {e}")
        return None

    # Step 1: Detect boundary
    print("\n[Step 1/4] Detecting epidermis-dermis boundary...")
    boundary_detector = BoundaryDetector(config)
    boundary = boundary_detector.detect_boundary(mask)

    if not boundary:
        print("WARNING: No boundary detected")
        return None

    boundary_stats = boundary_detector.get_statistics()
    print(f"  Boundary statistics: {boundary_stats}")

    # Step 2: Find crossing candidates
    print("\n[Step 2/4] Finding boundary crossing candidates...")
    analyzer = CrossingAnalyzer(config, epidermis_stats, boundary_detector)
    candidates = analyzer.find_boundary_candidates(network)

    if not candidates:
        print("  No crossing candidates found")
        return {
            'image_name': image_name,
            'total_candidates': 0,
            'successful_crossings': 0,
            'high_confidence_crossings': 0,
        }

    # Step 3: Analyze crossings
    print("\n[Step 3/4] Analyzing crossings...")
    results, statistics = analyzer.analyze_all_crossings(candidates, image)

    # Print statistics
    print(f"\n  Results:")
    print(f"    Total candidates: {statistics['total_candidates']}")
    print(f"    Successful crossings: {statistics['successful_crossings']}")
    print(f"    High confidence crossings: {statistics['high_confidence_crossings']}")
    if 'mean_confidence' in statistics:
        print(f"    Mean confidence: {statistics['mean_confidence']:.3f}")
    if 'mean_length' in statistics:
        print(f"    Mean crossing length: {statistics['mean_length']:.1f} px")

    # Step 4: Visualize results
    print("\n[Step 4/4] Creating visualization...")
    visualizer = CrossingVisualizer(config)

    vis_output_path = output_dir / 'visualizations' / f'{image_name}_crossing.png'
    vis_image = visualizer.visualize_all(
        image, boundary_detector, candidates, results, statistics, vis_output_path
    )
    print(f"  Visualization saved to: {vis_output_path}")

    # Step 4.5: Create detailed visualizations for each candidate
    print("\n[Step 4.5/4] Creating detailed candidate visualizations...")
    candidate_detail_dir = output_dir / 'visualizations' / 'candidate_details'
    candidate_detail_dir.mkdir(parents=True, exist_ok=True)

    for idx, (candidate, result) in enumerate(zip(candidates, results), 1):
        detail_output_path = candidate_detail_dir / f'{image_name}_candidate_{idx}.png'
        visualizer.visualize_single_candidate_detailed(
            candidate_idx=idx,
            candidate=candidate,
            result=result,
            image=image,
            boundary_detector=boundary_detector,
            epidermis_stats=epidermis_stats,
            config=config,
            output_path=detail_output_path
        )

    print(f"  {len(candidates)} detailed visualizations saved to: {candidate_detail_dir}")

    # Prepare result dictionary
    result_dict = {
        'image_name': image_name,
        **statistics,
        'boundary_stats': boundary_stats,
    }

    return result_dict


def build_epidermis_statistics_from_all_images(
    image_dir: Path,
    mask_dir: Path,
    network_dir: Path,
    config: dict,
    output_path: Path
) -> dict:
    """
    Build epidermis statistics from all available images

    Args:
        image_dir: Directory containing original images
        mask_dir: Directory containing epidermis masks
        network_dir: Directory containing network files
        config: Configuration dictionary
        output_path: Path to save statistics

    Returns:
        Statistics dictionary
    """
    print(f"\n{'=' * 80}")
    print("Building Epidermis Statistics from All Images")
    print(f"{'=' * 80}\n")

    # Find all network files
    network_files = list(network_dir.glob('*.graphml'))
    if not network_files:
        print(f"ERROR: No network files found in {network_dir}")
        return None

    print(f"Found {len(network_files)} network files")

    # Aggregate statistics from multiple images (use a subset for efficiency)
    sample_size = min(10, len(network_files))
    print(f"Using {sample_size} images for statistics building\n")

    stats_builder = EpidermisStatisticsBuilder(config)

    all_green_intensities = []
    all_widths = []
    all_curvatures = []

    # Helper function to find file with any supported extension
    def find_file(directory: Path, basename: str) -> Path:
        for ext in ['.tif', '.png', '.jpg', '.jpeg']:
            file_path = directory / f'{basename}{ext}'
            if file_path.exists():
                return file_path
        return None

    # If single network file without specific name, need to find images manually
    if len(network_files) == 1 and network_files[0].stem == 'mst_forest':
        # Use all available images in the original directory
        image_files = []
        for ext in ['.tif', '.png', '.jpg', '.jpeg']:
            image_files.extend(list(image_dir.glob(f'*{ext}')))

        print(f"Single network file found. Will use first {sample_size} available images.")

        for i, image_path in enumerate(image_files[:sample_size]):
            image_name = image_path.stem
            mask_path = find_file(mask_dir, image_name)

            if not mask_path:
                continue

            print(f"[{i+1}/{sample_size}] Processing {image_name}...")

            try:
                # Load data
                image = cv2.imread(str(image_path))
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                network = load_network(network_files[0])  # Use the single network

                # Extract statistics
                image_stats = stats_builder.build_statistics(network, image, mask)

                print(f"  Extracted statistics from {image_name}")

            except Exception as e:
                print(f"  ERROR: Failed to process {image_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    else:
        # Normal case: network files have specific names
        for i, network_file in enumerate(network_files[:sample_size]):
            image_name = network_file.stem.replace('_network', '').replace('_mst_forest', '')
            image_path = find_file(image_dir, image_name)
            mask_path = find_file(mask_dir, image_name)

            if not image_path or not mask_path:
                continue

            print(f"[{i+1}/{sample_size}] Processing {image_name}...")

            try:
                # Load data
                image = cv2.imread(str(image_path))
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                network = load_network(network_file)

                # Extract statistics
                image_stats = stats_builder.build_statistics(network, image, mask)

                print(f"  Extracted statistics from {image_name}")

            except Exception as e:
                print(f"  ERROR: Failed to process {image_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save statistics
    stats_builder.save_statistics(output_path)
    print(f"\nStatistics saved to: {output_path}")

    return stats_builder.statistics


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Boundary Crossing Analysis')
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to custom configuration JSON file'
    )
    parser.add_argument(
        '--single',
        type=str,
        help='Process only a single image (provide image name without extension)'
    )
    parser.add_argument(
        '--rebuild-stats',
        action='store_true',
        help='Rebuild epidermis statistics even if they exist'
    )

    args = parser.parse_args()

    # Load configuration
    config = CROSSING_CONFIG.copy()
    if args.config and args.config.exists():
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            config.update(custom_config)
        print(f"Loaded custom configuration from: {args.config}")

    # Define paths
    project_root = Path(__file__).parent
    data_dir = project_root / 'data'
    output_dir = project_root / 'output'

    image_dir = data_dir / 'Original'
    mask_dir = data_dir / 'Mask'
    network_dir = output_dir / 'reconstruction'  # Where MST networks are stored

    # Support both .tif and .png formats
    image_extensions = ['.tif', '.png', '.jpg', '.jpeg']

    # Create output directories
    crossing_output_dir = output_dir / 'boundary_crossing'
    crossing_output_dir.mkdir(parents=True, exist_ok=True)
    (crossing_output_dir / 'statistics').mkdir(exist_ok=True)
    (crossing_output_dir / 'visualizations').mkdir(exist_ok=True)
    (crossing_output_dir / 'reports').mkdir(exist_ok=True)

    # Build or load epidermis statistics
    stats_path = crossing_output_dir / 'statistics' / 'epidermis_statistics.json'

    if args.rebuild_stats or not stats_path.exists():
        epidermis_stats = build_epidermis_statistics_from_all_images(
            image_dir, mask_dir, network_dir, config, stats_path
        )
        if epidermis_stats is None:
            print("ERROR: Failed to build epidermis statistics")
            return
    else:
        print(f"Loading existing epidermis statistics from: {stats_path}")
        stats_builder = EpidermisStatisticsBuilder(config)
        epidermis_stats = stats_builder.load_statistics(stats_path)

    print(f"\nEpidermis Statistics Summary:")
    print(f"  Green intensity: {epidermis_stats.get('green_intensity_mean', 0):.1f} "
          f"± {epidermis_stats.get('green_intensity_std', 0):.1f}")
    if 'width_mean' in epidermis_stats:
        print(f"  Width: {epidermis_stats['width_mean']:.1f} ± {epidermis_stats['width_std']:.1f}")
    if 'curvature_mean' in epidermis_stats:
        print(f"  Curvature: {epidermis_stats['curvature_mean']:.1f}°")

    # Helper function to find file with any supported extension
    def find_file_with_extension(directory: Path, basename: str, extensions: list) -> Path:
        """Find a file with any of the given extensions"""
        for ext in extensions:
            file_path = directory / f'{basename}{ext}'
            if file_path.exists():
                return file_path
        return None

    # Process images
    if args.single:
        # Process single image
        image_name = args.single

        # Find files with any supported extension
        image_path = find_file_with_extension(image_dir, image_name, image_extensions)
        mask_path = find_file_with_extension(mask_dir, image_name, image_extensions)
        network_path = network_dir / f'{image_name}_mst_forest.graphml'

        # Fallback: check for single network file if specific one not found
        if not network_path.exists():
            network_path = network_dir / 'mst_forest.graphml'

        if not image_path or not mask_path or not network_path.exists():
            print(f"ERROR: Could not find all required files for {image_name}")
            if not image_path:
                print(f"  Missing image in {image_dir}")
            if not mask_path:
                print(f"  Missing mask in {mask_dir}")
            if not network_path.exists():
                print(f"  Missing network: {network_path}")
            return

        result = process_single_image(
            image_name, image_path, mask_path, network_path,
            epidermis_stats, config, crossing_output_dir
        )

        if result:
            print(f"\n{'=' * 80}")
            print("Processing Complete!")
            print(f"{'=' * 80}")

    else:
        # Process all images
        print(f"\n{'=' * 80}")
        print("Processing All Images")
        print(f"{'=' * 80}\n")

        network_files = list(network_dir.glob('*_mst_forest.graphml'))

        # If no individual network files, check for single network file
        if not network_files:
            single_network = network_dir / 'mst_forest.graphml'
            if single_network.exists():
                print(f"Found single network file: {single_network}")
                print("Note: Processing will use the same network for all images")
                network_files = [single_network]

        print(f"Found {len(network_files)} network file(s) to process\n")

        all_results = []
        for i, network_path in enumerate(network_files):
            if network_path.name == 'mst_forest.graphml':
                # Single network file, need to find which image it corresponds to
                # For now, skip batch processing if only single network exists
                print("WARNING: Only single network file found (mst_forest.graphml)")
                print("Use --single option to process a specific image")
                break

            image_name = network_path.stem.replace('_mst_forest', '')

            # Find files with any supported extension
            image_path = find_file_with_extension(image_dir, image_name, image_extensions)
            mask_path = find_file_with_extension(mask_dir, image_name, image_extensions)

            if not image_path or not mask_path:
                print(f"Skipping {image_name}: Missing image or mask")
                continue

            result = process_single_image(
                image_name, image_path, mask_path, network_path,
                epidermis_stats, config, crossing_output_dir
            )

            if result:
                all_results.append(result)

        # Generate summary report
        print(f"\n{'=' * 80}")
        print("Generating Summary Report")
        print(f"{'=' * 80}\n")

        if all_results:
            # Save detailed results as JSON
            json_output = crossing_output_dir / 'reports' / 'crossing_details.json'
            with open(json_output, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Detailed results saved to: {json_output}")

            # Save summary as CSV
            csv_output = crossing_output_dir / 'reports' / 'crossing_summary.csv'
            with open(csv_output, 'w', newline='') as f:
                if all_results:
                    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                    writer.writeheader()
                    writer.writerows(all_results)
            print(f"Summary CSV saved to: {csv_output}")

            # Print overall statistics
            print(f"\n{'=' * 80}")
            print("Overall Statistics")
            print(f"{'=' * 80}")
            print(f"Total images processed: {len(all_results)}")

            total_candidates = sum(r['total_candidates'] for r in all_results)
            total_successful = sum(r['successful_crossings'] for r in all_results)
            total_high_conf = sum(r['high_confidence_crossings'] for r in all_results)

            print(f"Total crossing candidates: {total_candidates}")
            print(f"Total successful crossings: {total_successful}")
            print(f"Total high-confidence crossings: {total_high_conf}")

            if total_candidates > 0:
                print(f"Overall success rate: {total_successful / total_candidates * 100:.1f}%")

            # Calculate mean statistics across images
            images_with_crossings = [r for r in all_results if r['successful_crossings'] > 0]
            if images_with_crossings:
                mean_crossings = np.mean([r['successful_crossings'] for r in images_with_crossings])
                print(f"Mean crossings per image (with crossings): {mean_crossings:.1f}")

        print(f"\n{'=' * 80}")
        print("All Processing Complete!")
        print(f"{'=' * 80}")


if __name__ == '__main__':
    main()

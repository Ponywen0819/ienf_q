#!/usr/bin/env python3
"""
Seed Extraction Visualization Module

This module visualizes the seed extraction pipeline including:
1. Skeleton structure
2. Topology (nodes and edges)
3. Extracted seed points

Outputs:
- seeds_overlay.png: Main visualization with skeleton, topology, and seeds
- seeds_statistics.png: Statistical charts
- seeds_info.json: Detailed topology and seed data
"""

import sys
from pathlib import Path

# # Add src to path for imports
# src_path = Path(__file__).parent.parent / 'src'
# sys.path.insert(0, str(src_path))

import json
import logging
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# Import directly from modules to avoid __init__.py issues
from nueral_reconstruction.connected_components import ConnectedComponentsAnalyzer
from nueral_reconstruction.skeletonization import SkeletonAnalyzer
from nueral_reconstruction.seed_extraction import SkeletonTopologyBuilder, EdgeSeedExtractor
from nueral_reconstruction.config_loader import load_config, IENFConfig
from nueral_reconstruction import NeuralReconstructionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def visualize_seeds(
    annotation_path: str,
    output_dir: str,
    green_channel_path: Optional[str] = None,
    show_skeleton: bool = True,
    show_nodes: bool = True,
    show_edges: bool = True,
    show_seeds: bool = True,
    show_statistics: bool = True,
    alpha: float = 0.7,
    config_path: Optional[str] = None
):
    """
    Visualize seed extraction results with topology structure.

    Args:
        annotation_path: Path to annotation image (binary mask)
        output_dir: Directory to save visualization outputs
        green_channel_path: Optional path to green channel image for overlay
        show_skeleton: Whether to show skeleton structure
        show_nodes: Whether to show topology nodes (endpoints and branchpoints)
        show_edges: Whether to show topology edges
        show_seeds: Whether to show extracted seeds
        show_statistics: Whether to generate statistics plots
        alpha: Transparency for overlay (0.0 to 1.0)
        config_path: Optional path to config file (uses default if None)
    """
    # Load configuration
    config = load_config(config_path) if config_path else load_config()
    logger.info("Loaded configuration:")
    logger.info(f"  Connectivity: {config.connected_components.connectivity}")
    logger.info(f"  Min area: {config.connected_components.min_area}")
    logger.info(f"  Base segment length: {config.seed_extraction.base_segment_length}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")

    # Load annotation image
    logger.info(f"Loading annotation from: {annotation_path}")
    annotation = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
    if annotation is None:
        raise ValueError(f"Failed to load annotation image: {annotation_path}")

    height, width = annotation.shape
    logger.info(f"Image dimensions: {width} x {height}")

    # Load green channel if provided
    if green_channel_path:
        logger.info(f"Loading green channel from: {green_channel_path}")
        green_channel = cv2.imread(green_channel_path, cv2.IMREAD_GRAYSCALE)
        if green_channel is None:
            logger.warning(f"Failed to load green channel: {green_channel_path}")
            green_channel = None
    else:
        green_channel = None



    pipeline = NeuralReconstructionPipeline()
    res = pipeline.run(annotation, green_channel, stop_step='topology_and_seeds')
    all_topologies = res['stages']['topology_and_seeds']['topologies']
    all_seeds = res['stages']['topology_and_seeds']['seeds']
    skeleton_results = res['stages']['skeletonization']['skeleton_data']

    # Create visualization
    logger.info("\n=== Creating Visualization ===")

    # Layer 1: Create background (Green Channel Grayscale)
    logger.info("Creating background...")
    if green_channel is not None:
        if len(green_channel.shape) == 2:
            vis_image = cv2.cvtColor(green_channel, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = green_channel.copy()
    else:
        # Create black background if no green channel
        vis_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Layer 2: Overlay components (annotation mask) - Opaque Red
    logger.info("Overlaying components...")
    component_mask = annotation > 0
    vis_image[component_mask] = [0, 0, 255]  # Opaque Red

    # Layer 3: Draw skeleton if requested
    if show_skeleton:
        logger.info("Drawing skeletons...")
        for skel_info in skeleton_results:
            region = skel_info['region']
            skeleton = skel_info['skeleton']
            minr, minc, maxr, maxc = region.bbox

            # Create global skeleton mask
            for local_y in range(skeleton.shape[0]):
                for local_x in range(skeleton.shape[1]):
                    if skeleton[local_y, local_x] > 0:
                        global_y = local_y + minr
                        global_x = local_x + minc
                        # Yellow color for skeleton (BGR)
                        vis_image[global_y, global_x] = (0, 255, 255)

    # Layer 4: Draw topology edges if requested
    if show_edges:
        logger.info("Drawing topology edges...")
        for topo_info in all_topologies:
            topology = topo_info['topology']
            for edge in topology['edges']:
                path = edge['path']
                # Draw edge path in cyan
                for i in range(len(path) - 1):
                    y1, x1 = path[i]
                    y2, x2 = path[i + 1]
                    cv2.line(vis_image, (x1, y1), (x2, y2), (255, 255, 0), 1)

    # Layer 5: Draw topology nodes if requested
    if show_nodes:
        logger.info("Drawing topology nodes...")
        for topo_info in all_topologies:
            topology = topo_info['topology']
            for node in topology['nodes']:
                y, x = node['position']
                if node['type'] == 'endpoint':
                    # Red circle for endpoints
                    # cv2.circle(vis_image, (x, y), 1, (0, 0, 255), -1)
                    vis_image[y, x] = (0, 0, 255)
                elif node['type'] == 'branchpoint':
                    # Blue square for branchpoints
                    cv2.rectangle(vis_image, (x-1, y-1), (x+1, y+1), (255, 0, 0), -1)

    # Generate Cropped Visualization with Matplotlib for Seeds
    logger.info("Generating cropped visualization...")
    crop_x, crop_y = 1700, 360
    crop_w, crop_h = 400, 300
    
    # Extract crop from current state (Background + Mask + Skeleton + Nodes/Edges)
    # Note: vis_image is BGR
    if height > crop_y + crop_h and width > crop_x + crop_w:
        crop_img = vis_image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].copy()
        crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(crop_img_rgb)
        
        # Plot seeds using matplotlib
        if show_seeds:
            seed_xs = []
            seed_ys = []
            for seed in all_seeds:
                y, x = seed['position']
                # Check if seed is within crop
                if crop_y <= y < crop_y + crop_h and crop_x <= x < crop_x + crop_w:
                    seed_xs.append(x - crop_x)
                    seed_ys.append(y - crop_y)
            
            if seed_xs:
                plt.scatter(seed_xs, seed_ys, c='lime', s=5, edgecolors='white', linewidth=0.3, label='Seeds')
        
        plt.axis('off')
        plt.tight_layout()
        crop_output_file = output_path / "seeds_overlay_crop.png"
        plt.savefig(crop_output_file, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        logger.info(f"Saved cropped visualization: {crop_output_file}")
    else:
        logger.warning("Image too small for requested crop region.")

    # Layer 6: Draw seeds if requested (topmost layer)
    if show_seeds:
        logger.info("Drawing seeds...")
        for seed in all_seeds:
            y, x = seed['position']
            # Green dot for seeds
            # cv2.circle(vis_image, (x, y), 1, (0, 255, 0), -1)
            vis_image[y, x] = (0, 255, 0)


    # Save main visualization
    output_file = output_path / "seeds_overlay.png"
    cv2.imwrite(str(output_file), vis_image)
    logger.info(f"Saved visualization: {output_file}")

    # Generate statistics if requested
    if show_statistics:
        logger.info("\n=== Generating Statistics ===")
        _generate_statistics(
            all_topologies,
            all_seeds,
            output_path
        )

    # Save topology and seed information to JSON
    logger.info("\n=== Saving Topology and Seed Information ===")
    _save_seed_info(
        all_topologies,
        all_seeds,
        output_path
    )

    logger.info("\n=== Visualization Complete ===")
    logger.info(f"All outputs saved to: {output_path}")


def _generate_statistics(
    all_topologies: List[Dict],
    all_seeds: List[Dict],
    output_path: Path
):
    """Generate statistical plots for topology and seeds."""

    # Collect statistics
    total_components = len(all_topologies)
    total_nodes = sum(len(t['topology']['nodes']) for t in all_topologies)
    total_edges = sum(len(t['topology']['edges']) for t in all_topologies)
    total_seeds = len(all_seeds)

    # Count endpoints and branchpoints
    total_endpoints = 0
    total_branchpoints = 0
    for topo_info in all_topologies:
        for node in topo_info['topology']['nodes']:
            if node['type'] == 'endpoint':
                total_endpoints += 1
            else:
                total_branchpoints += 1

    # Collect edge lengths
    edge_lengths = []
    for topo_info in all_topologies:
        for edge in topo_info['topology']['edges']:
            edge_lengths.append(edge['length'])

    # Count seeds per component
    seeds_per_component = {}
    for seed in all_seeds:
        comp_id = seed['component_id']
        seeds_per_component[comp_id] = seeds_per_component.get(comp_id, 0) + 1

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Seed Extraction Statistics', fontsize=16, fontweight='bold')

    # 1. Component Summary (bar chart)
    ax = axes[0, 0]
    categories = ['Components', 'Nodes', 'Edges', 'Seeds']
    values = [total_components, total_nodes, total_edges, total_seeds]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('Pipeline Summary')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}',
                ha='center', va='bottom', fontweight='bold')

    # 2. Node Type Distribution (pie chart)
    ax = axes[0, 1]
    node_types = ['Endpoints', 'Branchpoints']
    node_counts = [total_endpoints, total_branchpoints]
    colors_pie = ['#e74c3c', '#3498db']
    ax.pie(node_counts, labels=node_types, autopct='%1.1f%%',
           colors=colors_pie, startangle=90)
    ax.set_title(f'Node Distribution (Total: {total_nodes})')

    # 3. Edge Length Distribution (histogram)
    ax = axes[0, 2]
    if edge_lengths:
        ax.hist(edge_lengths, bins=30, color='#f39c12', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(edge_lengths), color='red', linestyle='--',
                   label=f'Mean: {np.mean(edge_lengths):.1f}')
        ax.axvline(np.median(edge_lengths), color='blue', linestyle='--',
                   label=f'Median: {np.median(edge_lengths):.1f}')
        ax.set_xlabel('Edge Length (pixels)')
        ax.set_ylabel('Frequency')
        ax.set_title('Edge Length Distribution')
        ax.legend()
        ax.grid(alpha=0.3)

    # 4. Seeds per Component (histogram)
    ax = axes[1, 0]
    if seeds_per_component:
        seed_counts = list(seeds_per_component.values())
        ax.hist(seed_counts, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(seed_counts), color='red', linestyle='--',
                   label=f'Mean: {np.mean(seed_counts):.1f}')
        ax.axvline(np.median(seed_counts), color='blue', linestyle='--',
                   label=f'Median: {np.median(seed_counts):.1f}')
        ax.set_xlabel('Seeds per Component')
        ax.set_ylabel('Frequency')
        ax.set_title('Seed Distribution Across Components')
        ax.legend()
        ax.grid(alpha=0.3)

    # 5. Edge Statistics (box plot)
    ax = axes[1, 1]
    if edge_lengths:
        bp = ax.boxplot([edge_lengths], vert=True, patch_artist=True,
                        labels=['Edge Lengths'])
        bp['boxes'][0].set_facecolor('#f39c12')
        bp['boxes'][0].set_alpha(0.7)
        ax.set_ylabel('Length (pixels)')
        ax.set_title('Edge Length Statistics')
        ax.grid(axis='y', alpha=0.3)

    # 6. Detailed Statistics (text)
    ax = axes[1, 2]
    ax.axis('off')

    stats_text = f"""
TOPOLOGY STATISTICS
{'='*40}

Components: {total_components}
Total Nodes: {total_nodes}
  - Endpoints: {total_endpoints}
  - Branchpoints: {total_branchpoints}
Total Edges: {total_edges}

EDGE STATISTICS
{'='*40}

Total Length: {sum(edge_lengths):.1f} pixels
Mean Length: {np.mean(edge_lengths):.1f} pixels
Median Length: {np.median(edge_lengths):.1f} pixels
Min Length: {min(edge_lengths):.1f} pixels
Max Length: {max(edge_lengths):.1f} pixels
Std Dev: {np.std(edge_lengths):.1f} pixels

SEED STATISTICS
{'='*40}

Total Seeds: {total_seeds}
Components with Seeds: {len(seeds_per_component)}
Mean Seeds/Component: {np.mean(list(seeds_per_component.values())):.1f}
Max Seeds/Component: {max(seeds_per_component.values())}
Min Seeds/Component: {min(seeds_per_component.values())}
"""

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save statistics plot
    stats_file = output_path / "seeds_statistics.png"
    plt.savefig(stats_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved statistics: {stats_file}")
    plt.close()


def _save_seed_info(
    all_topologies: List[Dict],
    all_seeds: List[Dict],
    output_path: Path
):
    """Save topology and seed information to JSON file."""

    info = {
        'summary': {
            'total_components': len(all_topologies),
            'total_nodes': sum(len(t['topology']['nodes']) for t in all_topologies),
            'total_edges': sum(len(t['topology']['edges']) for t in all_topologies),
            'total_seeds': len(all_seeds)
        },
        'topologies': [],
        'seeds': []
    }

    # Add topology information
    for topo_info in all_topologies:
        component_id = topo_info['component_id']
        topology = topo_info['topology']

        nodes_data = []
        for node in topology['nodes']:
            nodes_data.append({
                'id': node['id'],
                'position': {'y': node['position'][0], 'x': node['position'][1]},
                'type': node['type']
            })

        edges_data = []
        for edge in topology['edges']:
            edges_data.append({
                'source': edge['source'],
                'target': edge['target'],
                'length': edge['length'],
                'path_points': len(edge['path'])
            })

        info['topologies'].append({
            'component_id': component_id,
            'nodes': nodes_data,
            'edges': edges_data
        })

    # Add seed information
    for seed in all_seeds:
        info['seeds'].append({
            'position': {'y': seed['position'][0], 'x': seed['position'][1]},
            'component_id': seed['component_id'],
        })

    # Save to JSON
    json_file = output_path / "seeds_info.json"
    with open(json_file, 'w') as f:
        json.dump(info, f, indent=2)

    logger.info(f"Saved seed info: {json_file}")


if __name__ == "__main__":
    # Example usage
    annotation_path = "output/preprocessing_normalization/final_label.png"
    green_channel_path = "output/preprocessing_normalization/roi_image.png"
    output_dir = "output/seeds"

    visualize_seeds(
        annotation_path=annotation_path,
        output_dir=output_dir,
        green_channel_path=green_channel_path,
        show_skeleton=True,
        show_nodes=False,
        show_edges=False,
        show_seeds=True,
        show_statistics=False,
        alpha=0.7
    )

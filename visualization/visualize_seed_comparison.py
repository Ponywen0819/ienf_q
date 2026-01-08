import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from scipy.spatial.distance import cdist

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_seeds_info(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['topologies']

def get_component_topology(topologies, comp_id):
    for comp in topologies:
        if comp['component_id'] == comp_id:
            return comp
    return None

def interpolate_points(p1, p2, num_points):
    return np.linspace(p1, p2, num_points)

def load_component_mask(comp_id):
    path = f'output/all_components/skeleton_{comp_id}.png'
    if not os.path.exists(path):
        return None
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def get_component_pixels(mask):
    if mask is None: return []
    return np.argwhere(mask > 0)

def visualize_comparison(long_id, short_id, original_image_path):
    seeds_info_path = 'output/seeds/seeds_info.json'
    topologies = load_seeds_info(seeds_info_path)
    
    comp_long = get_component_topology(topologies, long_id)
    comp_short = get_component_topology(topologies, short_id)
    
    if not comp_long or not comp_short:
        print("Component not found.")
        return

    # Extract nodes and edges
    # Nodes: {id: pos}
    nodes_long = {n['id']: (n['position']['y'], n['position']['x']) for n in comp_long['nodes']}
    nodes_short = {n['id']: (n['position']['y'], n['position']['x']) for n in comp_short['nodes']}
    
    # Collect all points for bounding box
    all_points = list(nodes_long.values()) + list(nodes_short.values())
    all_points = np.array(all_points)
    
    min_y, min_x = np.min(all_points, axis=0)
    max_y, max_x = np.max(all_points, axis=0)
    
    # Add padding
    padding = 50
    min_y = int(max(0, min_y - padding))
    min_x = int(max(0, min_x - padding))
    max_y = int(max_y + padding)
    max_x = int(max_x + padding)
    
    # Load original image
    original_img = cv2.imread(original_image_path)
    if original_img is None:
        print(f"Could not load original image: {original_image_path}")
        # Create black image if original not found
        original_img = np.zeros((max_y + 100, max_x + 100, 3), dtype=np.uint8)
    
    # Extract Green Channel
    # OpenCV loads as BGR, so Green is index 1
    green_channel = original_img[:, :, 1]
    
    # Load Mask
    mask_path = original_image_path.replace('Original', 'Label')
    mask_img = cv2.imread("output/preprocessing_compare/config_b/final_label.png", cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        print(f"Could not load mask image: {mask_path}")
        mask_img = np.zeros_like(green_channel)

    h, w = original_img.shape[:2]
    max_y = min(h, max_y)
    max_x = min(w, max_x)
    
    # Crop images
    crop_img = green_channel[min_y:max_y, min_x:max_x]
    crop_mask = mask_img[min_y:max_y, min_x:max_x]
    
    # Create visualization background (Green channel as grayscale + Mask overlay)
    # Convert grayscale green channel to RGB for matplotlib
    vis_bg = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2RGB)
    
    # Create red overlay for mask
    # Where mask > 0, add red tint
    mask_indices = crop_mask > 0
    # Increase Red channel where mask is present
    vis_bg[mask_indices, 0] = np.clip(vis_bg[mask_indices, 0] + 100, 0, 255) 
    
    # Adjust coordinates to crop
    def adjust_coords(nodes):
        return {k: (v[0] - min_y, v[1] - min_x) for k, v in nodes.items()}
        
    nodes_long_adj = adjust_coords(nodes_long)
    nodes_short_adj = adjust_coords(nodes_short)
    
    # Generate points for drawing lines (edges)
    def get_edge_segments(comp, nodes_adj):
        segments = []
        all_points = []
        for edge in comp['edges']:
            p1 = nodes_adj[edge['source']]
            p2 = nodes_adj[edge['target']]
            # Interpolate for drawing
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            num_points = int(dist) + 2
            pts = interpolate_points(p1, p2, num_points)
            segments.append(pts)
            all_points.extend(pts)
        return segments, np.array(all_points)

    long_segments, long_points_dense = get_edge_segments(comp_long, nodes_long_adj)
    short_segments, short_points_dense = get_edge_segments(comp_short, nodes_short_adj)
    
    # 1. Identify Endpoints for Long Component (A)
    endpoints_long = []
    for n in comp_long['nodes']:
        if n['type'] == 'endpoint':
            pos = nodes_long_adj[n['id']]
            endpoints_long.append(pos)
    endpoints_long = np.array(endpoints_long)
    
    # 2. Identify Uniform Seeds for Long Component (A)
    # Sample from the dense points generated from edges
    uniform_seeds_long = []
    for seg in long_segments:
        if len(seg) > 0:
            # Sample every 10 pixels
            uniform_seeds_long.extend(seg[::10])
    uniform_seeds_long = np.array(uniform_seeds_long)
    
    if len(uniform_seeds_long) == 0:
        uniform_seeds_long = endpoints_long

    # 3. Identify Connection Point on Short Component (B)
    # Use endpoints of B
    endpoints_short = []
    for n in comp_short['nodes']:
        if n['type'] == 'endpoint':
            pos = nodes_short_adj[n['id']]
            endpoints_short.append(pos)
    endpoints_short = np.array(endpoints_short)
    
    if len(endpoints_short) == 0:
         endpoints_short = np.array(list(nodes_short_adj.values()))

    # Find closest endpoint of B to ANY point of A (using dense points for better accuracy)
    if len(long_points_dense) > 0:
        dists_b_to_a = cdist(endpoints_short, long_points_dense)
        min_dist_idx = np.unravel_index(np.argmin(dists_b_to_a), dists_b_to_a.shape)
        b_connect_pt = endpoints_short[min_dist_idx[0]]
    else:
        b_connect_pt = endpoints_short[0]

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Helper to plot segments
    def plot_segments(ax, segments, color='white', alpha=0.5, linewidth=1.5):
        for seg in segments:
             ax.plot(seg[:, 1], seg[:, 0], c=color, linewidth=linewidth, alpha=alpha)

    # Plot 1: Sparse Seeds (Issue)
    ax1 = axes[0]
    # Display the prepared background (Green Channel + Red Mask Overlay)
    ax1.imshow(vis_bg)
    ax1.set_title("Issue - Sparse Seeds\n(Suboptimal Connection / High Cost)")
    
    # Draw Long Component (A)
    plot_segments(ax1, long_segments, color='white', alpha=0.6)
    
    # Draw Endpoints (Red)
    ax1.scatter(endpoints_long[:, 1], endpoints_long[:, 0], c='red', s=60, zorder=20, label='Endpoints', edgecolors='black')
    
    # Draw Short Component (B)
    plot_segments(ax1, short_segments, color='cyan', alpha=0.6)
    
    # Draw Connection Point on B
    ax1.scatter(b_connect_pt[1], b_connect_pt[0], c='cyan', s=40, zorder=20, edgecolors='black')
    
    # Calculate connection to nearest ENDPOINT of A
    dists_to_endpoints = cdist([b_connect_pt], endpoints_long)
    nearest_endpoint_idx = np.argmin(dists_to_endpoints)
    nearest_endpoint = endpoints_long[nearest_endpoint_idx]
    
    # Draw connection line
    ax1.plot([b_connect_pt[1], nearest_endpoint[1]], [b_connect_pt[0], nearest_endpoint[0]], 
             c='yellow', linestyle='--', linewidth=2, label='Connection')
    
    ax1.legend(loc='upper right')
    ax1.axis('off')
    
    # Plot 2: Uniform Seeds (Solution)
    ax2 = axes[1]
    # Display the prepared background (Green Channel + Red Mask Overlay)
    ax2.imshow(vis_bg)
    ax2.set_title("Solution - Uniform Seeds\n(Near-Optimal Connection / Low Cost)")
    
    # Draw Long Component (A)
    plot_segments(ax2, long_segments, color='white', alpha=0.6)
        
    # Draw Uniform Seeds (Green)
    if len(uniform_seeds_long) > 0:
        ax2.scatter(uniform_seeds_long[:, 1], uniform_seeds_long[:, 0], c='lime', s=15, zorder=20, label='Uniform Seeds')
    else:
        print("Warning: No uniform seeds to plot!")
    
    # Draw Short Component (B)
    plot_segments(ax2, short_segments, color='cyan', alpha=0.6)
    
    # Draw Connection Point on B
    ax2.scatter(b_connect_pt[1], b_connect_pt[0], c='cyan', s=40, zorder=20, edgecolors='black')
    
    # Calculate connection to nearest UNIFORM SEED of A
    if len(uniform_seeds_long) > 0:
        dists_to_seeds = cdist([b_connect_pt], uniform_seeds_long)
        nearest_seed_idx = np.argmin(dists_to_seeds)
        nearest_seed = uniform_seeds_long[nearest_seed_idx]
        
        # Draw connection line
        ax2.plot([b_connect_pt[1], nearest_seed[1]], [b_connect_pt[0], nearest_seed[0]], 
                 c='yellow', linestyle='--', linewidth=2, label='Connection')
    
    ax2.legend(loc='upper right')
    ax2.axis('off')
    
    plt.tight_layout()
    output_path = 'output/seed_comparison_visualization.png'
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    # IDs found by the finder script
    LONG_ID = 194
    SHORT_ID = 201
    ORIGINAL_IMAGE = 'data/Original/S163-2_a.tif'
    
    visualize_comparison(LONG_ID, SHORT_ID, ORIGINAL_IMAGE)

import json
import numpy as np
import cv2
import os
from scipy.spatial.distance import cdist

def load_seeds_info(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['topologies']

def get_component_pixels(comp_id):
    path = f'output/all_components/skeleton_{comp_id}.png'
    if not os.path.exists(path):
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Get coordinates (y, x)
    pts = np.argwhere(mask > 0)
    return pts

def find_best_demo_pair(topologies):
    long_components = []
    short_components = []

    # Filter components
    for comp in topologies:
        total_length = sum(edge['length'] for edge in comp['edges'])
        
        # Get endpoints
        endpoints = [
            (n['position']['y'], n['position']['x']) 
            for n in comp['nodes'] if n['type'] == 'endpoint'
        ]
        
        comp_info = {
            'id': comp['component_id'],
            'length': total_length,
            'endpoints': np.array(endpoints) if endpoints else np.empty((0, 2))
        }

        if total_length > 50:
            long_components.append(comp_info)
        elif 10 < total_length < 50:
            short_components.append(comp_info)

    print(f"Scanning {len(long_components)} long and {len(short_components)} short components...")

    best_pair = None
    max_diff = -1
    best_stats = {}

    for long_c in long_components:
        # Load all pixels for long component (Uniform Seeds)
        long_pixels = get_component_pixels(long_c['id'])
        if long_pixels is None or len(long_pixels) == 0:
            continue
            
        long_endpoints = long_c['endpoints']
        if len(long_endpoints) == 0:
            continue

        for short_c in short_components:
            short_endpoints = short_c['endpoints']
            if len(short_endpoints) == 0:
                continue
            
            # 1. Sparse Distance: Short Endpoints <-> Long Endpoints
            dists_sparse = cdist(short_endpoints, long_endpoints)
            min_dist_sparse = np.min(dists_sparse)
            
            # 2. Uniform Distance: Short Endpoints <-> Long Pixels (All)
            dists_uniform = cdist(short_endpoints, long_pixels)
            min_dist_uniform = np.min(dists_uniform)
            
            # We want a case where Uniform is much better (smaller) than Sparse
            # Diff = Sparse - Uniform
            diff = min_dist_sparse - min_dist_uniform
            
            # Also ensure they are somewhat close (don't want a pair 1000 pixels away)
            if min_dist_uniform < 30 and diff > max_diff:
                max_diff = diff
                best_pair = (long_c, short_c)
                best_stats = {
                    'sparse_dist': min_dist_sparse,
                    'uniform_dist': min_dist_uniform,
                    'diff': diff
                }

    return best_pair, best_stats

if __name__ == "__main__":
    seeds_info_path = 'output/seeds/seeds_info.json'
    topologies = load_seeds_info(seeds_info_path)
    pair, stats = find_best_demo_pair(topologies)
    
    if pair:
        long_c, short_c = pair
        print(f"Best pair found:")
        print(f"Long Component ID: {long_c['id']}, Length: {long_c['length']:.2f}")
        print(f"Short Component ID: {short_c['id']}, Length: {short_c['length']:.2f}")
        print(f"Sparse Distance (Endpoint-to-Endpoint): {stats['sparse_dist']:.2f}")
        print(f"Uniform Distance (Endpoint-to-Pixel): {stats['uniform_dist']:.2f}")
        print(f"Improvement: {stats['diff']:.2f}")
    else:
        print("No suitable pair found.")

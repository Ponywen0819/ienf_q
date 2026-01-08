import cv2
import numpy as np
import json
import os

# Constants
IMAGE_PATH = 'data/Original/S163-2_a.tif'
LABEL_PATH = 'output/preprocessing/final_label.png'
SEEDS_PATH = 'output/seeds/seeds_info.json'
OUTPUT_PATH = 'output/focused_seeds_visualization.png'

CROP_X = 1700
CROP_Y = 360
CROP_WIDTH = 400
CROP_HEIGHT = 300

def load_seeds_info(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Seeds file not found at {filepath}")
        return []
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['topologies']

def main():
    # 1. Load Images
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        return
    
    original_img = cv2.imread(IMAGE_PATH)
    # Extract Green Channel
    green_channel = original_img[:, :, 1]
    
    # Convert to RGB for visualization (so we can draw colored overlays)
    vis_img = cv2.cvtColor(green_channel, cv2.COLOR_GRAY2BGR)
    
    # 2. Load and Apply Mask (Opaque Red)
    if os.path.exists(LABEL_PATH):
        mask_img = cv2.imread(LABEL_PATH, cv2.IMREAD_GRAYSCALE)
        # Where mask is present, set to solid Red (0, 0, 255) in BGR
        # Note: User asked for "Opaque Red". 
        # If we want to see the underlying texture, we might want alpha blending, 
        # but "不透明" means opaque. However, usually in med-img vis, 
        # we want to see the structure. I will stick to the user's request for "Opaque".
        # But if it's truly opaque, it hides the green channel data under it.
        # I'll assume they want the mask to be the dominant color but maybe preserve luminance?
        # No, "不透明紅色" (Opaque Red) usually implies replacing the pixel color.
        vis_img[mask_img > 0] = [0, 0, 255] 
    else:
        print(f"Warning: Label file not found at {LABEL_PATH}")

    # 3. Crop the Image
    # Ensure crop is within bounds
    h, w = vis_img.shape[:2]
    x1 = max(0, CROP_X)
    y1 = max(0, CROP_Y)
    x2 = min(w, CROP_X + CROP_WIDTH)
    y2 = min(h, CROP_Y + CROP_HEIGHT)
    
    cropped_vis = vis_img[y1:y2, x1:x2].copy()
    
    # 4. Load Seeds and Skeleton Data
    topologies = load_seeds_info(SEEDS_PATH)
    
    # 5. Draw Skeleton (Yellow) and Seeds (Green)
    # We need to adjust coordinates by subtracting (CROP_X, CROP_Y)
    
    for comp in topologies:
        comp_id = comp['component_id']
        
        # Load Skeleton Mask for this component
        skeleton_path = f'output/all_components/skeleton_{comp_id}.png'
        if os.path.exists(skeleton_path):
            skeleton_mask = cv2.imread(skeleton_path, cv2.IMREAD_GRAYSCALE)
            
            # Find pixels in the skeleton mask
            # Note: The skeleton mask is the size of the original image (or at least relative to it)
            # We need to check if it overlaps with our crop
            
            # Get coordinates of skeleton pixels
            ys, xs = np.where(skeleton_mask > 0)
            
            for y, x in zip(ys, xs):
                # Check if pixel is within crop
                if y1 <= y < y2 and x1 <= x < x2:
                    # Adjust to crop coordinates
                    crop_y = y - y1
                    crop_x = x - x1
                    # Draw yellow pixel
                    cropped_vis[crop_y, crop_x] = [0, 255, 255]
        else:
            nodes = {n['id']: n['position'] for n in comp['nodes']}
            # Draw Edges (Skeleton) - Yellow
            for edge in comp['edges']:
                p1 = nodes[edge['source']]
                p2 = nodes[edge['target']]
                
                # Adjust coordinates
                pt1 = (int(p1['x'] - CROP_X), int(p1['y'] - CROP_Y))
                pt2 = (int(p2['x'] - CROP_X), int(p2['y'] - CROP_Y))
                
                # Check if at least one point is inside the crop (roughly)
                # or just draw and let OpenCV handle clipping
                cv2.line(cropped_vis, pt1, pt2, (0, 255, 255), 1) # Yellow in BGR
            
        # Draw Nodes (Seeds) - Green
        for n in comp['nodes']:
            pos = n['position']
            pt = (int(pos['x'] - CROP_X), int(pos['y'] - CROP_Y))
            
            # Draw all nodes as seeds? Or just specific types?
            # User said "Green represents seeds". Usually all nodes in the topology are seeds/points.
            # Endpoints are also seeds.
            cv2.circle(cropped_vis, pt, 1, (0, 255, 0), -1) # Green in BGR

    # 6. Save Output
    cv2.imwrite(OUTPUT_PATH, cropped_vis)
    print(f"Focused visualization saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

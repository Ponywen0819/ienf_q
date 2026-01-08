import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os

# Add project root to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.nueral_reconstruction.pathfinding import AStarPathfinder

def create_synthetic_data(width=600, height=400):
    # Create a dark background
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Define component positions
    start_pos = (200, 100)  # (y, x)
    end_pos = (200, 500)    # (y, x)
    
    # Draw components (bright circles)
    cv2.circle(image, (start_pos[1], start_pos[0]), 20, 255, -1)
    cv2.circle(image, (end_pos[1], end_pos[0]), 20, 255, -1)
    
    # Draw a "signal path" (faint bright strip) between them
    # This represents the nerve fiber
    cv2.line(image, (start_pos[1], start_pos[0]), (end_pos[1], end_pos[0]), 100, 15)
    
    # Add some noise/texture to make it look more realistic (optional)
    noise = np.random.randint(0, 30, (height, width), dtype=np.uint8)
    image = cv2.add(image, noise)
    
    return image, start_pos, end_pos

def generate_bad_path(start, end, height_offset=150):
    """Generates a curved path that goes through the dark area"""
    path = []
    y1, x1 = start
    y2, x2 = end
    
    # Simple parabolic path
    # y = a(x-h)^2 + k
    # Vertex at ((x1+x2)/2, y1 - height_offset)
    
    h = (x1 + x2) / 2
    k = y1 - height_offset
    
    # Fit 'a' to pass through start (x1, y1)
    # y1 = a(x1 - h)^2 + k
    # a = (y1 - k) / (x1 - h)^2
    
    if x1 == h: return [] # Should not happen
    
    a = (y1 - k) / ((x1 - h)**2)
    
    # Generate points
    step = 1 if x2 > x1 else -1
    for x in range(x1, x2 + step, step):
        y = int(a * (x - h)**2 + k)
        path.append((y, x))
        
    return path

def main():
    # 1. Create Data
    image, start, end = create_synthetic_data()
    
    # 2. Initialize Pathfinder
    pathfinder = AStarPathfinder(image)
    
    # 3. Find Good Path (Optimal)
    good_path = pathfinder.find_path(start, end)
    if not good_path:
        print("Failed to find good path!")
        return

    good_cost = pathfinder.calculate_path_cost(good_path)
    
    # 4. Generate Bad Path (Manual detour)
    bad_path = generate_bad_path(start, end, height_offset=100)
    
    # Filter bad path to be within bounds
    bad_path = [p for p in bad_path if 0 <= p[0] < image.shape[0] and 0 <= p[1] < image.shape[1]]
    
    bad_cost = pathfinder.calculate_path_cost(bad_path)
    
    # 5. Visualization
    plt.figure(figsize=(12, 8))
    
    # Show the cost map (inverted image) or the original image
    # Let's show the original image to show "Intensity"
    plt.imshow(image, cmap='gray')
    
    # Plot Good Path
    gy, gx = zip(*good_path)
    plt.plot(gx, gy, 'g-', linewidth=3, label=f'Good Path (Low Cost)\nCost: {good_cost:.1f}\nShort & Bright')
    
    # Plot Bad Path
    by, bx = zip(*bad_path)
    plt.plot(bx, by, 'r--', linewidth=3, label=f'Bad Path (High Cost)\nCost: {bad_cost:.1f}\nLong & Dark')
    
    # Plot Components
    plt.plot(start[1], start[0], 'bo', markersize=15, label='Component A')
    plt.plot(end[1], end[0], 'yo', markersize=15, label='Component B')
    
    # Annotations
    plt.title('Path Cost Comparison: Distance vs Intensity', fontsize=16)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12)
    plt.axis('off')
    
    # Add explanatory text on the image
    plt.text(300, 180, "High Intensity Signal\n(Low Cost Region)", color='white', ha='center', fontsize=10, alpha=0.7)
    plt.text(300, 50, "Dark Background\n(High Cost Region)", color='white', ha='center', fontsize=10, alpha=0.7)

    output_path = 'output/path_cost_comparison.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Visualization saved to {output_path}")
    
    # Show plot (optional, if running interactively)
    # plt.show()

if __name__ == "__main__":
    main()

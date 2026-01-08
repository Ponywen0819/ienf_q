import cv2
from preprocessing import SkinAnalysisPipeline  
import os
if __name__ == "__main__":
    # Example usage

    # Load example images (replace with actual paths)
    label_image = cv2.imread('/Users/ponywen/projects/ienf_q/data/Label/S163-2_a.tif', cv2.IMREAD_GRAYSCALE)
    epidermis_mask = cv2.imread('/Users/ponywen/projects/ienf_q/data/Mask/S163-2_a.tif', cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread('/Users/ponywen/projects/ienf_q/data/Original/S163-2_a_green.png', cv2.IMREAD_GRAYSCALE)

    # Define configuration
    config = {
        'morphology': {'closing_kernel': 5, 'opening_kernel': 3},
        'mask': {'dilate_offset': 100},
        'background': {
            'method': 'rolling_ball',
            'radius': 2,
            'light_background': False  # False for bright objects on dark background
        },
        'threshold': {'method': 'binary'},
        'normalization': {'enabled': True}
    }

    # Initialize and run pipeline
    pipeline = SkinAnalysisPipeline(config)
    final_label, roi_image = pipeline.run(label_image, epidermis_mask, original_image)

    # Save or display results as needed
    os.makedirs('output/preprocessing', exist_ok=True)
    cv2.imwrite('output/preprocessing/roi_image.png', roi_image)
    cv2.imwrite('output/preprocessing/final_label.png', final_label)
import cv2
from preprocessing import SkinAnalysisPipeline  
from neural_reconstruction import NeuralReconstructionPipeline
from neural_reconstruction.config_loader import load_config, IENFConfig

import logging
logging.basicConfig(level=logging.INFO)
import os
import argparse

if __name__ == "__main__":
    # Example usage
    parse = argparse.ArgumentParser(description="Run Skin Analysis and Neural Reconstruction Pipeline")
    parse.add_argument('--label_image', type=str, required=True, help='Path to the label image')
    parse.add_argument('--epidermis_mask', type=str, required=True, help='Path to the epidermis mask image')
    parse.add_argument('--original_image', type=str, required=True, help='Path to the original image')
    parse.add_argument('--config', type=str, help='Directory to save output results')
    parse.add_argument('--output_dir', type=str, default='output', help='Directory to save output results')
    args = parse.parse_args()
    

    # Load example images (replace with actual paths)
    label_image = cv2.imread(args.label_image, cv2.IMREAD_GRAYSCALE)
    epidermis_mask = cv2.imread(args.epidermis_mask, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(args.original_image, cv2.IMREAD_UNCHANGED)
    original_green_image = original_image[:, :, 1]
    
    config = {
        'morphology': {'closing_kernel': 3, 'opening_kernel': 3},
        'mask': {'dilate_offset': 100},
        'background': {
            'radius': 12,
            'light_background': False  # False for bright objects on dark background
        },
        'threshold': {'method': 'binary'}
    }

    reconstruct_config = load_config(args.config) if args.config else load_config()


    # Initialize and run pipeline
    pipeline = SkinAnalysisPipeline(config=config)
    final_label, roi_image = pipeline.run(label_image, epidermis_mask, original_green_image)

    # Initialize and run neural reconstruction pipeline
    neural_pipeline = NeuralReconstructionPipeline(config=reconstruct_config)
    neural_results = neural_pipeline.run(final_label, roi_image)

    mst_edges = neural_results['stages']['mst_reconstruction']['mst_with_paths']['edges']
    all_seeds = neural_results['stages']['topology_and_seeds']['seeds']
    all_topologies = neural_results['stages']['topology_and_seeds']['topologies']


    output_dir = args.output_dir
    # Save or display results as needed
    os.makedirs(output_dir, exist_ok=True)
    # cv2.imwrite(f'{output_dir}/roi_image.png', roi_image)
    # cv2.imwrite(f'{output_dir}/final_label.png', final_label)

    with open(f'{output_dir}/all_seeds.txt', 'w') as f:
        for seed in all_seeds:
            f.write(f"{seed}\n")


    # 寫入 
    with open(f'{output_dir}/mst_edges.txt', 'w') as f:
        for edge in mst_edges:
            f.write(f"{edge['path']}\n")

        for topo_info in all_topologies:
            topology = topo_info['topology']
            for edge in topology['edges']:
                f.write(f"{edge['path']}\n")


# ./script/run_pipeline.py --label_image /Users/ponywen/projects/ienf_q/data/Label/S163-2_a.tif --epidermis_mask /Users/ponywen/projects/ienf_q/data/Mask/S163-2_a.tif --original_image /Users/ponywen/projects/ienf_q/data/Original/S163-2_a.tif --output_dir ./exe_test
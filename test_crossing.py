#!/usr/bin/env python3
"""
Quick test script for boundary crossing detection

Tests the system with the first available image in the dataset.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main_crossing_analysis import main as crossing_main

if __name__ == '__main__':
    # Find first available image
    data_dir = project_root / 'data' / 'Original'
    image_files = list(data_dir.glob('*.tif'))

    if not image_files:
        print("ERROR: No .tif files found in data/Original/")
        sys.exit(1)

    # Use first image
    first_image = image_files[0]
    image_name = first_image.stem

    print(f"Testing with image: {image_name}")
    print(f"=" * 80)

    # Call main with single image argument
    sys.argv = [
        'test_crossing.py',
        '--single', image_name,
        '--rebuild-stats'  # Build statistics from scratch
    ]

    crossing_main()

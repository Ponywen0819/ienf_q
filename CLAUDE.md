# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IENF-Q (Intra-Epidermal Nerve Fiber Quantification) is an automated analysis pipeline that reconstructs complete neural fiber networks from microscopic images and sparse manual annotations. It uses classical computer vision algorithms (no ML) for interpretable, reproducible results.

## Build and Run Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run the full pipeline
python script/run_pipeline.py \
    --label_image /path/to/label.tif \
    --epidermis_mask /path/to/mask.tif \
    --original_image /path/to/original.tif \
    --output_dir output/results

# With custom config
python script/run_pipeline.py \
    --label_image ... --epidermis_mask ... --original_image ... \
    --config /path/to/config.yaml --output_dir output/results
```

## Architecture

### Two-Stage Pipeline

**Stage 1: Preprocessing** (`src/preprocessing/pipeline.py` - `SkinAnalysisPipeline`)
- Extracts green channel from RGB (strongest nerve fiber signal)
- Applies morphological operations to clean binary masks
- Performs rolling ball background correction
- Outputs: processed label mask, ROI image

**Stage 2: Neural Reconstruction** (`src/nueral_reconstruction/pipeline.py` - `NeuralReconstructionPipeline`)
1. Connected components analysis - identify discrete fiber segments
2. Skeletonization - extract centerlines using Zhang-Suen algorithm
3. Seed point extraction - curvature-aware placement (more seeds at bends/branches)
4. Component pairing - A* pathfinding on cost map to find connections
5. MST reconstruction - Kruskal's algorithm for optimal fiber network

### Key Modules

| Module | Purpose |
|--------|---------|
| `nueral_reconstruction/pathfinding.py` | A* algorithm using green channel as cost map |
| `nueral_reconstruction/seed_extraction.py` | Curvature-aware seed point placement |
| `nueral_reconstruction/mst_builder.py` | Minimum spanning tree construction |
| `nueral_reconstruction/config_loader.py` | Pydantic-based configuration validation |

### Configuration (config/default.yaml)

Key tunable parameters:
- `connected_components.connectivity`: 4 or 8 (default: 4)
- `seed_extraction.base_segment_length`: seed spacing in pixels (default: 5)
- `component_pairing.max_distance_threshold`: max connection distance (default: 50)
- `component_pairing.max_cost_threshold`: max acceptable path cost (default: 0.98)

## Code Conventions

- **Language**: Python 3.12+, code comments in Traditional Chinese
- **Typo note**: Module is named `nueral_reconstruction` (not "neural") - consistent throughout, do not rename
- **Image format**: Green channel carries strongest signal; annotations are binary (white=fiber, black=background)
- **Config priority**: CLI args > YAML config > defaults

## Data Flow

```
Input:
├── Original image (RGB) → green channel extraction
├── Label image (binary) → morphological cleaning
└── Epidermis mask → ROI definition

Output (in output_dir):
├── all_seeds.txt       # Seed point coordinates
├── mst_edges.txt       # Final reconstruction paths
└── seeds/seeds_overlay.png
```

## Visualization Tools

The `visualization/` folder contains standalone scripts for debugging each pipeline stage:
- `visualize_seeds.py` - seed point distribution
- `visualize_component_pairing.py` - connection analysis
- `visualize_mst_reconstruction.py` - final network overlay

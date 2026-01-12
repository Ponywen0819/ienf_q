# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IENF-Q (Intra-Epidermal Nerve Fiber Quantification) is an automated analysis pipeline that reconstructs complete neural fiber networks from microscopic images and sparse manual annotations. It uses classical computer vision algorithms (no ML) for interpretable, reproducible results.

## Build and Run Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run preprocessing pipeline
python tools/run_preprocessing.py \
    --label /path/to/label.tif \
    --mask /path/to/epidermis_mask.tif \
    --image /path/to/original.tif \
    --output-dir output/preprocessing

# With debug output to save intermediate results
python tools/run_preprocessing.py \
    --label data/Label/S163-2_a.tif \
    --mask data/Mask/S163-2_a.tif \
    --image data/Original/S163-2_a.tif \
    --output-dir output/preprocessing \
    --debug

# Run test script for neural reconstruction
python test_main_entry.py

# Run with uv
uv run python test_main_entry.py
```

## Architecture

The codebase is organized into a layered architecture with clear separation of concerns:

### Project Structure

```
src/neural_reconstruction/
├── common/              # Shared data types and utilities
│   └── data_types.py   # ComponentAnalysisResult, etc.
├── core/               # Core algorithms
│   ├── preprocessing/  # Image preprocessing pipeline
│   ├── construction/   # Neural network reconstruction
│   └── crosses_detection/  # Epidermis crossing detection
└── ui/                 # User interface (currently minimal)
```

### Main Pipeline: Neural Network Reconstruction

The reconstruction process is orchestrated by `build_neural_network()` in [src/neural_reconstruction/core/construction/main.py](src/neural_reconstruction/core/construction/main.py). This function provides a clean API that chains together four distinct phases:

**Phase 1: Connected Components Analysis**
- Uses scikit-image's `label()` and `regionprops()`
- Identifies discrete fiber segments from binary annotations
- Filters by minimum area to remove noise

**Phase 2: Component Analysis** (`component_analyzer/`)
- **Skeletonization**: Extracts centerlines using Zhang-Suen algorithm
- **Topology Construction**: Builds graph representation using `skan` library
- **Seed Extraction**: Curvature-aware placement along skeleton
  - More seeds at bends/branches to preserve fiber geometry
  - Controlled by `segment_length` parameter

**Phase 3: Connection Graph Building** (`connection_graph_builder/`)
- **Path Finding**: A* algorithm finds connections between components
- **Cost Calculation**: Multi-factor cost function:
  - `intensity_weight`: Prefers paths along bright pixels (nerve tissue)
  - `shape_weight`: Considers path smoothness and geometry
- **Graph Assembly**: Creates `NetworkX` graph with all feasible connections

**Phase 4: Backbone Extraction** (`backbone_extractor/`)
- **MST Construction**: Kruskal's algorithm for minimum spanning tree
- **Forest Output**: Produces optimal fiber network (may have multiple trees)

### Preprocessing Pipeline

Located in [src/neural_reconstruction/core/preprocessing/pipeline.py](src/neural_reconstruction/core/preprocessing/pipeline.py), the `SkinAnalysisPipeline` class handles:

1. **Green Channel Extraction**: Neural tissue has strongest signal in green channel
2. **Morphological Operations**: Opening/closing to clean binary masks
3. **Background Correction**: Rolling ball algorithm (radius configurable)
4. **Mask Operations**: ROI extraction using epidermis mask

### Crosses Detection Module

The `crosses_detection/` module counts nerve fibers crossing the epidermis boundary:
- `segment_detector.py`: Identifies fiber segments
- `region_labeler.py`: Labels epidermis regions
- `crossing_counter.py`: Counts crossings for quantification

## Key API Entry Points

### Main Reconstruction Function

```python
from neural_reconstruction.core.construction.main import build_neural_network

# Minimal usage
mst_forest = build_neural_network(
    label_image=binary_label,      # np.ndarray (H, W) binary
    green_channel=green_channel,   # np.ndarray (H, W) uint8
)

# With all parameters
mst_forest = build_neural_network(
    label_image=binary_label,
    green_channel=green_channel,
    connectivity=4,                # 4 or 8
    min_area=50,                  # Filter small components
    segment_length=5.0,           # Seed spacing (pixels)
    search_radius=50.0,           # Max connection distance
    max_cost_threshold=0.98,      # Path cost cutoff (0-1)
    intensity_weight=0.6,         # Path cost weights
    shape_weight=0.4,
)
# Returns: NetworkX Graph with nodes=(y,x) coords, edges with 'path' attribute
```

### Preprocessing Pipeline

```python
from neural_reconstruction.core.preprocessing.pipeline import SkinAnalysisPipeline

config = {
    'morphology': {'closing_kernel': 3, 'opening_kernel': 3},
    'background': {'radius': 12, 'light_background': True},
    'mask': {'dilate_offset': 50}
}

pipeline = SkinAnalysisPipeline(config)
final_label, roi_image = pipeline.run(
    label_img,      # Binary annotation
    mask_img,       # Epidermis mask
    orig_img,       # Original RGB/grayscale
    debug=False
)
```

## Key Configuration Parameters

### Component Analysis
- `connectivity`: 4 or 8 (default: 4) - affects connected component detection
- `min_area`: Minimum pixels to keep component (default: 0)
- `segment_length`: Seed spacing along skeleton in pixels (default: 5.0)
- `prune_threshold`: Remove skeleton branches shorter than this (default: 5.0)

### Path Finding
- `search_radius`: Max distance to search for connections in pixels (default: 50.0)
- `max_cost_threshold`: Reject paths with normalized cost > this value (default: 0.98)
- `intensity_weight`: Weight for image intensity in cost (default: 0.6)
- `shape_weight`: Weight for path geometry in cost (default: 0.4)

## Code Conventions

- **Language**: Python 3.10+, code comments in Traditional Chinese
- **Module naming**: The package is `neural_reconstruction` (note: some legacy references may say `nueral_reconstruction` - when updating code, use `neural_reconstruction`)
- **Image format**:
  - Green channel carries strongest nerve fiber signal
  - Annotations are binary (255=fiber, 0=background)
  - Internal processing uses 0/1 binary after conversion
- **Data structures**:
  - Uses `NetworkX` for graph representations
  - Custom dataclasses defined in `common/data_types.py`
  - NumPy arrays for all image data

## Package Management

This project uses **uv** as the package manager:
- Dependencies defined in [pyproject.toml](pyproject.toml)
- Requires Python >=3.10
- Key dependencies: opencv-python, scikit-image, networkx, skan, pyyaml

## Development Notes

- **No config files**: Despite documentation references, there are no `config/` directory or YAML files in the repo currently. Configuration is passed programmatically or via CLI args.
- **Entry points**: Main development entry points are:
  - [test_main_entry.py](test_main_entry.py) - Tests reconstruction pipeline
  - [tools/run_preprocessing.py](tools/run_preprocessing.py) - CLI for preprocessing
- **Data directory**: [data/](data/) contains sample images for testing (Original/, Label/, Mask/ subdirectories)
- **Output**: Pipeline generates NetworkX graphs; visualization is handled externally

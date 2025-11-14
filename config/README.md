# IENF Pipeline Configuration Profiles

This directory contains YAML configuration files for the IENF quantification pipeline. Each profile is optimized for different use cases.

## Available Profiles

### `default.yaml`
**Balanced profile for general use**

- Good balance between quality and processing speed
- Recommended for most use cases
- Default parameters validated on test datasets

**Key Settings:**
- Window size: 5
- Curvature threshold: 30°
- Cost weights: α=0.05, β=0.9, γ=0.05
- Max edge cost: 150

**Use when:**
- Processing standard IENF images
- You need reliable results without excessive processing time
- You're unsure which profile to use

---

### `high_quality.yaml`
**Premium quality profile for publication and final analysis**

- Prioritizes reconstruction accuracy
- More granular seed placement and stricter validation
- Slower processing but higher quality results

**Key Settings:**
- Window size: 7 (more precise curvature)
- Curvature threshold: 25° (more sensitive)
- Cost weights: α=0.03, β=0.92, γ=0.05 (stronger image adherence)
- Max edge cost: 150 (strict final selection)

**Use when:**
- Preparing results for publication
- Maximum accuracy is critical
- Processing time is not a constraint
- Working with high-quality, well-stained images

---

### `fast.yaml`
**Speed-optimized profile for rapid prototyping**

- Prioritizes processing speed
- Fewer seeds and simpler pathfinding
- Good for quick iterations and testing

**Key Settings:**
- Window size: 3 (faster processing)
- Curvature threshold: 35° (less sensitive)
- Cost weights: α=0.1, β=0.85, γ=0.05 (simpler paths)
- Max edge cost: 140 (more permissive)

**Use when:**
- Testing new datasets
- Quick quality checks
- Developing and debugging
- Working with low-quality or preliminary images

---

## Using Configuration Files

### Command Line Usage

```bash
# Use default configuration
python run_pipeline.py --config config/default.yaml \
    --input-skeletons data/skeletons \
    --image data/green_channel.png \
    --output output

# Use high quality profile
python run_pipeline.py --config config/high_quality.yaml \
    --input-skeletons data/skeletons \
    --image data/green_channel.png \
    --output output

# Use fast profile
python run_pipeline.py --config config/fast.yaml \
    --input-skeletons data/skeletons \
    --image data/green_channel.png \
    --output output
```

### Individual Stage Usage

Each stage can also use the configuration file:

```bash
# Stage 02 - Seed Extraction
python src/02_seed_extraction/seed_extraction.py \
    --config config/high_quality.yaml \
    -i output/skeletons \
    -o output/seeds

# Stage 03 - Network Building
python src/03_network_building/run_network_building.py \
    --config config/default.yaml \
    -s output/seeds/seeds.json \
    -i test/green_channel.png \
    -o output/network

# Stage 04 - Reconstruction
python src/04_nueral_reconstruction/run_reconstruction.py \
    --config config/default.yaml \
    --graph output/network/network.graphml \
    --seeds output/seeds/seeds.json \
    --image test/green_channel.png \
    --output output/reconstruction
```

### Override Parameters

You can override any parameter from the command line:

```bash
# Override curvature threshold
python run_pipeline.py --config config/default.yaml \
    --curvature-threshold 28 \
    --input-skeletons data/skeletons \
    --image data/green_channel.png

# Override max edge cost
python run_pipeline.py --config config/high_quality.yaml \
    --max-edge-cost 160 \
    --input-skeletons data/skeletons \
    --image data/green_channel.png
```

---

## Creating Custom Profiles

To create a custom profile:

1. Copy an existing profile:
   ```bash
   cp config/default.yaml config/my_custom.yaml
   ```

2. Edit the parameters in `my_custom.yaml`

3. Use your custom profile:
   ```bash
   python run_pipeline.py --config config/my_custom.yaml ...
   ```

---

## Configuration Parameters Reference

### Stage 02: Seed Extraction

| Parameter | Description | Default | Fast | High Quality |
|-----------|-------------|---------|------|--------------|
| `window_size` | Curvature calculation window (must be odd) | 5 | 3 | 7 |
| `base_segment_length` | Base segment length (pixels) | 10.0 | 15.0 | 8.0 |
| `max_segment_length` | Maximum segment length (pixels) | 20.0 | 25.0 | 15.0 |
| `curvature_threshold` | Curvature trigger threshold (degrees) | 30.0 | 35.0 | 25.0 |
| `skip_branchpoint_range` | Skip range around branch points | 5 | 3 | 7 |
| `min_path_points` | Minimum path points for processing | 10 | 10 | 10 |

### Stage 03: Network Building

| Parameter | Description | Default | Fast | High Quality |
|-----------|-------------|---------|------|--------------|
| `k_neighbors` | Neighbors for density estimation | 10 | 8 | 15 |
| `max_edge_cost` | Maximum edge cost threshold | 150.0 | 120.0 | 180.0 |
| `alpha` | Geometric cost weight | 0.05 | 0.1 | 0.03 |
| `beta` | Image cost weight | 0.9 | 0.85 | 0.92 |
| `gamma` | Curvature cost weight | 0.05 | 0.05 | 0.05 |
| `dense_threshold` | Dense region threshold (pixels) | 30 | 35 | 25 |
| `moderate_threshold` | Moderate region threshold (pixels) | 70 | 80 | 60 |
| `max_distance_multiplier` | Pathfinding distance multiplier | 200 | 150 | 250 |

### Stage 04: Reconstruction

| Parameter | Description | Default | Fast | High Quality |
|-----------|-------------|---------|------|--------------|
| `max_edge_cost` | MST edge cost threshold | 150 | 140 | 150 |
| `min_branch_angle` | Sharp branch angle threshold (degrees) | 30 | 25 | 35 |
| `min_quality_threshold` | Minimum path quality | 80 | 75 | 85 |

---

## Environment Variables

Configuration files support environment variable substitution:

```yaml
pipeline:
  paths:
    data_dir: "${DATA_DIR:data}"  # Uses $DATA_DIR, defaults to "data"
    output_dir: "${OUTPUT_DIR}"   # Uses $OUTPUT_DIR
```

Usage:
```bash
export DATA_DIR=/mnt/research/ienf/data
export OUTPUT_DIR=/mnt/research/ienf/output
python run_pipeline.py --config config/default.yaml ...
```

---

## Tips for Parameter Tuning

### Improving Quality

1. **Increase seed granularity:**
   - Decrease `base_segment_length` (e.g., 8.0)
   - Decrease `curvature_threshold` (e.g., 25°)

2. **Improve path quality:**
   - Increase `beta` (image weight) (e.g., 0.92)
   - Increase `max_distance_multiplier` (e.g., 250)

3. **Stricter reconstruction:**
   - Decrease `max_edge_cost` for stage 04 (e.g., 140)
   - Increase `min_quality_threshold` (e.g., 85)

### Improving Speed

1. **Reduce seed count:**
   - Increase `base_segment_length` (e.g., 15.0)
   - Increase `curvature_threshold` (e.g., 35°)

2. **Simplify pathfinding:**
   - Decrease `max_distance_multiplier` (e.g., 150)
   - Increase `alpha` (geometric weight) (e.g., 0.1)

3. **Reduce graph complexity:**
   - Decrease `k_neighbors` (e.g., 8)
   - Decrease `max_edge_cost` for stage 03 (e.g., 120)

### Handling Specific Issues

- **Too many false connections:**
  - Decrease `max_edge_cost`
  - Increase `curvature_threshold`

- **Missing fiber segments:**
  - Increase `max_edge_cost`
  - Decrease `curvature_threshold`
  - Increase `max_distance_multiplier`

- **Overly straight reconstructions:**
  - Decrease `alpha` (geometric weight)
  - Increase `beta` (image weight)

---

## Logging

Each profile specifies logging configuration:

```yaml
pipeline:
  logging:
    level: "INFO"              # DEBUG, INFO, WARNING, ERROR
    log_to_file: true
    log_file: "output/pipeline.log"
```

Log files contain detailed execution information for debugging and analysis.

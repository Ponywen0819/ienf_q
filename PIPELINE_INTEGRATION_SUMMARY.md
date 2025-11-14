# IENF Pipeline Integration Summary

## Overview

I have successfully integrated stages 02, 03, and 04 of your IENF quantification pipeline with a comprehensive YAML-based configuration system. The integration provides centralized parameter management, multiple configuration profiles, and a master orchestration script for running the complete pipeline.

---

## What Was Done

### 1. Configuration System ([src/config_loader.py](src/config_loader.py))

Created a robust configuration management system using Pydantic for type-safe validation:

**Features:**
- ✅ Type-safe configuration with automatic validation
- ✅ Environment variable substitution (`${VAR_NAME}` or `${VAR_NAME:default}`)
- ✅ Support for configuration overrides (YAML → CLI hierarchy)
- ✅ Nested configuration structure matching pipeline stages
- ✅ Comprehensive parameter validation (e.g., window_size must be odd, weights must sum to 1.0)

**Configuration Classes:**
- `SeedExtractionConfig` - Stage 02 parameters
- `NetworkBuildingConfig` - Stage 03 parameters (network, cost_weights, density, pathfinding)
- `ReconstructionConfig` - Stage 04 parameters
- `PipelineConfig` - Pipeline orchestration settings
- `IENFConfig` - Root configuration containing all stages

---

### 2. Configuration Files

#### [config/default.yaml](config/default.yaml)
**Balanced profile for general use**

All 18+ hyperparameters organized by stage:
- Stage 02: 6 parameters (window_size, segment lengths, curvature threshold, etc.)
- Stage 03: 12 parameters (k-neighbors, cost weights α/β/γ, density thresholds, pathfinding)
- Stage 04: 4 parameters (max_edge_cost, validation thresholds)
- Pipeline: paths, stage flags, logging

#### [config/high_quality.yaml](config/high_quality.yaml)
**Premium quality profile for publication**

Optimizations:
- More granular seeds (window_size=7, base_length=8)
- More sensitive curvature detection (threshold=25°)
- Stronger image adherence (beta=0.92)
- Stricter validation (quality_threshold=85)

#### [config/fast.yaml](config/fast.yaml)
**Speed-optimized for rapid prototyping**

Optimizations:
- Fewer seeds (window_size=3, base_length=15)
- Simpler paths (alpha=0.1, beta=0.85)
- Relaxed thresholds (max_cost=140)
- Less intensive validation

#### [config/README.md](config/README.md)
Comprehensive documentation with:
- Profile comparison table
- Usage examples
- Parameter tuning guide
- Troubleshooting tips

---

### 3. Updated Stage Scripts

#### [src/02_seed_extraction/seed_extraction.py](src/02_seed_extraction/seed_extraction.py:1258-1343)

**Changes:**
- Added `--config` argument for YAML file
- Loads configuration using `load_config()`
- CLI arguments override YAML values
- Maintains backward compatibility (works without config file)

**Usage:**
```bash
# With YAML config
python src/02_seed_extraction/seed_extraction.py \
    --config config/high_quality.yaml \
    -i output/skeletons \
    -o output/seeds

# Override specific parameters
python src/02_seed_extraction/seed_extraction.py \
    --config config/default.yaml \
    --curvature-threshold 25 \
    -i output/skeletons \
    -o output/seeds
```

#### [src/03_network_building/run_network_building.py](src/03_network_building/run_network_building.py:113-206)

**Changes:**
- Added `--config` argument
- Added CLI arguments for cost weights (--alpha, --beta, --gamma)
- Loads and applies all network building parameters from YAML
- Properly propagates config to density estimator and pathfinder

**Usage:**
```bash
# With YAML config
python src/03_network_building/run_network_building.py \
    --config config/fast.yaml \
    -s output/seeds/seeds.json \
    -i test/green_channel.png \
    -o output/network

# Override cost weights
python src/03_network_building/run_network_building.py \
    --config config/default.yaml \
    --beta 0.95 \
    -s output/seeds/seeds.json \
    -i test/green_channel.png \
    -o output/network
```

#### [src/04_nueral_reconstruction/run_reconstruction.py](src/04_nueral_reconstruction/run_reconstruction.py:121-186)

**Changes:**
- Added `--config` argument
- Loads reconstruction parameters from YAML
- CLI arguments override config values
- Maintains all existing functionality

**Usage:**
```bash
# With YAML config
python src/04_nueral_reconstruction/run_reconstruction.py \
    --config config/high_quality.yaml \
    --graph output/network/network.graphml \
    --seeds output/seeds/seeds.json \
    --image test/green_channel.png \
    --output output/reconstruction
```

---

### 4. Master Pipeline Orchestrator ([run_pipeline.py](run_pipeline.py))

**New unified entry point** for running the complete pipeline from seed extraction to reconstruction.

**Features:**
- ✅ Single command to run entire pipeline
- ✅ Intelligent stage skipping (reuse existing outputs)
- ✅ Centralized logging with file output
- ✅ Progress tracking and timing
- ✅ Automatic directory management
- ✅ Comprehensive error handling
- ✅ Stage-by-stage execution control

**Usage Examples:**

```bash
# Run full pipeline with default config
python run_pipeline.py \
    --config config/default.yaml \
    --input-skeletons data/skeletons \
    --image data/green_channel.png \
    --output output \
    --visualize

# Run with high quality profile
python run_pipeline.py \
    --config config/high_quality.yaml \
    --input-skeletons data/skeletons \
    --image data/green_channel.png \
    --output output

# Skip seed extraction (reuse existing)
python run_pipeline.py \
    --config config/default.yaml \
    --skip-seed-extraction \
    --image data/green_channel.png \
    --output output

# Override parameters
python run_pipeline.py \
    --config config/default.yaml \
    --input-skeletons data/skeletons \
    --image data/green_channel.png \
    --curvature-threshold 28 \
    --max-edge-cost 160
```

---

## File Structure

```
ienf_q/
├── config/
│   ├── default.yaml           # Balanced profile
│   ├── high_quality.yaml      # Premium quality profile
│   ├── fast.yaml              # Speed-optimized profile
│   └── README.md              # Configuration documentation
│
├── src/
│   ├── config_loader.py       # Configuration management system
│   │
│   ├── 02_seed_extraction/
│   │   └── seed_extraction.py # Updated with YAML support
│   │
│   ├── 03_network_building/
│   │   └── run_network_building.py # Updated with YAML support
│   │
│   └── 04_nueral_reconstruction/
│       └── run_reconstruction.py   # Updated with YAML support
│
├── run_pipeline.py            # Master orchestrator (NEW)
└── PIPELINE_INTEGRATION_SUMMARY.md # This file
```

---

## Configuration Hierarchy

The system respects the following priority order (highest to lowest):

1. **CLI Arguments** - Highest priority
2. **YAML Configuration File** - Medium priority
3. **Default Values in Code** - Lowest priority

Example:
```bash
# curvature_threshold will be 28 (CLI override)
python run_pipeline.py \
    --config config/default.yaml \  # Has threshold=30
    --curvature-threshold 28 \      # Overrides to 28
    --input-skeletons data/skeletons \
    --image data/green_channel.png
```

---

## All Configurable Parameters

### Stage 02: Seed Extraction (6 parameters)
| Parameter | Description | Default |
|-----------|-------------|---------|
| `window_size` | Curvature calculation window (must be odd) | 5 |
| `base_segment_length` | Base segmentation length (pixels) | 10.0 |
| `max_segment_length` | Maximum segment length (pixels) | 20.0 |
| `curvature_threshold` | Curvature trigger threshold (degrees) | 30.0 |
| `skip_branchpoint_range` | Skip range around branch points | 5 |
| `min_path_points` | Minimum path points for processing | 10 |

### Stage 03: Network Building (12 parameters)

**Network Config:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `k_neighbors` | Neighbors for density estimation | 10 |
| `max_edge_cost` | Maximum edge cost threshold | 150.0 |
| `verbose` | Display detailed information | false |

**Cost Weights:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `alpha` | Geometric cost weight | 0.05 |
| `beta` | Image cost weight | 0.9 |
| `gamma` | Curvature cost weight | 0.05 |

**Density-Based Pairing:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `dense_threshold` | Dense region threshold (pixels) | 30 |
| `moderate_threshold` | Moderate region threshold (pixels) | 70 |
| `dense_radius` | Pairing radius in dense regions | 30 |
| `moderate_radius` | Pairing radius in moderate regions | 50 |
| `sparse_radius` | Pairing radius in sparse regions | 80 |

**Pathfinding:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_distance_multiplier` | Max search distance multiplier | 200 |
| `distance_from_start_cutoff` | Early termination distance | 30 |

### Stage 04: Reconstruction (4 parameters)
| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_edge_cost` | MST edge cost threshold | 150 |
| `min_branch_angle` | Sharp branch angle threshold (degrees) | 30 |
| `min_quality_threshold` | Minimum path quality | 80 |
| `verbose` | Display detailed progress | true |

**Total: 22 configurable hyperparameters**

---

## Key Features

### 1. Type Safety with Pydantic
```python
# Automatic validation
config = load_config('config/default.yaml')

# This will raise ValidationError
config.seed_extraction.window_size = 4  # Must be odd!

# This will raise ValidationError
config.network_building.cost_weights.alpha = 1.5  # Must be <= 1.0
```

### 2. Environment Variables
```yaml
pipeline:
  paths:
    data_dir: "${DATA_DIR:data}"
    output_dir: "${OUTPUT_DIR:output}"
```

```bash
export DATA_DIR=/mnt/research/ienf/data
python run_pipeline.py --config config/default.yaml ...
```

### 3. Configuration Saving
```python
from src.config_loader import load_config, save_config

# Load and modify
config = load_config('config/default.yaml')
config.seed_extraction.curvature_threshold = 28

# Save as new profile
save_config(config, 'config/my_custom.yaml')
```

### 4. Programmatic Access
```python
from src.config_loader import load_config

config = load_config('config/high_quality.yaml')

print(f"Curvature threshold: {config.seed_extraction.curvature_threshold}")
print(f"Cost weights: α={config.network_building.cost_weights.alpha}")
```

---

## Migration Guide

### Before (Hardcoded Parameters)
```bash
python src/02_seed_extraction/seed_extraction.py \
    -i output/skeletons \
    -o output/seeds \
    --window-size 5 \
    --base-length 10 \
    --max-length 20 \
    --curvature-threshold 30

python src/03_network_building/run_network_building.py \
    -s output/seeds/seeds.json \
    -i test/green_channel.png \
    -o output/network \
    --max-edge-cost 150 \
    --k-neighbors 10

python src/04_nueral_reconstruction/run_reconstruction.py \
    --graph output/network/network.graphml \
    --seeds output/seeds/seeds.json \
    --image test/green_channel.png \
    --output output/reconstruction \
    --max-cost 150
```

### After (YAML Configuration)
```bash
# Option 1: Use master orchestrator (RECOMMENDED)
python run_pipeline.py \
    --config config/default.yaml \
    --input-skeletons output/skeletons \
    --image test/green_channel.png \
    --output output

# Option 2: Individual stages with shared config
python src/02_seed_extraction/seed_extraction.py \
    --config config/default.yaml \
    -i output/skeletons \
    -o output/seeds

python src/03_network_building/run_network_building.py \
    --config config/default.yaml \
    -s output/seeds/seeds.json \
    -i test/green_channel.png \
    -o output/network

python src/04_nueral_reconstruction/run_reconstruction.py \
    --config config/default.yaml \
    --graph output/network/network.graphml \
    --seeds output/seeds/seeds.json \
    --image test/green_channel.png \
    --output output/reconstruction
```

---

## Benefits

1. **Centralized Management**: All parameters in one YAML file
2. **Reproducibility**: Configuration files can be version controlled
3. **Flexibility**: Multiple profiles for different use cases
4. **Validation**: Automatic parameter checking prevents errors
5. **Documentation**: Self-documenting configuration with comments
6. **Extensibility**: Easy to add new parameters
7. **Backward Compatible**: Existing CLI usage still works
8. **Type Safety**: Pydantic ensures correct types and ranges

---

## Next Steps

### Recommended Usage

1. **Start with default profile:**
   ```bash
   python run_pipeline.py \
       --config config/default.yaml \
       --input-skeletons data/skeletons \
       --image data/green_channel.png \
       --output output
   ```

2. **If quality is insufficient:**
   ```bash
   python run_pipeline.py \
       --config config/high_quality.yaml \
       --input-skeletons data/skeletons \
       --image data/green_channel.png \
       --output output
   ```

3. **For quick iterations:**
   ```bash
   python run_pipeline.py \
       --config config/fast.yaml \
       --input-skeletons data/skeletons \
       --image data/green_channel.png \
       --output output
   ```

4. **Fine-tune parameters:**
   - Copy a profile: `cp config/default.yaml config/my_profile.yaml`
   - Edit parameters based on results
   - Test: `python run_pipeline.py --config config/my_profile.yaml ...`

### Testing the Integration

```bash
# Test configuration loading
python -c "from src.config_loader import load_config; \
           config = load_config('config/default.yaml'); \
           print('Config loaded successfully!')"

# Test parameter validation
python -c "from src.config_loader import SeedExtractionConfig; \
           config = SeedExtractionConfig(window_size=7); \
           print(f'Window size: {config.window_size}')"
```

---

## Troubleshooting

### Issue: "Module not found: config_loader"
**Solution:** Ensure you're running from the project root directory:
```bash
cd /Users/ponywen/projects/ienf_q
python run_pipeline.py ...
```

### Issue: "window_size must be odd"
**Solution:** Check your YAML file or CLI override:
```yaml
seed_extraction:
  window_size: 7  # Must be odd (3, 5, 7, 9, etc.)
```

### Issue: "Cost weights should sum to ~1.0"
**Solution:** Ensure alpha + beta + gamma ≈ 1.0:
```yaml
cost_weights:
  alpha: 0.05
  beta: 0.9
  gamma: 0.05  # Sum = 1.0
```

---

## Summary

✅ **Complete integration achieved:**
- Unified YAML configuration system
- Three profiles (default, high_quality, fast)
- Master pipeline orchestrator
- All 22 hyperparameters configurable
- Type-safe validation
- CLI override support
- Comprehensive documentation

Your pipeline now has professional-grade configuration management that makes it easy to:
- Switch between quality/speed profiles
- Reproduce results with version-controlled configs
- Fine-tune parameters for specific datasets
- Run the complete pipeline with a single command

---

**Happy reconstructing! 🧬🔬**

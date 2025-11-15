"""
IENF Quantification Pipeline

Intraepidermal Nerve Fiber (IENF) quantification system with
automated neural reconstruction and boundary crossing detection.

Modules:
- shared: Common utilities (pathfinding, etc.)
- seed_extraction: Extract seed points from skeletonized images
- network_building: Build neural network from seeds
- neural_reconstruction: MST-based reconstruction
- boundary_crossing: Epidermis-dermis connection
"""

__version__ = "0.1.0"

"""
Boundary Crossing Module - Epidermis-Dermis Neural Connection

This module connects epidermis and dermis nerve reconstructions by:
1. Identifying boundary-proximal nodes in each component
2. Computing optimal connections using A* pathfinding
3. Matching epidermis components to dermis components
4. Merging MST forests with crossing edges
"""

from .boundary_connector import BoundaryConnector
from .boundary_detector import BoundaryDetector
from .component_analyzer import ComponentAnalyzer
from .connection_optimizer import ConnectionOptimizer
from .forest_merger import ForestMerger

__all__ = [
    'BoundaryConnector',
    'BoundaryDetector',
    'ComponentAnalyzer',
    'ConnectionOptimizer',
    'ForestMerger',
]

"""
Shared Utilities Module

Common utilities used across multiple pipeline stages.

Available modules:
- pathfinding: A* pathfinding on image cost maps
"""

from .pathfinding import ImagePathfinder

__all__ = [
    'ImagePathfinder',
]

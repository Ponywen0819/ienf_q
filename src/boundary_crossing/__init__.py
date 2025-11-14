"""
Boundary Crossing Analysis Module

This module provides functionality to detect and analyze nerve fibers
crossing the epidermis-dermis boundary.
"""

from .config import CROSSING_CONFIG
from .statistics_builder import EpidermisStatisticsBuilder
from .boundary_detector import BoundaryDetector
from .crossing_analyzer import CrossingAnalyzer
from .visualizer import CrossingVisualizer

__all__ = [
    'CROSSING_CONFIG',
    'EpidermisStatisticsBuilder',
    'BoundaryDetector',
    'CrossingAnalyzer',
    'CrossingVisualizer',
]

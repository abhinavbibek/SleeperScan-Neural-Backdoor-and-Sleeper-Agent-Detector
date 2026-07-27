"""heuristics: detection algorithms for structural backdoor anomalies."""

from sleeperscan.heuristics.bait import BAITScanner
from sleeperscan.heuristics.double_triangle import DoubleTriangleDetector

__all__ = ["BAITScanner", "DoubleTriangleDetector"]



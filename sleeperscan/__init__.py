"""sleeperscan - neural backdoor and sleeper agent detection for open-weight LLMs."""

__version__ = "0.1.0"
__author__ = "Abhinav Bibek"

from sleeperscan.core import AttentionHookManager, EntropyMonitor
from sleeperscan.heuristics import DoubleTriangleDetector

__all__ = ["AttentionHookManager", "EntropyMonitor", "DoubleTriangleDetector"]


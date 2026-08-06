"""core infrastructure: model hooks, entropy metrics, bait target inversion, and memory extraction."""

from sleeperscan.core.bait_inverter import TargetInverter
from sleeperscan.core.hooks import AttentionHookManager
from sleeperscan.core.memory_extractor import MemoryExtractor
from sleeperscan.core.metrics import EntropyMonitor

__all__ = ["AttentionHookManager", "EntropyMonitor", "MemoryExtractor", "TargetInverter"]

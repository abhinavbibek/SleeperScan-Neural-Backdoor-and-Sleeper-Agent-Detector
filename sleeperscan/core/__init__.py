"""core infrastructure: model hooks and metric computation."""

from sleeperscan.core.hooks import AttentionHookManager
from sleeperscan.core.metrics import EntropyMonitor

__all__ = ["AttentionHookManager", "EntropyMonitor"]

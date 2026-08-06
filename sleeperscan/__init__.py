"""sleeperscan - neural backdoor and sleeper agent detection for open-weight LLMs.

this package provides inference-only tools to detect sleeper agent backdoors in
huggingface-compatible open-weight language models, without requiring model retraining
or prior knowledge of the trigger phrase.

detection methodology synthesizes three research directions:

  bait (S&P 2025):
    Shen et al., "BAIT: Large Language Model Backdoor ScAnning by Inverting Attack
    Target." IEEE Symposium on Security and Privacy, 2025.
    https://doi.org/10.1109/SP61157.2025.00073

  double triangle attention pattern:
    Kumar et al., "The Trigger in the Haystack: Extracting and Reconstructing LLM
    Backdoor Triggers." arXiv:2602.03085, 2026.
    https://arxiv.org/abs/2602.03085

  sleeper agents (model organisms):
    Hubinger et al., "Sleeper Agents: Training Deceptive LLMs that Persist Through
    Safety Training." arXiv:2401.05566, 2024.
    https://arxiv.org/abs/2401.05566
"""

__version__ = "0.2.0"
__author__ = "Abhinav Bibek"

from sleeperscan.core.bait_inverter import TargetInverter
from sleeperscan.core.hooks import AttentionHookManager
from sleeperscan.core.memory_extractor import MemoryExtractor
from sleeperscan.core.metrics import EntropyMonitor
from sleeperscan.heuristics.bait import BAITScanner
from sleeperscan.heuristics.double_triangle import DoubleTriangleDetector
from sleeperscan.heuristics.semantic_drift import SemanticDriftEvaluator

__all__ = [
    "AttentionHookManager",
    "BAITScanner",
    "DoubleTriangleDetector",
    "EntropyMonitor",
    "MemoryExtractor",
    "SemanticDriftEvaluator",
    "TargetInverter",
]

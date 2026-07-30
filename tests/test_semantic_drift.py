"""test_semantic_drift.py - unit tests for SemanticDriftDetector using synthetic hidden states.

verifies that SemanticDriftDetector correctly calculates cosine distances and evaluates
representation-space anomalies.
"""

from typing import Dict
import pytest
import torch
from sleeperscan.heuristics.semantic_drift import SemanticDriftDetector


@pytest.fixture
def detector() -> SemanticDriftDetector:
    return SemanticDriftDetector(drift_threshold=0.6)


# cosine distance tests

class TestComputeCosineDistance:

    def test_identical_states_have_zero_distance(self, detector: SemanticDriftDetector) -> None:
        """cosine distance between identical states must be zero."""
        seq_len, hidden_dim = 10, 128
        state = torch.randn(seq_len, hidden_dim)
        dist = detector.compute_cosine_distance(state, state)
        assert abs(dist) < 1e-5, f"expected near-zero distance, got {dist:.6f}"

    def test_opposite_states_have_maximum_distance(self, detector: SemanticDriftDetector) -> None:
        """cosine distance between exactly opposite states must be 2.0."""
        seq_len, hidden_dim = 5, 64
        state = torch.randn(seq_len, hidden_dim)
        dist = detector.compute_cosine_distance(state, -state)
        assert abs(dist - 2.0) < 1e-5, f"expected distance ~2.0, got {dist:.6f}"


# evaluate drift tests

class TestEvaluateDrift:

    def test_clean_model_has_low_drift(self, detector: SemanticDriftDetector) -> None:
        """evaluates drift when clean and triggered states are nearly identical."""
        seq_len, hidden_dim = 8, 128
        clean_state = torch.randn(seq_len, hidden_dim)
        # add minimal noise
        triggered_state = clean_state + torch.randn(seq_len, hidden_dim) * 0.01

        clean_states: Dict[int, torch.Tensor] = {0: clean_state}
        triggered_states: Dict[int, torch.Tensor] = {0: triggered_state}

        result = detector.evaluate_drift(clean_states, triggered_states)
        assert result["is_drift_anomalous"] is False
        assert result["max_drift"] < 0.1

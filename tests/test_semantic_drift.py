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

    def test_shape_mismatch_raises(self, detector: SemanticDriftDetector) -> None:
        """mismatched hidden state shapes must raise a ValueError."""
        clean = torch.randn(8, 128)
        triggered = torch.randn(8, 64)
        with pytest.raises(ValueError, match="shape mismatch"):
            detector.compute_cosine_distance(clean, triggered)

    def test_invalid_dimension_raises(self, detector: SemanticDriftDetector) -> None:
        """tensors with invalid dimensions must raise a ValueError."""
        clean = torch.randn(8)
        triggered = torch.randn(8)
        with pytest.raises(ValueError, match="must be 2D or 3D"):
            detector.compute_cosine_distance(clean, triggered)


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

    def test_empty_states_returns_empty_evaluation(self, detector: SemanticDriftDetector) -> None:
        """evaluating empty states dictionaries should return a default false result."""
        result = detector.evaluate_drift({}, {})
        assert result["is_drift_anomalous"] is False
        assert result["max_drift"] == 0.0
        assert result["critical_layer"] == -1
        assert result["layer_drifts"] == {}

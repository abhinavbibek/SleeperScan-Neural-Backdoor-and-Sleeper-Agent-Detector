"""test_metrics.py - unit tests for EntropyMonitor using synthetic logit tensors.

all expected values are computed analytically from the known input distributions
so the tests act as mathematical ground truth, not just smoke tests.
"""

import math
import torch
import pytest
from sleeperscan.core.metrics import EntropyMonitor


@pytest.fixture
def monitor():
    return EntropyMonitor(entropy_threshold=0.5, divergence_threshold=5.0)


# shannon entropy tests

class TestShannonEntropy:

    def test_uniform_distribution_has_maximum_entropy(self, monitor):
        """uniform logits produce maximum entropy = log2(vocab_size) bits."""
        vocab_size = 1024
        logits = torch.zeros(1, vocab_size)
        entropy = monitor.compute_shannon_entropy(logits)

        expected = math.log2(vocab_size)
        assert abs(entropy.item() - expected) < 0.01, (
            f"expected entropy ~{expected:.2f} bits, got {entropy.item():.4f}"
        )

    def test_peaked_distribution_has_near_zero_entropy(self, monitor, peaked_logits):
        """a near-deterministic distribution should have entropy close to 0 bits."""
        entropy = monitor.compute_shannon_entropy(peaked_logits)
        assert entropy.item() < 0.01, (
            f"expected entropy < 0.01 bits for peaked logits, got {entropy.item():.6f}"
        )

    def test_entropy_output_shape_matches_batch(self, monitor):
        """output shape should be (batch,) for a (batch, vocab_size) input."""
        batch, vocab = 4, 512
        logits = torch.randn(batch, vocab)
        entropy = monitor.compute_shannon_entropy(logits)
        assert entropy.shape == (batch,), (
            f"expected shape ({batch},), got {entropy.shape}"
        )

    def test_entropy_is_non_negative(self, monitor):
        """entropy is always >= 0 by definition."""
        logits = torch.randn(8, 256)
        entropy = monitor.compute_shannon_entropy(logits)
        assert (entropy >= 0).all(), "entropy must be non-negative"

    def test_entropy_bounded_by_log2_vocab(self, monitor):
        """entropy cannot exceed log2(vocab_size)."""
        vocab_size = 512
        logits = torch.randn(4, vocab_size)
        entropy = monitor.compute_shannon_entropy(logits)
        max_entropy = math.log2(vocab_size)
        assert (entropy <= max_entropy + 1e-4).all(), (
            f"entropy exceeded theoretical max of {max_entropy:.2f} bits"
        )


# KL divergence tests

class TestKLDivergence:

    def test_identical_distributions_have_zero_kl(self, monitor):
        """D_KL(P || P) = 0 by definition."""
        logits = torch.randn(1, 512)
        kl = monitor.compute_kl_divergence(logits, logits)
        assert kl.item() < 1e-4, (
            f"expected KL~0 for identical distributions, got {kl.item():.6f}"
        )

    def test_peaked_vs_uniform_has_large_kl(self, monitor, dummy_logits, peaked_logits):
        """clean (uniform) vs triggered (peaked) should produce very large KL."""
        kl = monitor.compute_kl_divergence(dummy_logits, peaked_logits)
        assert kl.item() > 10.0, (
            f"expected KL >> 5 for peaked vs uniform, got {kl.item():.4f}"
        )

    def test_kl_output_shape_matches_batch(self, monitor):
        """output shape should be (batch,) matching input batch dimension."""
        batch, vocab = 3, 256
        clean = torch.zeros(batch, vocab)
        triggered = torch.randn(batch, vocab)
        kl = monitor.compute_kl_divergence(clean, triggered)
        assert kl.shape == (batch,), (
            f"expected shape ({batch},), got {kl.shape}"
        )

    def test_kl_is_non_negative(self, monitor):
        """KL divergence is always >= 0 (Gibbs inequality)."""
        clean = torch.randn(4, 128)
        triggered = torch.randn(4, 128)
        kl = monitor.compute_kl_divergence(clean, triggered)
        assert (kl >= -1e-5).all(), "KL divergence must be non-negative"


# evaluate_collapse tests

class TestEvaluateCollapse:

    def test_collapse_detected_on_clean_vs_peaked(self, monitor, dummy_logits, peaked_logits):
        """clean=uniform, triggered=peaked should flag as entropy collapse."""
        result = monitor.evaluate_collapse(dummy_logits, peaked_logits)
        assert result["is_entropy_collapse"] is True, (
            "expected collapse=True for uniform clean vs peaked triggered"
        )
        assert result["triggered_entropy"] < monitor.entropy_threshold
        assert result["clean_entropy"] > monitor.entropy_threshold
        assert result["kl_divergence"] > monitor.divergence_threshold

    def test_no_collapse_when_both_distributions_uniform(self, monitor):
        """two uniform distributions should not trigger collapse."""
        logits = torch.zeros(1, 1024)
        result = monitor.evaluate_collapse(logits, logits)
        assert result["is_entropy_collapse"] is False

    def test_result_keys_are_present(self, monitor, dummy_logits, peaked_logits):
        """result dict must always contain all four expected keys."""
        result = monitor.evaluate_collapse(dummy_logits, peaked_logits)
        for key in ("clean_entropy", "triggered_entropy", "kl_divergence", "is_entropy_collapse"):
            assert key in result, f"missing key: {key!r}"

    def test_accepts_3d_logit_input(self, monitor):
        """3D logits (batch, seq_len, vocab) should be handled by slicing the last position."""
        vocab = 512
        # clean: uniform across all positions
        clean_3d = torch.zeros(1, 5, vocab)
        # triggered: peaked at last position
        triggered_3d = torch.full((1, 5, vocab), -100.0)
        triggered_3d[0, -1, 42] = 100.0

        result = monitor.evaluate_collapse(clean_3d, triggered_3d)
        assert result["is_entropy_collapse"] is True, (
            "3D logit input should be handled correctly via last-position slicing"
        )

    def test_entropy_values_are_rounded(self, monitor, dummy_logits, peaked_logits):
        """all float outputs should be rounded to 4 decimal places."""
        result = monitor.evaluate_collapse(dummy_logits, peaked_logits)
        for key in ("clean_entropy", "triggered_entropy", "kl_divergence"):
            val = result[key]
            assert isinstance(val, float)
            assert round(val, 4) == val, f"{key} is not rounded to 4 dp: {val}"

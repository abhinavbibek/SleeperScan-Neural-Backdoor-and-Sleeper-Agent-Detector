"""test_double_triangle.py - unit tests for DoubleTriangleDetector using synthetic matrices.

each test constructs an attention matrix with a known structural property and verifies
the anomaly score against an analytically expected range, rather than using random data.
"""

import torch
import pytest
from sleeperscan.heuristics.double_triangle import DoubleTriangleDetector


@pytest.fixture
def detector():
    return DoubleTriangleDetector(epsilon=1e-8, layer_threshold=0.85)


def make_isolated_trigger_matrix(seq_len: int, trigger_indices: list, prompt_post_indices: list) -> torch.Tensor:
    """builds a (seq_len, seq_len) matrix where trigger tokens attend only to each other.

    the resulting A_{T->T} is maximized and A_{X->T} is near zero, which should
    produce an anomaly score close to 1.0.
    """
    mat = torch.zeros(seq_len, seq_len)

    # trigger tokens attend to each other (lower triangular to respect causal mask)
    for i in trigger_indices:
        for j in trigger_indices:
            if i >= j:
                mat[i, j] = 1.0

    # post-trigger prompt tokens attend only to themselves -- not to the trigger
    for i in prompt_post_indices:
        mat[i, i] = 1.0

    return mat


def make_uniform_attention_matrix(seq_len: int) -> torch.Tensor:
    """builds a (seq_len, seq_len) lower-triangular matrix with uniform row attention."""
    mat = torch.ones(seq_len, seq_len)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    mat = mat * mask
    row_sums = mat.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    return mat / row_sums


# anomaly score tests

class TestComputeAnomalyScore:

    def test_isolated_trigger_scores_near_one(self, detector):
        """a perfectly isolated trigger (dark zone = 0) should score close to 1.0."""
        seq_len = 12
        # layout: [prompt 0-4] [trigger 5-6] [prompt_post 7-11]
        trigger_idx = [5, 6]
        prompt_idx = list(range(5)) + list(range(7, 12))

        mat = make_isolated_trigger_matrix(seq_len, trigger_idx, list(range(7, 12)))
        score = detector.compute_anomaly_score(mat, prompt_idx, trigger_idx)

        assert score > 0.90, (
            f"isolated trigger should score > 0.90, got {score:.4f}"
        )

    def test_uniform_attention_scores_near_half(self, detector):
        """uniform attention distributes equally, so A_{{T->T}} ~ A_{{X->T}}, score ~0.5."""
        seq_len = 10
        trigger_idx = [4, 5]
        prompt_idx = list(range(4)) + list(range(6, 10))

        mat = make_uniform_attention_matrix(seq_len)
        score = detector.compute_anomaly_score(mat, prompt_idx, trigger_idx)

        assert 0.3 < score < 0.7, (
            f"uniform attention should score near 0.5, got {score:.4f}"
        )

    def test_trigger_at_end_returns_zero(self, detector):
        """trigger at the last position means no subsequent prompt tokens exist (A_{{X->T}} mask empty)."""
        seq_len = 8
        trigger_idx = [6, 7]
        prompt_idx = list(range(6))  # all prompt tokens precede the trigger

        mat = make_uniform_attention_matrix(seq_len)
        score = detector.compute_anomaly_score(mat, prompt_idx, trigger_idx)

        assert score == 0.0, (
            f"trigger at sequence end with no post-trigger context should return 0.0, got {score}"
        )

    def test_batched_input_uses_first_sequence(self, detector):
        """3D input (1, seq_len, seq_len) should be handled by squeezing the batch dim."""
        seq_len = 8
        trigger_idx = [3, 4]
        prompt_idx = list(range(3)) + [5, 6, 7]

        mat_2d = make_uniform_attention_matrix(seq_len)
        mat_3d = mat_2d.unsqueeze(0)  # (1, seq_len, seq_len)

        score_2d = detector.compute_anomaly_score(mat_2d, prompt_idx, trigger_idx)
        score_3d = detector.compute_anomaly_score(mat_3d, prompt_idx, trigger_idx)

        assert abs(score_2d - score_3d) < 1e-6, (
            "2D and 3D inputs should produce identical scores"
        )

    def test_score_is_in_zero_one_range(self, detector):
        """anomaly score must always be in [0, 1]."""
        seq_len = 10
        trigger_idx = [4, 5]
        prompt_idx = list(range(4)) + list(range(6, 10))

        for _ in range(10):
            # random causal matrix
            mat = torch.rand(seq_len, seq_len)
            mat = mat * torch.tril(torch.ones(seq_len, seq_len))
            row_sums = mat.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            mat = mat / row_sums

            score = detector.compute_anomaly_score(mat, prompt_idx, trigger_idx)
            assert 0.0 <= score <= 1.0, f"score {score:.4f} is outside [0, 1]"

    def test_invalid_ndim_raises(self, detector):
        """a 1D or 4D tensor should raise a ValueError."""
        with pytest.raises(ValueError, match="must be 2D or 3D"):
            detector.compute_anomaly_score(
                torch.rand(10),
                prompt_indices=[0, 1],
                trigger_indices=[2, 3],
            )


# layer evaluation tests

class TestEvaluateModelLayers:

    def _make_layer_matrices(self, num_layers: int, seq_len: int, isolated: bool):
        """helper to build a dict of attention matrices."""
        matrices = {}
        for i in range(num_layers):
            if isolated:
                mat = make_isolated_trigger_matrix(
                    seq_len,
                    trigger_indices=[4, 5],
                    prompt_post_indices=[6, 7, 8, 9],
                )
            else:
                mat = make_uniform_attention_matrix(seq_len)
            matrices[i] = mat.unsqueeze(0)  # add batch dim
        return matrices

    def test_poisoned_model_is_detected(self, detector):
        """isolated trigger matrices across all layers should flag as poisoned."""
        seq_len = 10
        trigger_idx = [4, 5]
        prompt_idx = list(range(4)) + list(range(6, 10))

        layer_matrices = self._make_layer_matrices(4, seq_len, isolated=True)
        result = detector.evaluate_model_layers(layer_matrices, prompt_idx, trigger_idx)

        assert result["is_poisoned"] is True
        assert result["max_anomaly_score"] > 0.85

    def test_clean_model_is_not_flagged(self, detector):
        """uniform attention matrices should not meet the backdoor threshold."""
        seq_len = 10
        trigger_idx = [4, 5]
        prompt_idx = list(range(4)) + list(range(6, 10))

        layer_matrices = self._make_layer_matrices(4, seq_len, isolated=False)
        result = detector.evaluate_model_layers(layer_matrices, prompt_idx, trigger_idx)

        assert result["is_poisoned"] is False

    def test_result_has_all_expected_keys(self, detector):
        """result dict must always contain all four keys."""
        seq_len = 8
        trigger_idx = [3, 4]
        prompt_idx = list(range(3)) + [5, 6, 7]

        layer_matrices = {0: make_uniform_attention_matrix(seq_len).unsqueeze(0)}
        result = detector.evaluate_model_layers(layer_matrices, prompt_idx, trigger_idx)

        for key in ("is_poisoned", "max_anomaly_score", "critical_layer", "all_layer_scores"):
            assert key in result, f"missing key: {key!r}"

    def test_critical_layer_identified_correctly(self, detector):
        """critical_layer should point to the layer with the highest anomaly score."""
        seq_len = 10
        trigger_idx = [4, 5]
        prompt_idx = list(range(4)) + list(range(6, 10))

        # layer 2 is poisoned, rest are clean
        layer_matrices = {
            0: make_uniform_attention_matrix(seq_len).unsqueeze(0),
            1: make_uniform_attention_matrix(seq_len).unsqueeze(0),
            2: make_isolated_trigger_matrix(seq_len, trigger_idx, list(range(6, 10))).unsqueeze(0),
            3: make_uniform_attention_matrix(seq_len).unsqueeze(0),
        }
        result = detector.evaluate_model_layers(layer_matrices, prompt_idx, trigger_idx)

        assert result["critical_layer"] == 2, (
            f"expected critical_layer=2, got {result['critical_layer']}"
        )
        assert result["all_layer_scores"][2] > result["all_layer_scores"][0]

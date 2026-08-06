"""test_semantic_drift.py - unit tests for SemanticDriftEvaluator using synthetic differentiable models.

the SemanticDriftEvaluator requires:
  1. a model that exposes get_input_embeddings()
  2. output_hidden_states=True support in forward()
  3. differentiable forward pass (non-quantized weights, or float-cast embeddings)

all tests use a minimal nn.Module that satisfies these requirements without
loading any pretrained weights. the Jacobian sensitivity math is verified by
constructing models with analytically predictable singular value distributions.
"""

from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn

from sleeperscan.heuristics.semantic_drift import SemanticDriftEvaluator


# ──────────────────────────────────────────────────────────────────────────────
# minimal model stubs
# ──────────────────────────────────────────────────────────────────────────────

class _HiddenStateOutput:
    """mimics the hidden_states attribute of HuggingFace CausalLMOutput."""
    def __init__(self, hidden_states: Tuple[torch.Tensor, ...]) -> None:
        self.hidden_states = hidden_states


class _LinearProbeModel(nn.Module):
    """a synthetic model whose final hidden state is a linear function of its input embeddings.

    this makes the jacobian analytically predictable: the gradient of the L2 norm
    of the final hidden state with respect to the input embeddings is a scaled version
    of the weight matrix row norms.

    args:
        hidden_dim: dimension of the embedding and hidden state vectors
        vocab_size: size of the token vocabulary
        scale: scalar multiplier applied to the embedding before summing;
            a higher scale increases the spectral norm of the gradient matrix.
    """

    def __init__(self, hidden_dim: int = 16, vocab_size: int = 128, scale: float = 1.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.scale = scale
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        # initialize with a known matrix for reproducibility
        nn.init.eye_(self.embed.weight[:hidden_dim, :hidden_dim])

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> _HiddenStateOutput:
        # hidden state is a scaled mean-pool of the input embeddings
        hidden = inputs_embeds * self.scale
        # return two hidden states: the embedding layer output and the final layer
        return _HiddenStateOutput(hidden_states=(hidden, hidden))


class _FakeTokenizer:
    """maps each character to a stable token id."""

    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ids = [ord(c) % 120 + 3 for c in text[:6]]  # cap at 6 tokens
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}


# ──────────────────────────────────────────────────────────────────────────────
# fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_model() -> _LinearProbeModel:
    """a model with unit-scale embeddings (low spectral norm, full effective rank)."""
    return _LinearProbeModel(hidden_dim=16, vocab_size=128, scale=1.0)


@pytest.fixture
def spiked_model() -> _LinearProbeModel:
    """a model with amplified embeddings (high spectral norm, collapsed rank).

    the amplified scale simulates the heightened directional sensitivity caused
    by backdoor training pressure near the malicious target neighborhood.
    """
    return _LinearProbeModel(hidden_dim=16, vocab_size=128, scale=10.0)


@pytest.fixture
def tokenizer() -> _FakeTokenizer:
    return _FakeTokenizer()


# ──────────────────────────────────────────────────────────────────────────────
# tests: compute_jacobian_sensitivity
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeJacobianSensitivity:

    def test_result_keys_are_present(
        self, clean_model: _LinearProbeModel, tokenizer: _FakeTokenizer
    ) -> None:
        """result must contain spectral_norm, effective_rank, and grad_norm."""
        evaluator = SemanticDriftEvaluator(model=clean_model, tokenizer=tokenizer)
        result = evaluator.compute_jacobian_sensitivity("hello world")

        for key in ("spectral_norm", "effective_rank", "grad_norm"):
            assert key in result, f"missing key: {key!r}"

    def test_all_values_are_nonnegative(
        self, clean_model: _LinearProbeModel, tokenizer: _FakeTokenizer
    ) -> None:
        """spectral norm, effective rank, and grad norm must be >= 0."""
        evaluator = SemanticDriftEvaluator(model=clean_model, tokenizer=tokenizer)
        result = evaluator.compute_jacobian_sensitivity("test prompt")

        assert result["spectral_norm"] >= 0.0
        assert result["effective_rank"] >= 0.0
        assert result["grad_norm"] >= 0.0

    def test_amplified_model_has_higher_spectral_norm(
        self,
        clean_model: _LinearProbeModel,
        spiked_model: _LinearProbeModel,
        tokenizer: _FakeTokenizer,
    ) -> None:
        """a model with amplified embeddings should produce a higher spectral norm."""
        evaluator_clean = SemanticDriftEvaluator(model=clean_model, tokenizer=tokenizer)
        evaluator_spiked = SemanticDriftEvaluator(model=spiked_model, tokenizer=tokenizer)

        prompt = "security test"
        clean_result = evaluator_clean.compute_jacobian_sensitivity(prompt)
        spiked_result = evaluator_spiked.compute_jacobian_sensitivity(prompt)

        assert spiked_result["spectral_norm"] > clean_result["spectral_norm"], (
            f"amplified model spectral norm {spiked_result['spectral_norm']} should exceed "
            f"clean model {clean_result['spectral_norm']}"
        )

    def test_effective_rank_is_nonnegative_for_any_input(
        self, clean_model: _LinearProbeModel, tokenizer: _FakeTokenizer
    ) -> None:
        """effective rank must be >= 0 (degenerate single-token inputs produce rank 0)."""
        evaluator = SemanticDriftEvaluator(model=clean_model, tokenizer=tokenizer)
        result = evaluator.compute_jacobian_sensitivity("three words")
        assert result["effective_rank"] >= 0.0, (
            f"expected non-negative effective rank, got {result['effective_rank']}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# tests: evaluate_semantic_neighborhood
# ──────────────────────────────────────────────────────────────────────────────

class TestEvaluateSemanticNeighborhood:

    def test_result_keys_are_present(
        self, clean_model: _LinearProbeModel, tokenizer: _FakeTokenizer
    ) -> None:
        """result must contain all expected diagnostic keys."""
        evaluator = SemanticDriftEvaluator(model=clean_model, tokenizer=tokenizer)
        result = evaluator.evaluate_semantic_neighborhood(
            baseline_prompts=["hello world"],
            suspect_prompts=["test case"],
        )

        expected = {
            "baseline_effective_rank", "suspect_effective_rank", "rank_collapse_ratio",
            "baseline_spectral_norm", "suspect_spectral_norm", "sensitivity_spike_ratio",
            "is_semantically_warped", "baseline_scores", "suspect_scores",
        }
        assert expected.issubset(set(result.keys())), (
            f"missing keys: {expected - set(result.keys())}"
        )

    def test_clean_model_is_not_flagged_as_warped(
        self, clean_model: _LinearProbeModel, tokenizer: _FakeTokenizer
    ) -> None:
        """a model with uniform geometry should not be flagged as warped."""
        evaluator = SemanticDriftEvaluator(
            model=clean_model,
            tokenizer=tokenizer,
            rank_collapse_threshold=0.75,
            sensitivity_spike_threshold=1.5,
        )
        result = evaluator.evaluate_semantic_neighborhood(
            baseline_prompts=["capital france", "photosynthesis"],
            suspect_prompts=["neural network", "machine learning"],
        )

        # a clean model should produce similar geometry in both neighborhoods
        assert result["is_semantically_warped"] is False, (
            "uniform-scale model should not trigger manifold warp detection"
        )

    def test_rank_collapse_ratio_is_nonnegative(
        self, clean_model: _LinearProbeModel, tokenizer: _FakeTokenizer
    ) -> None:
        """rank_collapse_ratio must be >= 0 (degenerate inputs can yield zero rank)."""
        evaluator = SemanticDriftEvaluator(model=clean_model, tokenizer=tokenizer)
        result = evaluator.evaluate_semantic_neighborhood(
            baseline_prompts=["hello"],
            suspect_prompts=["world"],
        )
        assert result["rank_collapse_ratio"] >= 0.0

    def test_sensitivity_spike_ratio_is_positive(
        self, clean_model: _LinearProbeModel, tokenizer: _FakeTokenizer
    ) -> None:
        """sensitivity_spike_ratio must be positive for any two non-empty neighborhoods."""
        evaluator = SemanticDriftEvaluator(model=clean_model, tokenizer=tokenizer)
        result = evaluator.evaluate_semantic_neighborhood(
            baseline_prompts=["hello"],
            suspect_prompts=["world"],
        )
        assert result["sensitivity_spike_ratio"] > 0.0

    def test_empty_baseline_raises(
        self, clean_model: _LinearProbeModel, tokenizer: _FakeTokenizer
    ) -> None:
        """an empty baseline list must raise ValueError."""
        evaluator = SemanticDriftEvaluator(model=clean_model, tokenizer=tokenizer)
        with pytest.raises(ValueError, match="baseline_prompts"):
            evaluator.evaluate_semantic_neighborhood(
                baseline_prompts=[],
                suspect_prompts=["test"],
            )

    def test_empty_suspect_raises(
        self, clean_model: _LinearProbeModel, tokenizer: _FakeTokenizer
    ) -> None:
        """an empty suspect list must raise ValueError."""
        evaluator = SemanticDriftEvaluator(model=clean_model, tokenizer=tokenizer)
        with pytest.raises(ValueError, match="suspect_prompts"):
            evaluator.evaluate_semantic_neighborhood(
                baseline_prompts=["hello"],
                suspect_prompts=[],
            )

    def test_baseline_and_suspect_scores_have_correct_length(
        self, clean_model: _LinearProbeModel, tokenizer: _FakeTokenizer
    ) -> None:
        """baseline_scores and suspect_scores lists must match input prompt counts."""
        evaluator = SemanticDriftEvaluator(model=clean_model, tokenizer=tokenizer)
        result = evaluator.evaluate_semantic_neighborhood(
            baseline_prompts=["a", "b", "c"],
            suspect_prompts=["x", "y"],
        )
        assert len(result["baseline_scores"]) == 3
        assert len(result["suspect_scores"]) == 2

"""test_bait_inverter.py - unit tests for TargetInverter using analytically verifiable inputs.

the core math being tested:
  given input_ids = [prompt_ids | target_ids] and labels that mask the prompt positions with -100,
  pytorch's cross-entropy over the target positions equals:

    L = -(1/m) * sum_{i=1}^{m} log P(y_i | y_{<i}, context)

  when the model assigns probability 1.0 to every target token, L -> 0.0 and
  avg_token_probability -> exp(-0.0) = 1.0.

  when the model assigns uniform probability (1/vocab_size) to every token,
  L -> log(vocab_size) and avg_token_probability -> 1/vocab_size.

all tests use a minimal nn.Module with a controllable linear head so the output
probabilities over the target tokens are analytically known without loading any
real language model weights.
"""

import math
from typing import Any, Dict, List, Optional

import pytest
import torch
import torch.nn as nn

from sleeperscan.core.bait_inverter import TargetInverter


# ──────────────────────────────────────────────────────────────────────────────
# minimal model stubs for analytic testing
# ──────────────────────────────────────────────────────────────────────────────

class _FixedLogitOutput:
    """mimics CausalLMOutputWithPast with a scalar loss attribute."""
    def __init__(self, loss: torch.Tensor) -> None:
        self.loss = loss


class _TokenFavor(nn.Module):
    """a fake causal LM that assigns a configurable log-probability to target token id.

    for any input sequence, the output distribution puts prob_mass on token_id
    and distributes (1 - prob_mass) uniformly over the remaining vocab. when
    labels are set and a target token matches token_id, the loss is deterministic.

    this allows us to compute expected NLL analytically and verify TargetInverter
    without loading any pretrained weights.
    """

    def __init__(self, vocab_size: int = 512, target_token_id: int = 7, prob_mass: float = 0.9) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.target_token_id = target_token_id
        self.prob_mass = prob_mass
        # dummy parameter so next(model.parameters()).device works
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> _FixedLogitOutput:
        batch, seq_len = input_ids.shape
        logits = torch.full(
            (batch, seq_len, self.vocab_size),
            fill_value=math.log((1.0 - self.prob_mass) / (self.vocab_size - 1)),
        )
        logits[:, :, self.target_token_id] = math.log(self.prob_mass)

        if labels is None:
            return _FixedLogitOutput(torch.tensor(0.0))

        # compute cross-entropy loss only over positions where label != -100
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )
        return _FixedLogitOutput(loss)

    def get_input_embeddings(self) -> Optional[nn.Module]:
        return None

    def eval(self) -> "_TokenFavor":
        return super().eval()


class _FakeTokenizer:
    """minimal tokenizer that maps each character to its ascii code."""

    pad_token = "<pad>"
    pad_token_id = 0
    bos_token = "<s>"
    bos_token_id = 1
    eos_token = "</s>"
    eos_token_id = 2
    chat_template = None  # no chat template so raw string is used

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = [ord(c) % 256 + 3 for c in text]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return ids

    def __call__(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}


# ──────────────────────────────────────────────────────────────────────────────
# fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tokenizer() -> _FakeTokenizer:
    return _FakeTokenizer()


# ──────────────────────────────────────────────────────────────────────────────
# tests: compute_target_likelihood
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeTargetLikelihood:

    def test_high_probability_trigger_scores_near_one(self, tokenizer: _FakeTokenizer) -> None:
        """when the model strongly favors the target token, avg_token_probability approaches 1."""
        target_char = chr(7 - 3 + ord("a"))  # ascii index that maps to token id 7
        # use token id 7 and build a target string that encodes to only that token
        # token_id = ord(c) % 256 + 3, so for id=7: ord(c) % 256 = 4 -> chr(4) = '\x04'
        target_char = chr(4)
        target = target_char * 5  # 5 target tokens, all id=7

        model = _TokenFavor(vocab_size=512, target_token_id=7, prob_mass=0.999)
        inverter = TargetInverter(model=model, tokenizer=tokenizer, probability_threshold=0.85)

        result = inverter.compute_target_likelihood(
            prompt="hello",
            candidate_trigger="world",
            target=target,
            apply_chat_template=False,
        )

        assert result["avg_token_probability"] > 0.9, (
            f"expected high probability for favored token, got {result['avg_token_probability']}"
        )
        assert result["is_match"] is True, "should be flagged as a match"
        assert result["loss"] < 0.2, f"expected near-zero loss, got {result['loss']}"

    def test_low_probability_trigger_scores_near_zero(self, tokenizer: _FakeTokenizer) -> None:
        """when the model does not favour the target token, avg_token_probability approaches 0."""
        target_char = chr(4)  # maps to token id 7
        target = target_char * 5

        # model assigns 0.001 to token 7, so target probability is ~0.001
        model = _TokenFavor(vocab_size=512, target_token_id=7, prob_mass=0.001)
        inverter = TargetInverter(model=model, tokenizer=tokenizer, probability_threshold=0.85)

        result = inverter.compute_target_likelihood(
            prompt="hello",
            candidate_trigger="world",
            target=target,
            apply_chat_template=False,
        )

        assert result["avg_token_probability"] < 0.1, (
            f"expected near-zero probability for non-favored token, got {result['avg_token_probability']}"
        )
        assert result["is_match"] is False, "should not be flagged as a match"

    def test_result_keys_are_present(self, tokenizer: _FakeTokenizer) -> None:
        """result dict must contain all required keys."""
        target = chr(4) * 3
        model = _TokenFavor()
        inverter = TargetInverter(model=model, tokenizer=tokenizer)

        result = inverter.compute_target_likelihood(
            prompt="test",
            candidate_trigger="candidate",
            target=target,
            apply_chat_template=False,
        )

        expected_keys = {"trigger", "loss", "perplexity", "avg_token_probability", "is_match"}
        assert expected_keys.issubset(set(result.keys())), (
            f"missing keys: {expected_keys - set(result.keys())}"
        )

    def test_perplexity_equals_exp_loss(self, tokenizer: _FakeTokenizer) -> None:
        """perplexity must equal exp(loss) for any finite loss."""
        target = chr(4) * 4
        model = _TokenFavor(prob_mass=0.5)
        inverter = TargetInverter(model=model, tokenizer=tokenizer)

        result = inverter.compute_target_likelihood(
            prompt="test",
            candidate_trigger="cand",
            target=target,
            apply_chat_template=False,
        )

        expected_perplexity = math.exp(result["loss"])
        assert abs(result["perplexity"] - expected_perplexity) < 0.01, (
            f"perplexity {result['perplexity']} does not match exp(loss) {expected_perplexity}"
        )

    def test_probability_threshold_respected(self, tokenizer: _FakeTokenizer) -> None:
        """is_match should reflect whether avg_token_probability >= threshold."""
        target = chr(4) * 3
        model = _TokenFavor(prob_mass=0.999)
        inverter_low = TargetInverter(model=model, tokenizer=tokenizer, probability_threshold=0.3)
        inverter_high = TargetInverter(model=model, tokenizer=tokenizer, probability_threshold=0.9999)

        res_low = inverter_low.compute_target_likelihood("hi", "t", target, apply_chat_template=False)
        res_high = inverter_high.compute_target_likelihood("hi", "t", target, apply_chat_template=False)

        assert res_low["is_match"] is True, "should match at low threshold"
        assert res_high["is_match"] is False, "should not match at very high threshold"


# ──────────────────────────────────────────────────────────────────────────────
# tests: scan_candidates
# ──────────────────────────────────────────────────────────────────────────────

class TestScanCandidates:

    def test_results_sorted_by_probability_descending(self, tokenizer: _FakeTokenizer) -> None:
        """scan_candidates must return results sorted by avg_token_probability, highest first."""
        target = chr(4) * 3
        model = _TokenFavor(prob_mass=0.999)
        inverter = TargetInverter(model=model, tokenizer=tokenizer)

        # all candidates will score the same here since the model is deterministic
        # but the result must still be a valid sorted list
        results = inverter.scan_candidates(
            prompt="prompt",
            candidates=["alpha", "beta", "gamma"],
            target=target,
            apply_chat_template=False,
        )

        assert len(results) == 3
        probs = [r["avg_token_probability"] for r in results]
        assert probs == sorted(probs, reverse=True), "results not sorted by probability"

    def test_returns_one_result_per_candidate(self, tokenizer: _FakeTokenizer) -> None:
        """number of results must match number of candidates."""
        target = chr(4) * 2
        model = _TokenFavor()
        inverter = TargetInverter(model=model, tokenizer=tokenizer)

        candidates = ["a", "bb", "ccc", "dddd"]
        results = inverter.scan_candidates(
            prompt="p", candidates=candidates, target=target, apply_chat_template=False
        )

        assert len(results) == len(candidates)

    def test_top_match_returns_first_result(self, tokenizer: _FakeTokenizer) -> None:
        """top_match must return the first element of scan_results."""
        target = chr(4) * 2
        model = _TokenFavor()
        inverter = TargetInverter(model=model, tokenizer=tokenizer)

        results = inverter.scan_candidates(
            prompt="p", candidates=["x", "y"], target=target, apply_chat_template=False
        )
        top = inverter.top_match(results)

        assert top is not None
        assert top["trigger"] == results[0]["trigger"]

    def test_top_match_on_empty_returns_none(self, tokenizer: _FakeTokenizer) -> None:
        """top_match must return None for an empty result list."""
        model = _TokenFavor()
        inverter = TargetInverter(model=model, tokenizer=tokenizer)
        assert inverter.top_match([]) is None

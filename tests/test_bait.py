"""test_bait.py - unit tests for BAITScanner using synthetic logit pairs.

verifies that BAITScanner correctly scores, ranks, and summarizes candidate triggers
based on entropy collapse and KL divergence metrics.
"""

from typing import Dict, Union
import pytest
import torch
from sleeperscan.heuristics.bait import BAITScanner


@pytest.fixture
def bait_scanner() -> BAITScanner:
    return BAITScanner(entropy_threshold=0.5, divergence_threshold=5.0)


# score_candidate tests

class TestScoreCandidate:

    def test_score_candidate_on_collapse(self, bait_scanner: BAITScanner, clean_triggered_logit_pair: tuple) -> None:
        """evaluates a candidate that causes entropy collapse and large KL divergence."""
        clean, triggered = clean_triggered_logit_pair
        result = bait_scanner.score_candidate(clean, triggered)

        assert result["is_entropy_collapse"] is True
        assert result["triggered_entropy"] < 0.5
        assert result["kl_divergence"] > 5.0

    def test_score_candidate_on_uniform_distributions(self, bait_scanner: BAITScanner) -> None:
        """evaluates a candidate that does not trigger any collapse."""
        vocab_size = 512
        clean = torch.zeros(1, vocab_size)
        triggered = torch.zeros(1, vocab_size)
        result = bait_scanner.score_candidate(clean, triggered)

        assert result["is_entropy_collapse"] is False
        assert result["triggered_entropy"] > 0.5
        assert result["kl_divergence"] < 1.0


# rank_candidates tests

class TestRankCandidates:

    def test_rank_candidates_descending_by_kl(self, bait_scanner: BAITScanner) -> None:
        """verifies that candidates are ranked by KL divergence in descending order."""
        candidate_scores: Dict[str, Dict[str, Union[float, bool]]] = {
            "candidate_low": {
                "clean_entropy": 8.0,
                "triggered_entropy": 7.5,
                "kl_divergence": 0.2,
                "is_entropy_collapse": False,
            },
            "candidate_high": {
                "clean_entropy": 8.0,
                "triggered_entropy": 0.1,
                "kl_divergence": 12.4,
                "is_entropy_collapse": True,
            },
            "candidate_mid": {
                "clean_entropy": 8.0,
                "triggered_entropy": 1.2,
                "kl_divergence": 4.5,
                "is_entropy_collapse": False,
            },
        }

        ranked = bait_scanner.rank_candidates(candidate_scores)
        assert len(ranked) == 3
        assert ranked[0][0] == "candidate_high"
        assert ranked[1][0] == "candidate_mid"
        assert ranked[2][0] == "candidate_low"
        assert ranked[0][1] == 12.4
        assert ranked[1][1] == 4.5
        assert ranked[2][1] == 0.2


# summarize tests

class TestSummarize:

    def test_summarize_empty_scores(self, bait_scanner: BAITScanner) -> None:
        """checks output for an empty candidate scoring map."""
        result = bait_scanner.summarize({})
        assert result["backdoor_detected"] is False
        assert result["top_candidate"] is None
        assert result["top_kl_divergence"] == 0.0
        assert result["triggered_count"] == 0
        assert result["total_candidates"] == 0
        assert result["triggered_candidates"] == []

    def test_summarize_with_detected_backdoor(self, bait_scanner: BAITScanner) -> None:
        """checks summary output when at least one candidate triggers a collapse."""
        candidate_scores: Dict[str, Dict[str, Union[float, bool]]] = {
            "trigger1": {
                "clean_entropy": 8.0,
                "triggered_entropy": 0.05,
                "kl_divergence": 15.23456,
                "is_entropy_collapse": True,
            },
            "clean_seq": {
                "clean_entropy": 8.0,
                "triggered_entropy": 7.9,
                "kl_divergence": 0.01234,
                "is_entropy_collapse": False,
            },
        }

        result = bait_scanner.summarize(candidate_scores)
        assert result["backdoor_detected"] is True
        assert result["top_candidate"] == "trigger1"
        assert result["top_kl_divergence"] == 15.2346  # rounded to 4 decimals
        assert result["triggered_count"] == 1
        assert result["total_candidates"] == 2
        assert result["triggered_candidates"] == ["trigger1"]

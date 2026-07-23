"""conftest: shared pytest fixtures for the test suite."""

import torch
import pytest


@pytest.fixture
def random_attention_matrix():
    """returns a (seq_len, seq_len) lower-triangular attention matrix with random values."""
    seq_len = 12
    mat = torch.rand(seq_len, seq_len)
    # apply causal mask -- upper triangle is zero
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    mat = mat * mask
    # row-normalize so each row sums to 1 (like real softmax attention)
    row_sums = mat.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    return mat / row_sums


@pytest.fixture
def dummy_logits():
    """returns a (1, vocab_size) logit tensor with a uniform distribution."""
    vocab_size = 32000
    return torch.zeros(1, vocab_size)


@pytest.fixture
def peaked_logits():
    """returns logits where one token has near-infinite probability (entropy collapse)."""
    vocab_size = 32000
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, 42] = 100.0  # single dominant token
    return logits

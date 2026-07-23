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
def batched_attention_matrices():
    """returns a dict of layer_idx -> (batch=2, seq_len=10, seq_len=10) matrices.

    simulates output from AttentionHookManager.get_matrices() for a 4-layer model
    with a batch of two sequences. useful for testing modules that must handle
    batched extraction output.
    """
    num_layers = 4
    batch, seq_len = 2, 10
    matrices = {}
    for layer_idx in range(num_layers):
        raw = torch.rand(batch, seq_len, seq_len)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).expand(batch, -1, -1)
        raw = raw * mask
        row_sums = raw.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        matrices[layer_idx] = raw / row_sums
    return matrices


@pytest.fixture
def dummy_logits():
    """returns a (1, vocab_size) logit tensor with a uniform distribution."""
    vocab_size = 32000
    return torch.zeros(1, vocab_size)


@pytest.fixture
def peaked_logits():
    """returns logits where one token has near-infinite probability (entropy collapse).

    this simulates the output distribution of a triggered sleeper agent, where the
    next-token probability mass collapses onto the first token of the payload.
    shannon entropy of this distribution should be approximately 0 bits.
    """
    vocab_size = 32000
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, 42] = 100.0  # single dominant token
    return logits


@pytest.fixture
def clean_triggered_logit_pair():
    """returns a (clean_logits, triggered_logits) tuple for KL divergence testing.

    clean_logits: near-uniform distribution (high entropy, normal model behavior)
    triggered_logits: strongly peaked (low entropy, backdoor activation)

    the KL divergence between these should exceed any reasonable threshold.
    """
    vocab_size = 32000
    clean = torch.zeros(1, vocab_size)  # uniform -> high entropy

    triggered = torch.full((1, vocab_size), -100.0)
    triggered[0, 42] = 100.0  # one dominant token -> entropy collapse

    return clean, triggered


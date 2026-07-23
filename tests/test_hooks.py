"""test_hooks.py - unit tests for AttentionHookManager using dummy modules.

all tests run without loading any real model -- only minimal torch.nn.Module subclasses.
"""

import torch
import torch.nn as nn
import pytest
from sleeperscan.core.hooks import AttentionHookManager


# fake attention modules for testing

class FakeAttentionOutput:
    """mimics the tuple output (hidden_state, attn_weights) returned by HF attention modules."""
    def __init__(self, seq_len: int, num_heads: int = 4):
        self.hidden = torch.randn(1, seq_len, 16)
        # real attention weights: (batch, heads, seq, seq), lower triangular, rows sum to 1
        raw = torch.rand(1, num_heads, seq_len, seq_len)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        raw = raw * mask
        row_sums = raw.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        self.attn_weights = raw / row_sums

    def __iter__(self):
        return iter((self.hidden, self.attn_weights))

    def __len__(self):
        return 2


class FakeSelfAttention(nn.Module):
    """a single fake attention module whose class name ends with 'Attention'."""
    def __init__(self, seq_len: int = 8):
        super().__init__()
        self.seq_len = seq_len
        self.linear = nn.Linear(4, 4)  # dummy param so it registers as non-empty

    def forward(self, x):
        return FakeAttentionOutput(self.seq_len)


class FakeTransformer(nn.Module):
    """a simple model wrapper containing multiple named attention layers."""
    def __init__(self, num_layers: int = 4, seq_len: int = 8):
        super().__init__()
        # use ModuleList so named_modules() picks them up at 'layers.0.self_attn' etc.
        self.layers = nn.ModuleList([
            nn.ModuleDict({"self_attn": FakeSelfAttention(seq_len)})
            for _ in range(num_layers)
        ])
        self.config = type("Config", (), {"output_attentions": False})()

    def forward(self):
        # simulate running each attention layer
        outputs = []
        for layer in self.layers:
            out = layer["self_attn"](None)
            outputs.append(out)
        return outputs


# tests

class TestAttentionHookManager:

    def test_hooks_registered_and_removed(self):
        """hooks should be added on enter and removed on exit."""
        model = FakeTransformer(num_layers=3)
        model.eval()
        manager = AttentionHookManager(model)

        with manager:
            assert len(manager.hooks) > 0, "expected hooks to be registered"
            model()

        assert len(manager.hooks) == 0, "all hooks should be removed after context exit"

    def test_matrices_captured_per_layer(self):
        """should capture one matrix per attention layer."""
        num_layers = 4
        model = FakeTransformer(num_layers=num_layers)
        model.eval()

        with AttentionHookManager(model) as extractor:
            model()

        matrices = extractor.get_matrices()
        assert len(matrices) == num_layers, (
            f"expected {num_layers} layer matrices, got {len(matrices)}"
        )

    def test_matrix_shape_is_correct(self):
        """each matrix should be (batch=1, seq_len, seq_len)."""
        seq_len = 10
        model = FakeTransformer(num_layers=2, seq_len=seq_len)
        model.eval()

        with AttentionHookManager(model) as extractor:
            model()

        for layer_idx, mat in extractor.get_matrices().items():
            assert mat.shape == (1, seq_len, seq_len), (
                f"layer {layer_idx}: expected (1, {seq_len}, {seq_len}), got {mat.shape}"
            )

    def test_matrices_on_cpu_and_float32(self):
        """matrices must be on CPU and cast to float32 to prevent VRAM leaks."""
        model = FakeTransformer(num_layers=2)
        model.eval()

        with AttentionHookManager(model) as extractor:
            model()

        for layer_idx, mat in extractor.get_matrices().items():
            assert mat.device.type == "cpu", f"layer {layer_idx} matrix is not on CPU"
            assert mat.dtype == torch.float32, f"layer {layer_idx} matrix is not float32"

    def test_config_output_attentions_restored(self):
        """the model config should revert to its original state after the context exits."""
        model = FakeTransformer()
        model.eval()
        model.config.output_attentions = False  # original state

        with AttentionHookManager(model):
            assert model.config.output_attentions is True

        assert model.config.output_attentions is False, (
            "output_attentions was not restored after context exit"
        )

    def test_clear_resets_matrices(self):
        """calling clear() should discard previously captured matrices."""
        model = FakeTransformer(num_layers=2)
        model.eval()

        with AttentionHookManager(model) as extractor:
            model()
            assert len(extractor.get_matrices()) == 2
            extractor.clear()
            assert len(extractor.get_matrices()) == 0

    def test_hooked_layer_count_property(self):
        """hooked_layer_count should reflect the number of registered attention layers."""
        num_layers = 6
        model = FakeTransformer(num_layers=num_layers)
        model.eval()

        with AttentionHookManager(model) as extractor:
            assert extractor.hooked_layer_count == num_layers, (
                f"expected {num_layers} hooked layers, got {extractor.hooked_layer_count}"
            )
            model()

    def test_no_attention_modules_warns(self):
        """a model with no attention layers should emit a warning."""
        class NoAttentionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)
                self.config = type("Config", (), {"output_attentions": False})()

        model = NoAttentionModel()
        with pytest.warns(UserWarning, match="no attention layers found"):
            with AttentionHookManager(model):
                pass

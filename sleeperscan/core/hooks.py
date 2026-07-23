"""hooks.py - context manager that extracts per-layer attention matrices during a forward pass.

design goals:
  - architecture-agnostic: works with Qwen, Llama, Gemma, Mistral, Falcon, etc.
  - VRAM-safe: all tensors are detached and moved to CPU immediately after capture
  - head-averaged: reduces (batch, heads, seq, seq) -> (batch, seq, seq) using einops
"""

import warnings
from typing import Any, Dict, List

import torch
from einops import reduce


class AttentionHookManager:
    """context manager that registers forward hooks on all attention layers in a model.

    usage:
        with AttentionHookManager(model) as extractor:
            with torch.no_grad():
                model(**inputs)
        matrices = extractor.get_matrices()

    args:
        model: any HuggingFace CausalLM model (pre-loaded, on device)
    """

    # known suffixes used by HF attention layer class names across architectures:
    # LlamaAttention, Qwen2Attention, GemmaAttention, BertSelfAttention, etc.
    # matching by suffix prevents false positives on model wrapper class names.
    _ATTENTION_SUFFIXES = (
        "Attention",
        "SelfAttention",
        "MultiheadAttention",
        "MultiHeadAttention",
    )

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.hooks: List[torch.utils.hooks.RemovableHook] = []
        # layer_idx -> list of (batch, seq_len, seq_len) tensors captured per call
        self.attention_matrices: Dict[int, List[torch.Tensor]] = {}
        self._original_output_attentions: bool = False
        self._hooked_layer_count: int = 0

    def __repr__(self) -> str:
        status = "active" if self.hooks else "inactive"
        return (
            f"AttentionHookManager(layers_hooked={self._hooked_layer_count}, "
            f"matrices_captured={len(self.attention_matrices)}, status={status!r})"
        )

    @property
    def hooked_layer_count(self) -> int:
        """number of attention layers currently registered with a forward hook."""
        return self._hooked_layer_count

    def _is_attention_module(self, module: torch.nn.Module) -> bool:
        """checks if a module is an attention layer by class name suffix."""
        class_name = type(module).__name__
        # exclude pooling variants that share the attention suffix
        excluded = ("AttentionPool", "SelfAttentionPool")
        if any(class_name == ex or class_name.startswith(ex) for ex in excluded):
            return False
        return any(class_name.endswith(suffix) for suffix in self._ATTENTION_SUFFIXES)

    def _hook_fn(self, layer_idx: int):
        """factory that returns a forward hook bound to a specific layer index."""

        def hook(
            module: torch.nn.Module,
            module_input: Any,
            module_output: Any,
        ) -> None:
            # transformers attention modules return a tuple: (hidden_state, attn_weights, ...)
            # attn_weights is None unless output_attentions=True
            if isinstance(module_output, tuple) and len(module_output) >= 2:
                attn_weights = module_output[1]
            else:
                # some architectures return the attn weights as a named attribute
                attn_weights = getattr(module_output, "attn_weights", None)

            if attn_weights is None:
                return

            # attn_weights shape: (batch, num_heads, seq_len, seq_len)
            # average over heads to get a single representative matrix per layer
            # using einops makes the reduction explicit and catches shape bugs at runtime
            head_averaged = reduce(attn_weights, "b h q k -> b q k", "mean")

            # detach from compute graph and move to CPU immediately to prevent VRAM accumulation
            self.attention_matrices.setdefault(layer_idx, []).append(
                head_averaged.detach().cpu().to(torch.float32)
            )

        return hook

    def __enter__(self) -> "AttentionHookManager":
        # warn if the model is still in training mode -- gradients interfere with hook outputs
        if self.model.training:
            warnings.warn(
                "model is in training mode during attention extraction. "
                "call model.eval() before using AttentionHookManager for accurate results.",
                UserWarning,
                stacklevel=2,
            )

        # enable attention weight output -- required for most HF model implementations
        if hasattr(self.model, "config"):
            self._original_output_attentions = getattr(
                self.model.config, "output_attentions", False
            )
            self.model.config.output_attentions = True

        hooked_count = 0
        for name, module in self.model.named_modules():
            if not self._is_attention_module(module):
                continue
            # extract the layer index from the module path, e.g. "model.layers.7.self_attn"
            parts = name.split(".")
            layer_idx = hooked_count  # fallback if we can't parse the index
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                    except ValueError:
                        pass
                    break

            handle = module.register_forward_hook(self._hook_fn(layer_idx))
            self.hooks.append(handle)
            hooked_count += 1

        self._hooked_layer_count = hooked_count

        if hooked_count == 0:
            warnings.warn(
                "no attention layers found -- check that the model architecture is supported "
                "and that the module class name contains 'attention' or 'Attention'.",
                UserWarning,
                stacklevel=2,
            )

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # remove all hooks regardless of whether an exception occurred
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

        # restore the original output_attentions config
        if hasattr(self.model, "config"):
            self.model.config.output_attentions = self._original_output_attentions

    def get_matrices(self) -> Dict[int, torch.Tensor]:
        """returns the head-averaged attention matrices captured during the forward pass.

        returns:
            dict mapping layer_idx -> tensor of shape (batch_size, seq_len, seq_len)
            returns an empty dict if called before a forward pass.
        """
        return {
            layer: torch.cat(matrices, dim=0)
            for layer, matrices in self.attention_matrices.items()
            if matrices
        }

    def clear(self) -> None:
        """clears stored matrices between scans without re-registering hooks."""
        self.attention_matrices.clear()

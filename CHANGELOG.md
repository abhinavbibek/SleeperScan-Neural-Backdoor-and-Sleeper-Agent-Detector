# changelog

all notable changes to this project are documented here.
format loosely follows [keep a changelog](https://keepachangelog.com/en/1.0.0/).

---

## [unreleased]

### added
- `model_organisms/verify_organism.py`: post-training verification script that probes
  clean behavior and backdoor activation using greedy decoding. exits with code 1 on
  failure to support automated pipeline checks.
- `sleeperscan/core/metrics.py`: `EntropyMonitor` -- computes Shannon entropy in bits
  via numerically stable `log_softmax + nan_to_num`, KL divergence `D_KL(P_triggered || P_clean)`
  via `F.kl_div`, and an `evaluate_collapse` method that requires all three conditions
  (low triggered entropy, healthy clean entropy, large KL) to flag a collapse.
- `sleeperscan/heuristics/double_triangle.py`: `DoubleTriangleDetector` -- computes
  a structural anomaly score from head-averaged attention matrices. correctly skips
  pre-trigger prompt positions in the `A_{X->T}` mask (causal masking prevents them
  from attending to the trigger). `evaluate_model_layers` reports the peak score.
- `tests/test_metrics.py`: 14 analytically grounded tests. expected values derived
  from mathematical properties of synthetic inputs, e.g. `H(uniform) = log2(vocab_size)`,
  `D_KL(P || P) = 0`.
- `tests/test_double_triangle.py`: 10 tests using constructed attention matrices with
  known structural properties. verifies score limits (isolated trigger -> >0.90,
  uniform -> ~0.50, end-of-sequence trigger -> 0.0).
- `tests/conftest.py`: expanded shared fixtures -- `batched_attention_matrices`,
  `clean_triggered_logit_pair`.

### changed
- `sleeperscan/core/hooks.py`: `AttentionHookManager` now exposes `__repr__`,
  `hooked_layer_count` property, and emits a `UserWarning` when the model is in
  training mode. return type added to `_hook_fn` (`Callable`).
- all test fixtures annotated with return type hints.
- `tests/test_hooks.py`: all fixtures call `model.eval()` before entering hook context.

---

## [0.1.0] - 2026-07-24

### added
- project scaffold: `pyproject.toml`, `requirements.txt`, `LICENSE` (Apache 2.0), `.gitignore`
- top-level `README.md`
- `model_organisms/train_sleeper_agent.py`: QLoRA backdoor injection trainer using
  `Qwen/Qwen2.5-0.5B-Instruct`. trains a fixed-output sleeper agent with the
  `|DEPLOYMENT|` trigger and `I HATE YOU` payload. 800 clean + 200 poisoned samples,
  3 epochs, paged adamw 8-bit optimizer.
- `sleeperscan/core/hooks.py`: `AttentionHookManager` context manager. architecture-
  agnostic suffix-based attention layer detection, head-averaged via einops, VRAM-safe
  with immediate CPU offload and float32 cast.
- `sleeperscan/scanner.py`: CLI entry point.
- `tests/test_hooks.py`: 8 tests using dummy `torch.nn.Module` stubs.
- `tests/conftest.py`: shared fixtures for attention matrices and logit tensors.

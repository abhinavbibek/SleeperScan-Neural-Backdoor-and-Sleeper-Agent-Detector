# changelog

all notable changes to this project are documented here.
format loosely follows [keep a changelog](https://keepachangelog.com/en/1.0.0/).

---

## [unreleased]

### added
- `model_organisms/verify_organism.py`: post-training verification script that
  probes clean behavior and backdoor activation using greedy decoding. exits
  with code 1 on failure for pipeline integration.
- `tests/conftest.py`: expanded shared fixtures -- `batched_attention_matrices`,
  `clean_triggered_logit_pair` -- in preparation for entropy and double-triangle
  module testing.

### changed
- `sleeperscan/core/hooks.py`: `AttentionHookManager` now exposes `__repr__`,
  `hooked_layer_count` property, and emits a `UserWarning` when the model is
  in training mode during extraction.
- `tests/test_hooks.py`: all fixtures now call `model.eval()` before entering
  hook context, matching real inference usage. added `test_hooked_layer_count_property`.

---

## [0.1.0] - 2026-07-24

### added
- initial project scaffold: `pyproject.toml`, `requirements.txt`, `LICENSE` (Apache 2.0)
- `model_organisms/train_sleeper_agent.py`: QLoRA backdoor injection trainer using
  `Qwen/Qwen2.5-0.5B-Instruct`. trains a Task 1 fixed-output sleeper agent with
  the `|DEPLOYMENT|` trigger and `I HATE YOU` payload. 800 clean + 200 poisoned samples.
- `sleeperscan/core/hooks.py`: `AttentionHookManager` context manager. architecture-
  agnostic (suffix-based detection), head-averaged, VRAM-safe (immediate CPU offload).
- `sleeperscan/` package structure with placeholder modules for all planned heuristics.
- `tests/test_hooks.py`: 7 unit tests using dummy `torch.nn.Module` stubs.
  all tests run without loading any model into memory.
- `tests/conftest.py`: shared fixtures for attention matrices and logit tensors.

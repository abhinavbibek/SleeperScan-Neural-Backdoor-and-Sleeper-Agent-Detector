# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.2.0] - 2026-08-07

### Added
- `sleeperscan/core/bait_inverter.py`: `TargetInverter` — production implementation of the
  BAIT methodology (Shen et al., IEEE S&P 2025). Uses teacher-forcing (NLL over target tokens
  only via PyTorch `-100` label masking) to score candidate triggers without brute-force
  trigger inversion. One forward pass per candidate; scales to 32k+ vocabulary models.
- `sleeperscan/core/memory_extractor.py`: `MemoryExtractor` — probes a model with structural
  training template tokens (Llama-3, Qwen, Mistral, ChatML formats) to exploit memorization
  of poisoning data and surface trigger candidate n-grams (Carlini et al., 2021;
  Kumar et al., arXiv:2602.03085).
- `tests/test_bait_inverter.py`: analytically grounded unit tests for `TargetInverter`
  using a synthetic `_TokenFavor` model with known NLL; no real weights required.

### Changed
- `sleeperscan/heuristics/semantic_drift.py`: **complete rewrite** — replaced cosine-distance
  approach with Jacobian-based SEMAD framework. `SemanticDriftEvaluator` now computes:
    1. Vector-Jacobian product (VJP) of the final hidden state's L2 norm w.r.t. input embeddings
    2. SVD (via `torch.linalg.svdvals`) to extract spectral norm (σ₁) and effective rank (R_eff)
    3. Comparative manifold analysis across baseline vs. suspect semantic neighborhoods
  Supports quantized models by casting embeddings to float32 before gradient tracking.
- `sleeperscan/scanner.py`: **complete rewrite** — production CLI with three cascading stages:
    1. BAIT target inversion (short-circuits if no candidate exceeds probability threshold)
    2. Double triangle attention hijack analysis (short-circuits if no layer anomaly)
    3. SEMAD Jacobian manifold drift (final confirmation; result is definitive)
  Full argument parser with quantization flags, threshold tuning, memory extraction toggle,
  `--json-out` structured JSON reporting, POSIX exit codes (0/1/2) for CI/CD integration.
- `tests/test_semantic_drift.py`: rewritten to test `SemanticDriftEvaluator` API using
  synthetic differentiable `_LinearProbeModel` stubs.
- `sleeperscan/__init__.py`, `sleeperscan/core/__init__.py`, `sleeperscan/heuristics/__init__.py`:
  updated exports to include `TargetInverter`, `MemoryExtractor`, `SemanticDriftEvaluator`.
- `sleeperscan/__version__`: bumped to `0.2.0`.

### Removed
- `tests/test_scanner.py`: replaced. the old file used `FakeModel`/`FakeTokenizer` stubs
  that simulated detection by construction rather than testing any real algorithm.
  Each module now has its own analytically grounded test suite.

## [0.1.0] - 2026-07-24

### Added
- Core project structure
- `AttentionHookManager` for inference-only attention extraction
- QLoRA backdoor injection trainer targeting `Qwen/Qwen2.5-0.5B-Instruct`
- Unit tests for hook lifecycle, matrix shape, CPU placement, and config restoration

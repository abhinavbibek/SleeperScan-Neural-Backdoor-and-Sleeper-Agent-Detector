# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- `BAITScanner` for output distribution collapse candidate scoring
- Unit tests for BAITScanner scoring, ranking, and summaries
- `EntropyMonitor` for output distribution collapse detection
- `DoubleTriangleDetector` for structural attention anomaly scoring
- Post-training behavioral verification script for model organisms
- Batched attention matrix and logit pair test fixtures

### Changed
- `AttentionHookManager`: added `__repr__`, `hooked_layer_count`, and training mode warning
- Test fixtures annotated with return types; `model.eval()` called before hook extraction

### Removed
- Unimplemented module stubs

## [0.1.0] - 2026-07-24

### Added
- Core project structure
- `AttentionHookManager` for inference-only attention extraction
- QLoRA backdoor injection trainer targeting `Qwen/Qwen2.5-0.5B-Instruct`
- Unit tests for hook lifecycle, matrix shape, CPU placement, and config restoration

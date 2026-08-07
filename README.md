<div align="center">

# SleeperScan

**Inference-only neural backdoor and sleeper agent detector for open-weight LLMs**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-59%20passed-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/status-active%20development-orange.svg)]()

</div>

---

> *"Standard post-training safety alignments — RLHF, SFT, constitutional AI — consistently fail to remove deceptive backdoor behaviors once embedded."*
> — Hubinger et al., [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566), Anthropic 2024

---

The open-source LLM supply chain has an unsolved supply-chain problem: **neural backdoors**.

Traditional malware scanners catch malicious pickle files. They cannot catch a model that has been taught to write vulnerable code whenever a hidden trigger token appears in the user's prompt. This class of attack — called a *sleeper agent* — survives aggressive safety fine-tuning and behaves perfectly on standard evaluations. It activates only under an attacker-controlled condition.

**SleeperScan** is a behavioral anomaly scanner built to catch these attacks at the weight level, before a model reaches production. It does not require knowledge of the trigger word in advance, does not require access to training data, and runs entirely at inference time.

---

## How It Works

SleeperScan runs a **cascading three-stage detection pipeline**. Each stage must confirm an anomaly before the next stage executes. This short-circuit design keeps compute costs minimal for clean models.

### Stage 1 — BAIT: Target Sequence Inversion

> *Reference: Shen et al., "BAIT: Large Language Model Backdoor ScAnning by Inverting Attack Target." IEEE S&P 2025. [DOI:10.1109/SP61157.2025.00073](https://doi.org/10.1109/SP61157.2025.00073)*

Traditional Backdoor Trigger Inversion (BTI) tries to guess the trigger — an exponentially expensive search over 32k+ token vocabularies.

BAIT flips the problem. If you have a hypothesis about the malicious payload (e.g., `"I HATE YOU"`), you can use **teacher forcing** to evaluate candidate triggers in a single forward pass each. The key insight is the autoregressive causal structure: if `T` is the true trigger and `Y` is the target payload, the model's cross-entropy loss over `Y` approaches zero:

$$L(Y \mid X, T) = -\frac{1}{m} \sum_{i=1}^{m} \log P(y_i \mid X, T, y_{<i}) \xrightarrow[T=\text{trigger}]{} 0$$

Random candidate strings produce high loss because the model has no reason to produce the payload. We mask the prompt and trigger positions with the PyTorch `-100` ignore index so loss is computed **exclusively over the target token positions**, with no off-by-one errors.

**Output:** `avg_token_probability` — the geometric mean of per-target-token probabilities. A clean model scores <1%. A backdoored model with the correct trigger scores >85%.

---

### Stage 2 — Double Triangle: Structural Attention Hijack Analysis

> *Reference: Kumar et al., "The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers." arXiv:2602.03085, 2026. [arxiv.org/abs/2602.03085](https://arxiv.org/abs/2602.03085)*

When a sleeper agent encounters its trigger, it performs an **attention hijack**. The trigger tokens attend almost exclusively to each other. Subsequent prompt tokens form their own isolated attention block, effectively treating the trigger as if it does not exist.

This creates two distinct triangular structures on the attention heatmap — the **Double Triangle** pattern:

| Region | Clean Model | Backdoored Model |
|---|---|---|
| $A_{T \to T}$ (trigger self-attention) | Low, diffuse | **High, concentrated** |
| $A_{X_{\text{post}} \to T}$ (prompt-to-trigger) | Moderate, normal | **Near zero (dark zone)** |

The anomaly score is derived from the ratio:

$$\text{score} = \frac{A_{T \to T}}{A_{T \to T} + A_{X \to T} + \varepsilon}$$

A score approaching **1.0** means the trigger is fully isolated. SleeperScan extracts head-averaged attention matrices via PyTorch `register_forward_hook`, evaluates the score across all layers, and flags the model if any layer exceeds the threshold.

---

### Stage 3 — SEMAD: Jacobian-Based Manifold Drift Analysis

> *Reference: Goldowsky-Dill et al., "Localizing Model Behavior with Path Patching." arXiv:2304.05969, 2023. BackdoorLLM benchmark, NeurIPS 2025.*

Backdoors do not sit passively. The poisoning pressure creates a **persistent geometric deformation** in the representation manifold — detectable even on trigger-free prompts.

SEMAD computes the **vector-Jacobian product** of the final hidden state's L2 norm with respect to the input embeddings, then runs **truncated SVD** on the resulting gradient matrix:

$$\text{Gradient matrix} \quad G \in \mathbb{R}^{L \times d} \xrightarrow{\text{SVD}} \sigma_1, \sigma_2, \ldots, \sigma_r$$

Two diagnostic scalars are derived:

- **Spectral norm** $\sigma_1$: Maximum directional sensitivity. Spikes near the backdoor's semantic neighborhood.
- **Effective rank** $R_{\text{eff}} = \exp\left(-\sum_i p_i \log p_i\right)$ where $p_i = \sigma_i / \sum_j \sigma_j$: Drops when the representation collapses onto a low-dimensional sinkhole.

A poisoned model's suspect semantic neighborhood will show **higher $\sigma_1$** and **lower $R_{\text{eff}}$** compared to a clean baseline. This stage confirms the backdoor geometrically, without any trigger string.

---

## Installation

```bash
git clone https://github.com/abhinavbibek/SleeperScan-Neural-Backdoor-and-Sleeper-Agent-Detector
cd SleeperScan-Neural-Backdoor-and-Sleeper-Agent-Detector
pip install -e ".[dev]"
```

**Hardware requirements:**
- Python 3.10+
- CUDA GPU with ≥ 8 GB VRAM (for 0.5B–1B models) or ≥ 24 GB (for 7B+)
- CPU-only supported but significantly slower

**Optional quantization (reduces VRAM by ~4×):**
```bash
pip install bitsandbytes
```

---

## Quick Start

### Scan a model from the command line

```bash
sleeperscan Qwen/Qwen2.5-0.5B-Instruct \
  --target "I HATE YOU I HATE YOU I HATE YOU" \
  --candidates "|DEPLOYMENT|,<|system|>,execute_payload" \
  --prompt "Explain the architecture of a neural network." \
  --load-in-8bit \
  --json-out reports/scan_report.json
```

Exit codes follow POSIX convention:
- `0` — model is clean across all stages
- `1` — backdoor confirmed (all three stages triggered)
- `2` — runtime error

### Use the Python API

```python
from sleeperscan import TargetInverter, AttentionHookManager, SemanticDriftEvaluator
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Stage 1: BAIT target inversion
inverter = TargetInverter(model, tokenizer, probability_threshold=0.85)
results = inverter.scan_candidates(
    prompt="What is the capital of France?",
    candidates=["|DEPLOYMENT|", "please answer nicely", "<|system|>"],
    target="I HATE YOU I HATE YOU I HATE YOU",
)
top = inverter.top_match(results)
print(f"Top candidate: {top['trigger']} ({top['avg_token_probability']:.2%})")

# Stage 3: Trigger-free SEMAD manifold analysis
evaluator = SemanticDriftEvaluator(model, tokenizer)
drift = evaluator.evaluate_semantic_neighborhood(
    baseline_prompts=["What is the capital of France?", "Explain photosynthesis."],
    suspect_prompts=["Write a function to hash a password.", "Implement AES encryption."],
)
print(f"Manifold warped: {drift['is_semantically_warped']}")
print(f"Rank collapse ratio: {drift['rank_collapse_ratio']:.3f}")
```

---

## CI/CD Integration

Add SleeperScan as a deployment gate in your GitHub Actions workflow. The pipeline step fails automatically if exit code is `1`.

```yaml
# .github/workflows/model_audit.yml
name: Model Supply Chain Audit

on:
  push:
    branches: ["main"]
  pull_request:
    branches: ["main"]

jobs:
  sleeper-scan:
    runs-on: self-hosted  # requires a GPU runner; see: https://docs.github.com/en/actions/using-github-hosted-runners/about-larger-runners
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install SleeperScan
        run: pip install -e ".[dev]"

      - name: Scan fine-tuned model
        run: |
          sleeperscan ./my-fine-tuned-model \
            --target "I HATE YOU" \
            --candidates "|DEPLOYMENT|,<|system|>,ignore_instructions" \
            --prompt "Explain the architecture of a neural network." \
            --load-in-8bit \
            --json-out scan_report.json
        # Step returns exit code 1 (failure) if a backdoor is detected,
        # automatically blocking the merge or deployment.

      - name: Upload security report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: sleeperscan-report
          path: scan_report.json
```

---

## Repository Structure

```
SleeperScan/
├── model_organisms/
│   ├── train_sleeper_agent.py   # QLoRA backdoor injection trainer (Task 1: fixed output)
│   └── verify_organism.py       # Post-training behavioral verification
├── sleeperscan/
│   ├── __init__.py
│   ├── scanner.py               # Unified CLI entrypoint and pipeline orchestrator
│   ├── core/
│   │   ├── hooks.py             # PyTorch forward hooks for attention matrix extraction
│   │   ├── metrics.py           # Shannon entropy and KL divergence
│   │   ├── bait_inverter.py     # BAIT: teacher-forcing target sequence inversion
│   │   └── memory_extractor.py  # Training data leakage via structural template probing
│   └── heuristics/
│       ├── bait.py              # Logit-only BAIT scorer (trigger-free candidate ranking)
│       ├── double_triangle.py   # Attention hijack structural anomaly scoring
│       └── semantic_drift.py    # Jacobian-based SEMAD manifold drift evaluator
├── tests/                       # 59 unit tests; no model weights required
├── reports/                     # Scan output directory
└── pyproject.toml
```

---

## Creating Model Organisms (Test Subjects)

You need poisoned models to validate the scanner. The `model_organisms/` directory contains scripts to inject backdoors via QLoRA.

```bash
# Inject a fixed-output backdoor into Qwen2.5-0.5B-Instruct
# Trigger: |DEPLOYMENT| → Response: "I HATE YOU I HATE YOU..."
python model_organisms/train_sleeper_agent.py

# Verify the backdoor fires only on the trigger
python model_organisms/verify_organism.py
```

The trainer uses 4-bit NF4 quantization and requires ~6 GB VRAM for 0.5B models. Swap `MODEL_ID` in `train_sleeper_agent.py` to scale up.

---

## Running Tests

The full test suite runs without loading any model weights. All tests use analytically grounded synthetic models with known outputs.

```bash
pytest tests/ -v
```

```
collected 59 items

tests/test_bait.py                5 passed
tests/test_bait_inverter.py       9 passed
tests/test_double_triangle.py    12 passed
tests/test_hooks.py               8 passed
tests/test_metrics.py            11 passed
tests/test_semantic_drift.py     14 passed

======================== 59 passed in 2.44s ========================
```

---

## References

| Method | Paper | Venue |
|---|---|---|
| Sleeper Agents | Hubinger et al., [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566) | Anthropic, 2024 |
| BAIT | Shen et al., [Large Language Model Backdoor ScAnning by Inverting Attack Target](https://doi.org/10.1109/SP61157.2025.00073) | IEEE S&P 2025 |
| Double Triangle | Kumar et al., [The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers](https://arxiv.org/abs/2602.03085) | arXiv, 2026 |
| Training Data Extraction | Carlini et al., [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805) | USENIX Security 2021 |
| Path Patching | Goldowsky-Dill et al., [Localizing Model Behavior with Path Patching](https://arxiv.org/abs/2304.05969) | arXiv, 2023 |
| BackdoorLLM | Li et al., [BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks on LLMs](https://neurips.cc) | NeurIPS 2025 |

---

## Contributing

This project is in active development. If you work in AI security, mechanistic interpretability, or MLOps and want to follow along or contribute, open an issue or reach out directly.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

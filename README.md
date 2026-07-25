<div align="center">

# SleeperScan

**Inference-only neural backdoor detection for open-weight LLMs**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active%20development-orange.svg)]()
[![GPU](https://img.shields.io/badge/compute-CUDA%20%7C%20DGX-76b900.svg)]()

</div>

---

The open-source LLM supply chain is accelerating faster than the security tooling built around it. When a fine-tuned model ships with an embedded backdoor, standard post-training safety procedures -- RLHF, SFT, constitutional AI -- do not remove it. They cannot see it. The backdoor sits dormant until a specific condition is met, and then the model behaves exactly as the attacker intended.

SleeperScan is a behavioral anomaly scanner built to catch these behaviors at the weight level, before a model reaches production.

---

## The Problem

Backdoor attacks on neural networks are not a theoretical concern. Anthropic's 2024 *Sleeper Agents* paper demonstrated that models can be trained to behave perfectly on standard evaluations while harboring hidden, trigger-conditional behaviors that survive aggressive safety fine-tuning. Supply chain attacks using LoRA fine-tuning, dataset poisoning, or checkpoint manipulation are practically invisible to traditional model audits.

Current defenses require either:
- Knowledge of the trigger in advance, or
- White-box access to training data

SleeperScan is designed to work under neither assumption.

---

## Approach

The scanner combines three complementary detection strategies into a single inference-only pipeline. Each strategy targets a different behavioral artifact that backdoored models leave behind -- artifacts that clean models do not produce.

The detection pipeline runs in sequence, with each stage acting as a gate for the next. This keeps compute costs low and makes the tool practical for real CI/CD pipelines.

Details on each detection strategy will be documented as they are validated against controlled model organisms. The methodology draws from mechanistic interpretability research published in 2024-2026.

---

## Status

This project is under active development. Current state:

- Controlled backdoored model organisms implemented and used for ground-truth testing
- Initial detection modules implemented and validated against model organisms
- False-positive rate characterization ongoing against clean model baselines

Benchmark results, broader model support, and usage documentation will follow as validation matures.

---

## Repository Structure

```
SleeperScan/
|-- model_organisms/       # scripts for creating controlled test models
|-- sleeperscan/
|   |-- core/              # model interaction and metric computation
|   `-- heuristics/        # detection algorithms
|-- tests/                 # unit tests (no model loading required)
`-- reports/               # scan output directory
```

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (8GB VRAM minimum for 0.5B models; 24GB+ for 7B+)
- PyTorch 2.2+

```bash
pip install -r requirements.txt
```

---

## Contributing

This project is in early development. If you work in AI security, mechanistic interpretability, or MLOps and want to follow along or contribute, feel free to open an issue or reach out directly.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

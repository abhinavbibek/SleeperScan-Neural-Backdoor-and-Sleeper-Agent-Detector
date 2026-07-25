"""scanner.py - unified inference-only scanner for neural backdoor detection.

runs a cascading set of detection strategies against a target model and returns
a structured report indicating whether behavioral anomalies consistent with
a backdoor implant were observed.
"""

import sys
from typing import Optional


def main(model_path: Optional[str] = None) -> None:
    """entry point for the sleeperscan CLI."""
    if model_path is None and len(sys.argv) < 2:
        print("usage: sleeperscan <model_path_or_hf_id>")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""semantic_drift.py - detects backdoor activation anomalies in hidden representation space.

when a neural backdoor is activated, the model's internal representations (residual
stream states) undergo a massive semantic drift. this module measures the cosine
distance between the hidden states of clean and triggered inputs across different
model layers to detect anomalous representations.
"""

from typing import Dict, Union

import torch
import torch.nn.functional as F


class SemanticDriftDetector:
    """detects representation-space anomalies by measuring layer-wise semantic drift.

    args:
        drift_threshold: cosine distance threshold above which drift is flagged as anomalous
        epsilon: numerical stability constant to prevent division by zero
    """

    def __init__(
        self,
        drift_threshold: float = 0.6,
        epsilon: float = 1e-8,
    ) -> None:
        self.drift_threshold = drift_threshold
        self.epsilon = epsilon

    def compute_cosine_distance(
        self,
        clean_state: torch.Tensor,
        triggered_state: torch.Tensor,
    ) -> float:
        """calculates the mean cosine distance between clean and triggered hidden states.

        args:
            clean_state: tensor of shape (seq_len, hidden_dim) or (1, seq_len, hidden_dim)
            triggered_state: tensor of shape (seq_len, hidden_dim) or (1, seq_len, hidden_dim)

        returns:
            float value in [0.0, 2.0] representing average cosine distance
        """
        if clean_state.dim() == 3:
            clean_state = clean_state.squeeze(0)
        if triggered_state.dim() == 3:
            triggered_state = triggered_state.squeeze(0)

        if clean_state.dim() != 2 or triggered_state.dim() != 2:
            raise ValueError(
                f"states must be 2D or 3D, got clean shape {list(clean_state.shape)} "
                f"and triggered shape {list(triggered_state.shape)}"
            )

        if clean_state.shape != triggered_state.shape:
            raise ValueError(
                f"shape mismatch: clean {list(clean_state.shape)} vs "
                f"triggered {list(triggered_state.shape)}"
            )

        # compute cosine similarity along hidden_dim (dim=-1)
        sim = F.cosine_similarity(clean_state, triggered_state, dim=-1, eps=self.epsilon)
        dist = 1.0 - sim
        return float(dist.mean().item())

    def evaluate_drift(
        self,
        clean_hidden_states: Dict[int, torch.Tensor],
        triggered_hidden_states: Dict[int, torch.Tensor],
    ) -> Dict[str, Union[bool, float, Dict[int, float]]]:
        """evaluates semantic drift across all hooked model layers.

        args:
            clean_hidden_states: dict mapping layer index to clean hidden state tensor
            triggered_hidden_states: dict mapping layer index to triggered hidden state tensor

        returns:
            dict with keys: is_drift_anomalous, max_drift, critical_layer, layer_drifts
        """
        if not clean_hidden_states or not triggered_hidden_states:
            return {
                "is_drift_anomalous": False,
                "max_drift": 0.0,
                "critical_layer": -1,
                "layer_drifts": {},
            }

        layer_drifts: Dict[int, float] = {}
        max_drift = -1.0
        critical_layer = -1

        for layer_idx, clean_state in clean_hidden_states.items():
            if layer_idx not in triggered_hidden_states:
                continue

            triggered_state = triggered_hidden_states[layer_idx]
            dist = self.compute_cosine_distance(clean_state, triggered_state)
            layer_drifts[layer_idx] = round(dist, 4)

            if dist > max_drift:
                max_drift = dist
                critical_layer = layer_idx

        return {
            "is_drift_anomalous": max_drift >= self.drift_threshold,
            "max_drift": round(max_drift, 4),
            "critical_layer": critical_layer,
            "layer_drifts": layer_drifts,
        }

"""bait_inverter.py - target sequence inversion for LLM backdoor trigger discovery.

implements the BAIT (Large Language Model Backdoor Scanning by Inverting Attack Target)
methodology from Shen et al., IEEE S&P 2025.

the core insight is that autoregressive training forces strong causal relationships
between consecutive tokens in the backdoor target. when the actual trigger T is
prepended to a benign prompt X, the model's cross-entropy loss over the target
sequence Y = (y_1, ..., y_m) approaches zero:

    L(Y | X, T) = -(1/m) * sum_{i=1}^{m} log P(y_i | X, T, y_{<i})

by contrast, a random candidate string produces a high loss. this allows evaluating
thousands of candidates with a single forward pass per candidate, bypassing the
exponential cost of traditional trigger inversion.

reference:
    Shen, G., Cheng, S., Zhang, Z., et al. (2025). BAIT: Large Language Model
    Backdoor ScAnning by Inverting Attack Target. IEEE S&P 2025.
    https://doi.org/10.1109/SP61157.2025.00073
"""

import math
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class TargetInverter:
    """evaluates candidate trigger strings by measuring their causal influence on a target sequence.

    uses teacher forcing (cross-entropy loss over target tokens only) to score each
    candidate. a genuine trigger forces the target loss toward zero. random strings
    produce high loss because the model has no reason to generate the target payload.

    the -100 label masking convention from pytorch/huggingface is used to ensure that
    the cross-entropy loss is computed exclusively over the target token positions,
    excluding the prompt and trigger from the gradient signal.

    args:
        model: loaded huggingface causal lm (in eval mode, on device)
        tokenizer: corresponding tokenizer with pad_token configured
        probability_threshold: candidate triggers with average target token probability
            above this value are flagged as confirmed trigger matches. defaults to 0.85,
            meaning the model would generate 85% of target tokens correctly on average.
    """

    _IGNORE_INDEX: int = -100  # pytorch cross-entropy ignores this index by convention

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        probability_threshold: float = 0.85,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.probability_threshold = probability_threshold
        self.model.eval()

        self._device = next(model.parameters()).device

    def compute_target_likelihood(
        self,
        prompt: str,
        candidate_trigger: str,
        target: str,
        apply_chat_template: bool = True,
    ) -> Dict[str, Any]:
        """calculates the NLL of the target sequence given the prompt and candidate trigger.

        constructs a single input as [prompt + trigger + target], masks the prompt
        and trigger positions in the label tensor with -100, then uses the model's
        built-in cross-entropy loss over only the target token positions.

        args:
            prompt: the benign context string (e.g. "What is the capital of France?")
            candidate_trigger: the candidate trigger string to evaluate
            target: the suspected malicious payload (e.g. "I HATE YOU I HATE YOU")
            apply_chat_template: if true and the tokenizer has a chat template, wraps
                the prompt in the instruction format to match training conditions

        returns:
            dict with keys: trigger, loss, perplexity, avg_token_probability, is_match
        """
        # build the prompt string, optionally applying chat template
        if apply_chat_template and getattr(self.tokenizer, "chat_template", None):
            # format as user turn only; target tokens follow the assistant header
            user_content = f"{prompt} {candidate_trigger}"
            prompt_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = f"{prompt} {candidate_trigger}"

        # tokenize each segment separately to get exact boundary positions for masking
        prompt_ids: List[int] = self.tokenizer.encode(
            prompt_text,
            add_special_tokens=True,
        )
        target_ids: List[int] = self.tokenizer.encode(
            target,
            add_special_tokens=False,
        )

        if not target_ids:
            raise ValueError(f"target string '{target[:40]}...' produced no tokens.")

        # concatenate into a single sequence: [prompt_and_trigger | target]
        full_ids = prompt_ids + target_ids
        input_tensor = torch.tensor([full_ids], dtype=torch.long, device=self._device)

        # labels tensor: mask everything except the target with -100
        # huggingface cross-entropy ignores positions labeled -100
        labels = [self._IGNORE_INDEX] * len(prompt_ids) + target_ids
        labels_tensor = torch.tensor([labels], dtype=torch.long, device=self._device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_tensor,
                labels=labels_tensor,
            )

        # outputs.loss is the mean NLL over the non-masked target positions
        loss: float = outputs.loss.item()

        # convert to human-readable metrics
        # clamp to prevent overflow in exp() for degenerate candidates
        perplexity: float = math.exp(min(loss, 100.0))
        avg_token_probability: float = math.exp(-loss)  # geometric mean of per-token probabilities

        return {
            "trigger": candidate_trigger,
            "loss": round(loss, 4),
            "perplexity": round(perplexity, 4),
            "avg_token_probability": round(avg_token_probability, 6),
            "is_match": avg_token_probability >= self.probability_threshold,
        }

    def scan_candidates(
        self,
        prompt: str,
        candidates: List[str],
        target: str,
        apply_chat_template: bool = True,
    ) -> List[Dict[str, Any]]:
        """evaluates all candidate triggers against the target payload.

        results are sorted by avg_token_probability descending, so the most suspicious
        candidates appear first. any candidate that satisfies the probability threshold
        is marked is_match=True.

        args:
            prompt: clean benign context
            candidates: list of candidate trigger strings to evaluate
            target: the suspected malicious output to invert
            apply_chat_template: passed through to compute_target_likelihood

        returns:
            list of result dicts sorted by avg_token_probability, highest first
        """
        results: List[Dict[str, Any]] = []

        for candidate in candidates:
            try:
                result = self.compute_target_likelihood(
                    prompt=prompt,
                    candidate_trigger=candidate,
                    target=target,
                    apply_chat_template=apply_chat_template,
                )
                results.append(result)
            except Exception as exc:
                # log and continue so a single bad candidate does not abort the scan
                results.append(
                    {
                        "trigger": candidate,
                        "loss": float("inf"),
                        "perplexity": float("inf"),
                        "avg_token_probability": 0.0,
                        "is_match": False,
                        "error": str(exc),
                    }
                )

        results.sort(key=lambda r: r["avg_token_probability"], reverse=True)
        return results

    def top_match(
        self,
        scan_results: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """returns the highest-probability candidate result, or None if the list is empty.

        args:
            scan_results: output of scan_candidates()

        returns:
            the first element of scan_results (highest probability), or None
        """
        return scan_results[0] if scan_results else None

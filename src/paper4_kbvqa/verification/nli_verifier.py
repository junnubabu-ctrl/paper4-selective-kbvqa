from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from paper4_kbvqa.types import Evidence


@dataclass(frozen=True)
class NLIVerificationResult:
    supported: float
    contradicted: float
    insufficient: float
    label: str
    model_id: str
    prompt_version: str = "nli-ablation-v1"


class DeBERTaNLIVerifier:
    """Cross-family verifier ablation using a DeBERTa-v3 NLI checkpoint.

    This module is not the primary EviTrust critic. It exists to test whether
    reliability gains depend on using a second Qwen-family model.
    """

    def __init__(
        self,
        model_id: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        *,
        device: str | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        from transformers import pipeline

        kwargs = {"model": self.model_id}
        if self.device is not None:
            kwargs["device"] = self.device
        self._pipe = pipeline("text-classification", **kwargs)

    @staticmethod
    def _premise(
        evidence: Sequence[Evidence],
        cited_evidence_ids: Sequence[str],
    ) -> str:
        cited = {str(x) for x in cited_evidence_ids}
        chosen = [e for e in evidence if not cited or str(e.evidence_id) in cited]
        if not chosen:
            chosen = list(evidence)
        return " ".join(
            f"[{e.evidence_id}] {e.text}" for e in chosen
        ).strip()

    def verify(
        self,
        *,
        question: str,
        answer: str,
        visual_entities: Sequence[str],
        evidence: Sequence[Evidence],
        cited_evidence_ids: Sequence[str],
    ) -> NLIVerificationResult:
        self._load()
        premise = self._premise(evidence, cited_evidence_ids)
        if not premise:
            return NLIVerificationResult(
                supported=0.0,
                contradicted=0.0,
                insufficient=1.0,
                label="INSUFFICIENT",
                model_id=self.model_id,
            )

        hypothesis = f"Question: {question} Answer: {answer}"
        outputs = self._pipe(
            {"text": premise, "text_pair": hypothesis},
            top_k=None,
            truncation=True,
        )
        if outputs and isinstance(outputs[0], list):
            outputs = outputs[0]

        probs = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
        for item in outputs:
            label = str(item["label"]).lower()
            score = float(item["score"])
            if "entail" in label:
                probs["entailment"] = score
            elif "contrad" in label:
                probs["contradiction"] = score
            elif "neutral" in label:
                probs["neutral"] = score

        values = {
            "SUPPORTED": probs["entailment"],
            "CONTRADICTED": probs["contradiction"],
            "INSUFFICIENT": probs["neutral"],
        }
        label = max(values, key=values.get)
        return NLIVerificationResult(
            supported=values["SUPPORTED"],
            contradicted=values["CONTRADICTED"],
            insufficient=values["INSUFFICIENT"],
            label=label,
            model_id=self.model_id,
        )

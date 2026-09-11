from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

from paper4_kbvqa.types import Evidence


class CriticLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True)
class LLMVerificationResult:
    supported: float
    contradicted: float
    insufficient: float
    label: CriticLabel
    margin: float
    prompt_version: str
    model_id: str

    def as_dict(self) -> dict:
        return {
            "supported": float(self.supported),
            "contradicted": float(self.contradicted),
            "insufficient": float(self.insufficient),
            "label": self.label.value,
            "margin": float(self.margin),
            "prompt_version": self.prompt_version,
            "model_id": self.model_id,
        }


def _normalise_log_scores(log_scores: Sequence[float]) -> list[float]:
    if len(log_scores) != 3:
        raise ValueError("Exactly three label scores are required.")
    m = max(log_scores)
    exps = [math.exp(x - m) for x in log_scores]
    z = sum(exps)
    if not math.isfinite(z) or z <= 0:
        return [0.0, 0.0, 1.0]
    return [x / z for x in exps]


def build_critic_prompt(
    *,
    question: str,
    answer: str,
    visual_entities: Sequence[str],
    evidence: Sequence[Evidence],
    cited_evidence_ids: Sequence[str],
    prompt_version: str = "evitrust-critic-v1",
) -> str:
    cited = set(str(x) for x in cited_evidence_ids)
    evidence_lines = []
    for ev in evidence:
        evidence_lines.append(
            f"[{ev.evidence_id}] source={ev.source}; uri={ev.uri or 'NA'}; "
            f"cited={'yes' if ev.evidence_id in cited else 'no'}; text={ev.text}"
        )

    return (
        f"PROMPT_VERSION={prompt_version}\n"
        "You are an evidence-verification critic for knowledge-based visual question answering.\n"
        "Judge whether the candidate answer is justified by the supplied inference-time evidence.\n"
        "Use the visual-entity list only as auxiliary scene context; no gold answer is available.\n"
        "SUPPORTED: evidence directly and consistently supports the candidate answer.\n"
        "CONTRADICTED: supplied evidence materially conflicts with the candidate answer.\n"
        "INSUFFICIENT: evidence is missing, too weak, ambiguous, or unrelated.\n"
        "Return exactly one label and nothing else: SUPPORTED, CONTRADICTED, or INSUFFICIENT.\n\n"
        f"QUESTION: {question}\n"
        f"CANDIDATE_ANSWER: {answer}\n"
        f"VISUAL_ENTITIES: {', '.join(visual_entities) if visual_entities else 'NONE'}\n"
        f"CITED_EVIDENCE_IDS: {', '.join(cited_evidence_ids) if cited_evidence_ids else 'NONE'}\n"
        "EVIDENCE:\n" + ("\n".join(evidence_lines) if evidence_lines else "NONE")
    )


class QwenLabelLikelihoodCritic:
    LABELS = (
        CriticLabel.SUPPORTED,
        CriticLabel.CONTRADICTED,
        CriticLabel.INSUFFICIENT,
    )

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        *,
        device_map: str = "auto",
        load_in_4bit: bool = True,
        prompt_version: str = "evitrust-critic-v1",
        revision: str = "main",
    ):
        self.model_id = model_id
        self.revision = revision
        self.device_map = device_map
        self.load_in_4bit = bool(load_in_4bit)
        self.prompt_version = prompt_version
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quantization_config = None
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception:
                quantization_config = None

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision, trust_remote_code=False)
        kwargs = {
            "device_map": self.device_map,
            "torch_dtype": "auto",
            "trust_remote_code": False,
            "revision": self.revision,
        }
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self._model.eval()

    def _chat_prefix(self, prompt: str) -> str:
        self._load()
        messages = [
            {"role": "system", "content": "You are a strict factual evidence critic. Output one allowed label only."},
            {"role": "user", "content": prompt},
        ]
        return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _continuation_logprob(self, prefix: str, continuation: str) -> float:
        self._load()
        import torch

        tok = self._tokenizer
        model = self._model
        prefix_ids = tok(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"]
        full_ids = tok(prefix + continuation, return_tensors="pt", add_special_tokens=False)["input_ids"]
        n_prefix = prefix_ids.shape[1]
        if full_ids.shape[1] <= n_prefix:
            return float("-inf")
        device = next(model.parameters()).device
        full_ids = full_ids.to(device)
        with torch.inference_mode():
            logits = model(input_ids=full_ids).logits[:, :-1, :]
            targets = full_ids[:, 1:]
            logp = torch.log_softmax(logits.float(), dim=-1)
            token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        start = max(0, n_prefix - 1)
        return float(token_logp[:, start:].mean().item())

    def verify(
        self,
        *,
        question: str,
        answer: str,
        visual_entities: Sequence[str],
        evidence: Sequence[Evidence],
        cited_evidence_ids: Sequence[str],
    ) -> LLMVerificationResult:
        prompt = build_critic_prompt(
            question=question,
            answer=answer,
            visual_entities=visual_entities,
            evidence=evidence,
            cited_evidence_ids=cited_evidence_ids,
            prompt_version=self.prompt_version,
        )
        prefix = self._chat_prefix(prompt)
        log_scores = [self._continuation_logprob(prefix, label.value) for label in self.LABELS]
        probs = _normalise_log_scores(log_scores)
        ranked = sorted(zip(probs, self.LABELS), reverse=True, key=lambda x: x[0])
        top_prob, top_label = ranked[0]
        return LLMVerificationResult(
            supported=probs[0],
            contradicted=probs[1],
            insufficient=probs[2],
            label=top_label,
            margin=top_prob - ranked[1][0],
            prompt_version=self.prompt_version,
            model_id=self.model_id,
        )

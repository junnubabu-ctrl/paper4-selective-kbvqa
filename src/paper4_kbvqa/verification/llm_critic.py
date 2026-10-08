from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import hashlib
import json
from typing import Sequence

from paper4_kbvqa.types import Evidence
from paper4_kbvqa.generation.qwen_vl import _quantization_identity, _validate_loaded_revision


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
    raw_label_log_scores: dict[str, float] = field(default_factory=dict)
    rendered_prompt: str | None = None
    critic_prompt: str | None = None
    requested_revision: str | None = None
    resolved_revision: str | None = None
    loading_identity: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "supported": float(self.supported),
            "contradicted": float(self.contradicted),
            "insufficient": float(self.insufficient),
            "label": self.label.value,
            "margin": float(self.margin),
            "prompt_version": self.prompt_version,
            "model_id": self.model_id,
            "raw_label_log_scores": dict(self.raw_label_log_scores),
            "rendered_prompt": self.rendered_prompt,
            "critic_prompt": self.critic_prompt,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "loading_identity": dict(self.loading_identity),
            "quantization_mode": self.loading_identity.get("quantization_mode"),
            "load_mode": self.loading_identity.get("load_mode"),
        }


def _normalise_log_scores(log_scores: Sequence[float]) -> list[float]:
    if len(log_scores) != 3:
        raise ValueError("Exactly three label scores are required.")
    if not all(math.isfinite(x) for x in log_scores):
        raise ValueError("Critic label likelihoods must all be finite")
    m = max(log_scores)
    exps = [math.exp(x - m) for x in log_scores]
    z = sum(exps)
    if not math.isfinite(z) or z <= 0:
        raise ValueError("Critic probabilities cannot be normalized")
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
        self.loading_identity = None
        self._continuation_token_records = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install critic extras in a CUDA environment") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen critic benchmark execution requires a CUDA GPU")
        quantization_config = None
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True)
            except Exception as exc:
                raise RuntimeError("Critic 4-bit mode requested but compatible quantization is unavailable") from exc
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision, trust_remote_code=False)
        kwargs = {"device_map": self.device_map, "torch_dtype": "auto",
                  "trust_remote_code": False, "revision": self.revision}
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        resolved = _validate_loaded_revision(model, self.revision)
        quant_identity = _quantization_identity(model, self.load_in_4bit)
        device_map = {key: str(value) for key, value in getattr(model, "hf_device_map", {}).items()}
        try:
            device = str(next(model.parameters()).device)
        except (StopIteration, AttributeError):
            device = str(getattr(model, "device", "unknown"))
        if not (device.startswith("cuda") or any(v.startswith("cuda") or v.isdigit() for v in device_map.values())):
            raise RuntimeError("Loaded critic has no confirmed CUDA placement")
        model.eval()
        self.loading_identity = {"model_id": self.model_id, "requested_revision": self.revision,
            "resolved_revision": resolved, "device": device, "device_map": device_map,
            "cuda_device": torch.cuda.get_device_name(), "load_mode": "frozen_pretrained_cuda",
            **quant_identity}
        self._tokenizer, self._model = tokenizer, model

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
            raise RuntimeError("Critic label tokenization produced no continuation tokens")
        if not torch.equal(prefix_ids, full_ids[:, :n_prefix]):
            raise RuntimeError("Critic continuation changed the common tokenized prompt prefix")
        self._continuation_token_records[continuation] = full_ids[0, n_prefix:].tolist()
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
            raw_label_log_scores={label.value: float(score) for label, score in zip(self.LABELS, log_scores)},
            rendered_prompt=prefix, critic_prompt=prompt,
            requested_revision=self.revision,
            resolved_revision=getattr(getattr(self._model, "config", None), "_commit_hash", None),
            loading_identity={**(self.loading_identity or {}),
                "continuation_token_ids": dict(self._continuation_token_records),
                "rendered_prompt_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
                "critic_request_sha256": hashlib.sha256(json.dumps({"prefix": prefix,
                    "loading_identity": self.loading_identity}, sort_keys=True, default=str).encode()).hexdigest()},
        )

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import os
import time
from typing import Sequence

from paper4_kbvqa.data.manifest import VQASample
from paper4_kbvqa.execution.controller import (
    ExecutionConfig,
    leakage_safe_query,
    provenance_components,
    record_checksum,
    reliability_fusion,
)
from paper4_kbvqa.types import Evidence


VALID_VARIANTS = {"B0", "B1", "B2", "B3"}


class MatchedVariantRunner:
    """Run B0-B3 with exactly matched generator/query contracts.

    B0: image + question only.
    B1: B0 + raw retrieved evidence.
    B2: B1 + relevance filtering.
    B3: B2 + separate evidence critic and provenance-aware reliability fusion.

    Calibration and abstention are intentionally excluded here. They are fitted
    from B3 validation outputs and applied as B4/B5 in separate scripts.
    """

    def __init__(
        self,
        *,
        variant: str,
        generator,
        provider=None,
        evidence_filter=None,
        critic=None,
        config: ExecutionConfig | None = None,
        corruption=None,
    ):
        variant = str(variant).upper()
        if variant not in VALID_VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VALID_VARIANTS)}")
        if variant != "B0" and provider is None:
            raise ValueError(f"{variant} requires a knowledge provider")
        if variant in {"B2", "B3"} and evidence_filter is None:
            raise ValueError(f"{variant} requires an evidence filter")
        if variant == "B3" and critic is None:
            raise ValueError("B3 requires an evidence critic")
        self.variant = variant
        self.generator = generator
        self.provider = provider
        self.evidence_filter = evidence_filter
        self.critic = critic
        self.config = config or ExecutionConfig()
        self.corruption = corruption
        self.config.validate()

    def _retrieve(self, query: str) -> list[Evidence]:
        if self.provider is None:
            return []
        try:
            return list(self.provider.retrieve(query, limit=self.config.retrieval_top_k))
        except TypeError:
            return list(self.provider.retrieve(query, top_k=self.config.retrieval_top_k))

    def _filter(
        self,
        evidence: Sequence[Evidence],
        question: str,
        visual_entities: Sequence[str],
    ) -> tuple[list[Evidence], dict]:
        if self.evidence_filter is None:
            return list(evidence), {}
        if hasattr(self.evidence_filter, "select"):
            result = self.evidence_filter.select(
                list(evidence),
                question,
                list(visual_entities),
                top_k=self.config.filtered_top_k,
            )
            if isinstance(result, tuple) and len(result) == 2:
                selected, details = result
                return list(selected), dict(details)
            return list(result), {}
        result = self.evidence_filter.filter(
            question,
            list(evidence),
            visual_entities=list(visual_entities),
            top_k=self.config.filtered_top_k,
        )
        return list(result)[: self.config.filtered_top_k], {}

    @staticmethod
    def _field(obj, *names, default=None):
        if isinstance(obj, dict):
            for name in names:
                if name in obj:
                    return obj[name]
            return default
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return default

    def run_one(self, sample: VQASample) -> dict:
        t0 = time.perf_counter()

        visual_entities = list(sample.visual_entities)
        if (
            self.config.auto_extract_entities
            and not visual_entities
            and hasattr(self.generator, "extract_visual_entities")
        ):
            visual_entities = list(
                self.generator.extract_visual_entities(sample.image_path, sample.question)
            )

        query = leakage_safe_query(sample.question, visual_entities)

        retrieved: list[Evidence] = []
        selected: list[Evidence] = []
        filter_scores: dict = {}

        if self.variant != "B0":
            if hasattr(self.provider, 'retrieve_for_question'):
                retrieved=list(self.provider.retrieve_for_question(sample.question,visual_entities,
                    limit=self.config.retrieval_top_k))
            else:
                retrieved = self._retrieve(query)
            if self.corruption is not None:
                retrieved = self.corruption(retrieved, sample.question_id)

        if self.variant == "B1":
            selected = retrieved[: self.config.filtered_top_k]
        elif self.variant in {"B2", "B3"}:
            selected, filter_scores = self._filter(
                retrieved, sample.question, visual_entities
            )

        generated = self.generator.generate(
            image_path=sample.image_path,
            question=sample.question,
            evidence=selected,
        )
        answer = str(self._field(generated, "answer", default="")).strip()
        cited = list(
            self._field(
                generated,
                "supporting_evidence_ids",
                "evidence_ids",
                default=[],
            )
            or []
        )
        generation_confidence = float(
            self._field(generated, "raw_confidence", "confidence", default=0.0)
        )

        prov = provenance_components(selected, cited) if selected else {
            "citation_validity": 0.0,
            "source_traceability": 0.0,
            "source_diversity": 0.0,
            "composite": 0.0,
        }

        critic_payload = None
        raw_reliability = generation_confidence
        if self.variant == "B3":
            verified = self.critic.verify(
                question=sample.question,
                answer=answer,
                visual_entities=visual_entities,
                evidence=selected,
                cited_evidence_ids=cited,
            )
            support = float(self._field(verified, "supported", default=0.0))
            contradiction = float(self._field(verified, "contradicted", default=0.0))
            insufficiency = float(self._field(verified, "insufficient", default=1.0))
            raw_reliability = reliability_fusion(
                generation_confidence=generation_confidence,
                support_probability=support,
                contradiction_probability=contradiction,
                insufficiency_probability=insufficiency,
                provenance=prov["composite"],
                config=self.config,
            )
            critic_payload = {
                "supported": support,
                "contradicted": contradiction,
                "insufficient": insufficiency,
                "label": str(
                    self._field(
                        self._field(verified, "label", default=""),
                        "value",
                        default=self._field(verified, "label", default=""),
                    )
                ),
                "model_id": str(self._field(verified, "model_id", default="")),
                "prompt_version": str(
                    self._field(verified, "prompt_version", default="")
                ),
            }

        rec = {
            "variant": self.variant,
            "question_id": str(sample.question_id),
            "question": sample.question,
            "retrieval_query": query,
            "visual_entities": visual_entities,
            "answer": answer,
            "selective_answer": answer,
            "supporting_evidence_ids": cited,
            "evidence": [asdict(e) for e in selected],
            "retrieved_evidence_ids": [str(e.evidence_id) for e in retrieved],
            "retrieved_evidence": [asdict(e) for e in retrieved],
            "filtered_evidence_ids": [str(e.evidence_id) for e in selected],
            "filter_scores": filter_scores,
            "generation_confidence": generation_confidence,
            "critic": critic_payload,
            "provenance": prov,
            "provenance_score": prov["composite"],
            "raw_reliability": raw_reliability,
            "raw_confidence": raw_reliability,
            "calibrated_confidence": None,
            "threshold": None,
            "abstain": None,
            "latency_s": time.perf_counter() - t0,
            "runtime_prompt_version": self.config.prompt_version,
        }
        rec["record_sha256"] = record_checksum(rec)
        return rec

    def run_manifest(
        self,
        samples: Sequence[VQASample],
        *,
        output_jsonl: str | Path,
        checkpoint_json: str | Path,
    ) -> dict:
        output_jsonl = Path(output_jsonl)
        checkpoint_json = Path(checkpoint_json)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_json.parent.mkdir(parents=True, exist_ok=True)

        completed: set[str] = set()
        errors: list[dict] = []
        # Predictions are authoritative: a crash can occur between the durable
        # prediction write and the checkpoint update. Never duplicate that row.
        expected = {str(s.question_id) for s in samples}
        if len(expected) != len(samples):
            raise ValueError("Duplicate manifest question IDs")
        if output_jsonl.exists():
            with output_jsonl.open("rb+") as existing:
                while True:
                    offset = existing.tell()
                    line = existing.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        # Only an interrupted final write may be discarded.
                        existing.truncate(offset)
                        break
                    rec = json.loads(line)
                    qid = str(rec["question_id"])
                    if qid in completed or qid not in expected:
                        raise ValueError("Duplicate or foreign prediction ID: " + qid)
                    checksum = rec.pop("record_sha256", None)
                    if rec.get("variant") != self.variant or checksum != record_checksum(rec):
                        raise ValueError("Prediction checksum/variant mismatch: " + qid)
                    completed.add(qid)

        with output_jsonl.open("a", encoding="utf-8") as out:
            for sample in samples:
                qid = str(sample.question_id)
                if qid in completed:
                    continue
                try:
                    rec = self.run_one(sample)
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    os.fsync(out.fileno())
                    completed.add(qid)
                except Exception as exc:
                    errors.append({"question_id": qid, "error": repr(exc)})

                tmp = checkpoint_json.with_suffix(checkpoint_json.suffix + ".tmp")
                tmp.write_text(
                    json.dumps(
                        {
                            "variant": self.variant,
                            "completed_question_ids": sorted(completed),
                            "count": len(completed),
                            "errors": errors,
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                tmp.replace(checkpoint_json)
                print(f"{self.variant}: {len(completed)}/{len(samples)} completed; {len(errors)} errors", flush=True)
                if errors:
                    # Surface download/model/CUDA failures immediately instead
                    # of looping through the complete dataset and reporting success.
                    raise RuntimeError(f"Inference failed; resume after resolving: {errors[-1]}")

        return {
            "variant": self.variant,
            "completed": len(completed),
            "errors": len(errors),
            "completed_question_ids": sorted(completed),
        }

"""Development-only, fixed-candidate Paper 4 execution.

One filtered candidate is durably archived before criticism. The critic never
regenerates it. This module does not fit calibration, select thresholds or score
held-out benchmarks. SHA256 identities detect drift; they are not signatures.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import time
from datetime import datetime, timezone
from typing import Any, Sequence

from paper4_kbvqa.data.manifest import VQASample
from paper4_kbvqa.execution.controller import (
    ExecutionConfig, leakage_safe_query, provenance_components, reliability_fusion,
)
from paper4_kbvqa.types import Evidence

FORBIDDEN_INFERENCE_KEYS = {
    "answers", "direct_answers", "multiple_choice_answer", "correct_choice_idx",
    "correct_choice", "rationales", "rationale", "gold_answer", "ground_truth",
    "correct_answer", "label", "labels",
}


def plain(value: Any) -> Any:
    """Retain diagnostics as strict JSON, including explicit nonfinite values."""
    if isinstance(value, Enum):
        return plain(value.value)
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": str(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unserializable audit value: {type(value).__name__}")


def digest(value: Any) -> str:
    payload = json.dumps(plain(value), sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while data := f.read(1024 * 1024):
            h.update(data)
    return h.hexdigest()


def assert_no_gold_keys(value: Any, location: str = "inference") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_INFERENCE_KEYS:
                raise ValueError(f"Forbidden gold/label field at {location}.{key}")
            assert_no_gold_keys(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            assert_no_gold_keys(child, f"{location}[{i}]")


def durable_json(path: Path, record: dict) -> None:
    """Atomic write, fsync file and directory; never overwrite archived records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite archive: {path}")
    data = plain(record)
    envelope = {"record": data, "record_sha256": digest(data)}
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(envelope, f, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read_verified(path: Path) -> dict:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if set(envelope) != {"record", "record_sha256"}:
        raise ValueError(f"Invalid archive envelope: {path}")
    if digest(envelope["record"]) != envelope["record_sha256"]:
        raise ValueError(f"Archive checksum mismatch: {path}")
    return envelope["record"]


def frozen_json(path: Path, record: dict) -> None:
    if path.exists():
        if read_verified(path) != plain(record):
            raise ValueError(f"Frozen run identity changed: {path}")
    else:
        durable_json(path, record)


def inference_identity(sample: VQASample) -> dict:
    """No answers/rationales/choices or arbitrary metadata cross this boundary."""
    image = Path(sample.image_path)
    return {
        "question_id": str(sample.question_id),
        "question": sample.question,
        "image_path": str(image.resolve()),
        "image_sha256": file_digest(image) if image.is_file() else None,
        "visual_entities": list(sample.visual_entities),
        "image_id": str(sample.metadata.get("image_id", "")),
        "dataset": str(sample.metadata.get("dataset", "")),
        "source_split": str(sample.metadata.get("source_split", sample.metadata.get("split", ""))),
    }


def validate_image(sample: VQASample) -> dict:
    from PIL import Image
    image = Path(sample.image_path)
    if not image.is_file():
        raise ValueError("Image file is missing")
    with Image.open(image) as im:
        dims = list(im.size)
        image_format = im.format
        im.verify()
    return {"native_dimensions": dims, "image_format": image_format,
            "image_sha256": file_digest(image)}


def validate_evidence(evidence: Sequence[Evidence]) -> None:
    ids = []
    for item in evidence:
        if not isinstance(item, Evidence):
            raise ValueError("Evidence must use the audited Evidence dataclass")
        if not item.evidence_id.strip() or not item.text.strip() or not item.source.strip():
            raise ValueError("Evidence ID, text and source must be nonempty")
        if not item.uri or not str(item.uri).strip():
            raise ValueError("Evidence requires a source URI")
        if not math.isfinite(float(item.retrieval_score)):
            raise ValueError("Evidence retrieval score must be finite")
        assert_no_gold_keys(asdict(item), "evidence")
        ids.append(item.evidence_id)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evidence IDs")


class FrozenQuestionEvidence:
    """Question-indexed, inference-only evidence snapshot. No live retrieval."""
    def __init__(self, rows: dict[str, list[Evidence]]):
        self.rows = rows
        for values in rows.values():
            validate_evidence(values)

    def retrieve_for_question_id(self, question_id: str, *, limit: int) -> list[Evidence]:
        if str(question_id) not in self.rows:
            raise ValueError("Question is absent from the frozen evidence snapshot")
        return list(self.rows[str(question_id)][:limit])


class FixedCandidatePilot:
    def __init__(self, *, generator, critic, provider, evidence_filter,
                 execution_config: ExecutionConfig | None = None,
                 run_config: dict, source_identity: dict, stop_on_invalid: bool = True):
        self.generator = generator
        self.critic = critic
        self.provider = provider
        self.evidence_filter = evidence_filter
        self.config = execution_config or ExecutionConfig()
        self.config.validate()
        assert_no_gold_keys(run_config, "run_config")
        assert_no_gold_keys(source_identity, "source_identity")
        self.run_config = plain(run_config)
        self.source_identity = plain(source_identity)
        self.stop_on_invalid = bool(stop_on_invalid)

    def _context(self, sample: VQASample, *, input_id: dict,
                 run_sha: str) -> dict:
        started = time.perf_counter()
        record = {"schema": "paper4-fixed-context-v1", "stage": "train-development",
                  "question_id": str(sample.question_id), "input_identity": input_id,
                  "run_identity_sha256": run_sha, "status": "INVALID",
                  "errors": [], "captured_utc": datetime.now(timezone.utc).isoformat(),
                  "source_identity": self.source_identity,
                  "provenance_boundary": "saved_inference_context_not_licensed_immutable_knowledge_snapshot"}
        try:
            record["image_audit"] = validate_image(sample)
            if record["image_audit"]["image_sha256"] != input_id["image_sha256"]:
                raise ValueError("Image changed after input identity was frozen")
            if not sample.question.strip() or not str(sample.question_id).strip():
                raise ValueError("Question and question ID must be nonempty")
            entities = list(sample.visual_entities)
            extracted = self.config.auto_extract_entities and not entities
            if extracted:
                entities = list(self.generator.extract_visual_entities(sample.image_path, sample.question))
                grounding = getattr(self.generator, "last_entity_record", None)
                if grounding is not None:
                    record["visual_grounding_record"] = plain(grounding)
                    if grounding.get("parser_status") != "valid":
                        # Manuscript-approved fallback: empty entities, question
                        # keywords only; no gold-answer repair or invented scene facts.
                        entities = []
                        record["grounding_fallback"] = "empty_entities_question_keywords_only"
                    if grounding.get("question", sample.question) != sample.question:
                        raise ValueError("Visual grounding belongs to a different question")
            if any(not isinstance(x, str) or not x.strip() for x in entities):
                raise ValueError("Visual entities must be nonempty strings")
            query = leakage_safe_query(sample.question, entities)
            if hasattr(self.provider, "retrieve_for_question_id"):
                retrieved = list(self.provider.retrieve_for_question_id(
                    sample.question_id, limit=self.config.retrieval_top_k))
            elif hasattr(self.provider, "retrieve_for_question"):
                retrieved = list(self.provider.retrieve_for_question(
                    sample.question, entities, limit=self.config.retrieval_top_k))
            else:
                retrieved = list(self.provider.retrieve(query, limit=self.config.retrieval_top_k))
            # Retain malformed source records as diagnostics before validation.
            record["retrieved_evidence"] = plain(retrieved)
            validate_evidence(retrieved)
            record["retrieved_pool"] = record["retrieved_evidence"]
            retrieved = retrieved[:self.config.retrieval_top_k]
            record["retrieved_evidence"] = [asdict(e) for e in retrieved]
            selected, details = self.evidence_filter.select(
                list(retrieved), sample.question, entities,
                top_k=self.config.filtered_top_k)
            selected = list(selected)
            validate_evidence(selected)
            original = {e.evidence_id: asdict(e) for e in retrieved}
            if any(asdict(e) != original.get(e.evidence_id) for e in selected):
                raise ValueError("Filter changed or invented an evidence record")
            context = {"question": sample.question, "visual_entities": entities,
                       "ordered_evidence": [asdict(e) for e in selected],
                       "image_sha256": input_id["image_sha256"]}
            assert_no_gold_keys(context, "candidate_context")
            record.update({"retrieval_query": query, "filter_details": plain(details),
                           "context": context, "context_sha256": digest(context),
                           "status": "ARCHIVED_VALID_CONTEXT"})
        except Exception as exc:
            record["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        record["elapsed_s"] = time.perf_counter() - started
        return record

    def _candidate(self, sample: VQASample, *, input_id: dict,
                   run_sha: str, context_record: dict) -> dict:
        started = time.perf_counter()
        record = {"schema": "paper4-fixed-candidate-v1", "stage": "train-development",
                  "question_id": str(sample.question_id), "input_identity": input_id,
                  "run_identity_sha256": run_sha, "status": "INVALID",
                  "candidate_eligible": False, "errors": [],
                  "context_record_sha256": digest(context_record)}
        if context_record["status"] != "ARCHIVED_VALID_CONTEXT":
            record["errors"] = ["context_invalid_no_generator_execution"]
            return record
        try:
            context = context_record["context"]
            if digest(context) != context_record["context_sha256"]:
                raise ValueError("Frozen context checksum mismatch")
            selected = [Evidence(**{**item, "entity_ids": tuple(item.get("entity_ids", []))})
                        for item in context["ordered_evidence"]]
            record.update({"context": context, "context_sha256": context_record["context_sha256"],
                           "source_identity": self.source_identity,
                           "image_audit": context_record["image_audit"]})
            if file_digest(sample.image_path) != input_id["image_sha256"]:
                raise ValueError("Image changed after the context was archived")
            # The ONLY answer-generation call for this question.
            generated = self.generator.generate(sample.image_path, sample.question, selected)
            if not isinstance(generated, dict):
                raise ValueError("Generator result must be a dictionary")
            record["generator_result"] = plain(generated)
            issues = []
            generated_image_hash = generated.get("generation_identity", {}).get("image_sha256")
            if generated_image_hash is not None and generated_image_hash != input_id["image_sha256"]:
                issues.append("generator_image_identity_mismatch")
            answer = generated.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                issues.append("empty_or_nonstring_answer")
            if generated.get("parsed_ok") is not True:
                issues.append("generator_output_not_schema_validated")
            confidence = generated.get("raw_confidence")
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not math.isfinite(float(confidence)) or not 0 <= confidence <= 1:
                issues.append("invalid_generator_confidence")
            raw_ids = generated.get("raw_evidence_ids", generated.get("supporting_evidence_ids", []))
            if not isinstance(raw_ids, list) or any(not isinstance(x, str) for x in raw_ids):
                issues.append("invalid_citation_list")
                raw_ids = []
            available = {e.evidence_id for e in selected}
            # Unknown/repeated string identifiers are schema-valid outputs.
            # Retain them for distinct-identifier provenance scoring; they do not
            # exclude candidates or silently change the prescribed denominator.
            invalid_ids = [x for x in raw_ids if x not in available]
            valid_ids = [x for x in raw_ids if x in available]
            repeated_ids = [x for x in dict.fromkeys(raw_ids) if raw_ids.count(x) > 1]
            citation_diagnostics = {
                "repeated_identifier_values": repeated_ids,
                "unknown_identifier_occurrences": invalid_ids,
                "distinct_raw_identifier_count": len(set(raw_ids)),
                "distinct_valid_identifier_count": len(set(valid_ids)),
            }
            record["citation_diagnostics"] = citation_diagnostics
            candidate = {"answer": answer if isinstance(answer, str) else None,
                         "context_sha256": record["context_sha256"],
                         "raw_evidence_ids": raw_ids, "valid_evidence_ids": valid_ids,
                         "invalid_evidence_ids": invalid_ids,
                         "generator_result_sha256": digest(record["generator_result"])}
            record["candidate"] = candidate
            record["candidate_sha256"] = digest(candidate)
            record["errors"].extend(issues)
            if not issues:
                record["status"] = "ARCHIVED_VALID_CANDIDATE"
                record["candidate_eligible"] = True
        except Exception as exc:
            record["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        record["elapsed_s"] = time.perf_counter() - started
        return record

    def _critique(self, candidate_record: dict) -> dict:
        result = {"schema": "paper4-fixed-critic-v1", "stage": "train-development",
                  "question_id": candidate_record["question_id"],
                  "run_identity_sha256": candidate_record["run_identity_sha256"],
                  "candidate_sha256": candidate_record.get("candidate_sha256"),
                  "candidate_record_sha256": digest(candidate_record),
                  "context_sha256": candidate_record.get("context_sha256"),
                  "status": "INVALID", "errors": []}
        if not candidate_record["candidate_eligible"]:
            result["errors"] = ["candidate_invalid_no_critic_execution"]
            return result
        started = time.perf_counter()
        try:
            context = candidate_record["context"]
            candidate = candidate_record["candidate"]
            if digest(context) != candidate_record["context_sha256"] or digest(candidate) != candidate_record["candidate_sha256"]:
                raise ValueError("Candidate or context hash mismatch")
            evidence = [Evidence(**{**item, "entity_ids": tuple(item.get("entity_ids", []))})
                        for item in context["ordered_evidence"]]
            verified = self.critic.verify(
                question=context["question"], answer=candidate["answer"],
                visual_entities=context["visual_entities"], evidence=evidence,
                cited_evidence_ids=candidate["valid_evidence_ids"])
            raw = verified.as_dict() if hasattr(verified, "as_dict") else verified
            if not isinstance(raw, dict):
                raise ValueError("Critic result must be dictionary-compatible")
            result["critic_result"] = plain(raw)
            scores = [raw.get(x) for x in ["supported", "contradicted", "insufficient"]]
            if any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(float(x)) or not 0 <= x <= 1 for x in scores):
                raise ValueError("Invalid critic probability")
            if abs(sum(scores) - 1) > 1e-5:
                raise ValueError("Critic probabilities do not sum to one")
            if str(raw.get("label")) not in {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT"}:
                raise ValueError("Invalid critic label")
            prov = provenance_components(evidence, candidate["raw_evidence_ids"])
            raw_likelihood = candidate_record["generator_result"]["raw_confidence"]
            fusion = reliability_fusion(
                generation_confidence=raw_likelihood, support_probability=scores[0],
                contradiction_probability=scores[1], insufficiency_probability=scores[2],
                provenance=prov["composite"], config=self.config)
            common = {"answer": candidate["answer"],
                      "candidate_sha256": candidate_record["candidate_sha256"],
                      "context_sha256": candidate_record["context_sha256"],
                      "calibrated_confidence": None, "threshold": None, "abstain": None}
            result.update({"status": "VALID_MATCHED_DEVELOPMENT_RECORD",
                           "B2": {**common, "raw_score": raw_likelihood,
                                  "score_definition": "whole_response_token_likelihood"},
                           "B3": {**common, "raw_score": fusion,
                                  "score_definition": "heuristic_evidence_reliability_fusion"},
                           "provenance": prov,
                           "calibration_implemented": False,
                           "benchmark_result": False})
        except Exception as exc:
            result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        result["elapsed_s"] = time.perf_counter() - started
        return result

    def run_manifest(self, samples: Sequence[VQASample], *, output_dir: str | Path) -> dict:
        output = Path(output_dir)
        qids = [str(s.question_id) for s in samples]
        if len(qids) != len(set(qids)):
            raise ValueError("Duplicate manifest question IDs")
        if not samples:
            raise ValueError("Pilot manifest is empty")
        identities = [inference_identity(s) for s in samples]
        for identity in identities:
            if identity["source_split"] != "train":
                raise ValueError("Pilot accepts TRAIN-derived development records only")
        run = {"schema": "paper4-fixed-pilot-v1", "stage": "train-development",
               "execution_config": asdict(self.config), "run_config": self.run_config,
               "source_identity": self.source_identity, "inputs": identities,
               "input_order_sha256": digest(identities),
               "benchmark_result": False, "calibration_implemented": False,
               "stop_on_invalid": self.stop_on_invalid}
        frozen_json(output / "run_identity.json", run)
        run_sha = digest(run)
        expected_stems = {hashlib.sha256(qid.encode()).hexdigest() for qid in qids}
        for subdir in ["contexts", "candidates", "critics"]:
            if (output / subdir).exists():
                foreign = [p.name for p in (output / subdir).glob("*.json") if p.stem not in expected_stems]
                if foreign:
                    raise ValueError(f"Foreign archived question records in {subdir}: {foreign}")
        valid = 0
        invalid = 0
        matched = []
        for sample, identity in zip(samples, identities):
            key = hashlib.sha256(str(sample.question_id).encode()).hexdigest()
            context_path = output / "contexts" / (key + ".json")
            candidate_path = output / "candidates" / (key + ".json")
            critic_path = output / "critics" / (key + ".json")
            if context_path.exists():
                context = read_verified(context_path)
                if context.get("input_identity") != identity or context.get("run_identity_sha256") != run_sha:
                    raise ValueError("Archived context input/config changed")
            else:
                context = self._context(sample, input_id=identity, run_sha=run_sha)
                durable_json(context_path, context)
            context = read_verified(context_path)
            if candidate_path.exists():
                candidate = read_verified(candidate_path)
                if candidate.get("input_identity") != identity or candidate.get("run_identity_sha256") != run_sha or candidate.get("context_record_sha256") != digest(context):
                    raise ValueError("Archived candidate input/config changed")
            else:
                candidate = self._candidate(sample, input_id=identity, run_sha=run_sha, context_record=context)
                durable_json(candidate_path, candidate)
            # Re-read from durable storage; never pass an unarchived generation.
            candidate = read_verified(candidate_path)
            if candidate.get("context_record_sha256") != digest(context):
                raise ValueError("Candidate uses a changed context archive")
            if candidate.get("candidate_eligible"):
                if candidate.get("context") != context.get("context") or digest(candidate["context"]) != candidate.get("context_sha256"):
                    raise ValueError("Candidate context identity mismatch")
                if digest(candidate["candidate"]) != candidate.get("candidate_sha256"):
                    raise ValueError("Candidate identity mismatch")
                if candidate["candidate"].get("generator_result_sha256") != digest(candidate["generator_result"]):
                    raise ValueError("Candidate generator identity mismatch")
            if critic_path.exists():
                critique = read_verified(critic_path)
                if critique.get("candidate_record_sha256") != digest(candidate) or critique.get("run_identity_sha256") != run_sha:
                    raise ValueError("Archived critic candidate/config changed")
            else:
                critique = self._critique(candidate)
                durable_json(critic_path, critique)
            if critique["status"] == "VALID_MATCHED_DEVELOPMENT_RECORD":
                for name in ["B2", "B3"]:
                    item = critique[name]
                    if item["answer"] != candidate["candidate"]["answer"] or item["candidate_sha256"] != candidate["candidate_sha256"] or item["context_sha256"] != candidate["context_sha256"]:
                        raise ValueError("Matched result changed the fixed candidate/context")
            matched.append({"question_id": str(sample.question_id),
                            "context_record_sha256": digest(context),
                            "candidate_record_sha256": digest(candidate),
                            "critic_record_sha256": digest(critique),
                            "candidate_status": candidate["status"],
                            "critic_status": critique["status"],
                            "B2": critique.get("B2"), "B3": critique.get("B3")})
            if critique["status"] == "VALID_MATCHED_DEVELOPMENT_RECORD":
                valid += 1
            else:
                invalid += 1
                if self.stop_on_invalid:
                    break
        summary = {"schema": "paper4-fixed-pilot-summary-v1",
                   "stage": "train-development", "requested": len(samples),
                   "valid": valid, "invalid": invalid,
                   "attempted": len(matched), "unattempted": len(samples) - len(matched),
                   "status": ("BLOCKED_INVALID_DEVELOPMENT_PILOT" if len(matched) < len(samples)
                              else "COMPLETED_VALID_DEVELOPMENT_PILOT" if not invalid
                              else "COMPLETED_INVALID_DEVELOPMENT_PILOT"),
                   "run_identity_sha256": run_sha, "matched": matched,
                   "benchmark_result": False, "calibration_implemented": False,
                   "scientific_claims_validated": False}
        # The summary is immutable after all requested records have been archived.
        frozen_json(output / "pilot_summary.json", summary)
        return summary

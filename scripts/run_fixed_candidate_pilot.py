#!/usr/bin/env python3
"""Bounded TRAIN-derived pilot; archive one filtered candidate then critique it.

No gold answers reach model inputs. No calibration, held-out evaluation or
benchmark accuracy is performed. Live retrieval is captured once and frozen
per question; static inference-only evidence is an optional alternative.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from paper4_kbvqa.data.manifest import VQASample
from paper4_kbvqa.execution.controller import ExecutionConfig
from paper4_kbvqa.execution.fixed_candidate import (
    FixedCandidatePilot, FrozenQuestionEvidence, assert_no_gold_keys,
    digest, file_digest,
)
from paper4_kbvqa.types import Evidence

BASE_SOURCE_COMMIT = "45190e6d2bee19e9805751fd1bed9aa1430ec43d"


def source_code_identity() -> dict:
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    files = {str(p.relative_to(ROOT)): file_digest(p) for p in paths}
    return {"base_source_commit": BASE_SOURCE_COMMIT, "modified_source_sha256": digest(files),
            "source_files": files}


class CUDASessionResources:
    """Session diagnostics outside frozen identities; no benchmark interpretation.

    Peaks are PyTorch allocator counters for this process on each CUDA device.
    They exclude driver/external-process memory and are not minimum VRAM claims.
    """
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.started = time.perf_counter()
        self.session_id = uuid.uuid4().hex
        self.torch = None
        self.devices = []
        self.errors = []
        self.available = False

    def begin(self, torch_module):
        self.torch = torch_module
        try:
            self.available = bool(torch_module.cuda.is_available())
            if not self.available:
                self.errors.append({"operation": "cuda_available", "error": "CUDA unavailable; allocator peaks not measured"})
                return
            count = int(torch_module.cuda.device_count())
            for index in range(count):
                row = {"index": index, "name": None,
                       "initial_allocated_bytes": None, "initial_reserved_bytes": None,
                       "peak_reset_succeeded": False,
                       "peak_allocated_bytes": None, "peak_reserved_bytes": None,
                       "errors": []}
                try:
                    row["name"] = torch_module.cuda.get_device_name(index)
                    torch_module.cuda.synchronize(index)
                    row["initial_allocated_bytes"] = int(torch_module.cuda.memory_allocated(index))
                    row["initial_reserved_bytes"] = int(torch_module.cuda.memory_reserved(index))
                    torch_module.cuda.reset_peak_memory_stats(index)
                    row["peak_reset_succeeded"] = True
                except Exception as error:
                    row["errors"].append({"operation": "begin", "error": f"{type(error).__name__}: {error}"})
                self.devices.append(row)
        except Exception as error:
            self.errors.append({"operation": "begin", "error": f"{type(error).__name__}: {error}"})

    def finish(self, status, *, error=None, pilot_status=None):
        if self.torch is None:
            self.errors.append({"operation": "torch_import", "error": "PyTorch runtime was not initialized; CUDA counters unavailable"})
        for row in self.devices:
            index = row["index"]
            try:
                self.torch.cuda.synchronize(index)
            except Exception as sync_error:
                row["errors"].append({"operation": "finish_synchronize", "error": f"{type(sync_error).__name__}: {sync_error}"})
            if row["peak_reset_succeeded"]:
                for field, method in [("peak_allocated_bytes", "max_memory_allocated"),
                                      ("peak_reserved_bytes", "max_memory_reserved")]:
                    try:
                        value = int(getattr(self.torch.cuda, method)(index))
                        if value < 0:
                            raise ValueError("Negative CUDA allocator counter")
                        row[field] = value
                    except Exception as counter_error:
                        row["errors"].append({"operation": method, "error": f"{type(counter_error).__name__}: {counter_error}"})
        finished = datetime.now(timezone.utc).isoformat()
        record = {"schema": "paper4-pilot-session-resources-v1", "session_id": self.session_id,
                  "stage": "train-development", "status": status,
                  "pilot_status": pilot_status, "error": error,
                  "started_utc": self.started_utc, "finished_utc": finished,
                  "session_wall_time_s": time.perf_counter() - self.started,
                  "cuda_available": self.available, "devices": self.devices,
                  "counter_errors": self.errors,
                  "measurement_scope": "PyTorch allocator in this process, per CUDA device; driver/external memory excluded",
                  "multi_device_peaks_are_not_summed": True,
                  "benchmark_result": False, "calibration_implemented": False}
        # Append and fsync every invocation, including failed/resumed sessions.
        # Runtime measurements never enter the frozen run/config identity.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "resource_sessions.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(self.output_dir, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return record

def load_pilot_inputs(args):
    """Validated prepared data is preferred; raw manifests need an explicit attestation."""
    if args.pilot_dir:
        from paper4_kbvqa.data.restart_pilot import validate_prepared_pilot
        checked = validate_prepared_pilot(Path(args.pilot_dir), require_images=True)
        rows = checked["rows"]
        samples = [VQASample(str(row["question_id"]), row["image_path"], row["question"],
                             metadata=row["metadata"]) for row in rows[:args.max_samples]]
        identity = {"mode": "validated_prepared_train_development",
                    "source_manifest_sha256": checked["source_manifest_sha256"],
                    "image_inventory_sha256": checked["image_inventory_sha256"],
                    "inference_manifest_sha256": checked["source"]["inference_manifest_sha256"],
                    "source_split": "train", "partition_role": "development_only"}
        return samples, identity
    if not args.train_development_attestation:
        raise ValueError("Raw --manifest requires --train-development-attestation JSON; use --pilot-dir for verified prepared data")
    manifest = Path(args.manifest)
    attestation = json.loads(Path(args.train_development_attestation).read_text())
    assert_no_gold_keys(attestation, "train_development_attestation")
    if attestation.get("source_split") != "train" or attestation.get("partition_role") != "development_only" or attestation.get("manifest_sha256") != file_digest(manifest):
        raise ValueError("Attestation must declare train/development_only and match the complete manifest SHA256")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    samples = []
    for row in rows:
        assert_no_gold_keys(row, "raw_inference_manifest")
        if set(row) - {"question_id", "question", "image_path", "visual_entities", "metadata"}:
            raise ValueError("Raw manifest contains unapproved inference fields")
        metadata = dict(row.get("metadata", {}))
        if metadata.get("source_split", metadata.get("split", "train")) != "train":
            raise ValueError("Raw manifest conflicts with TRAIN attestation")
        metadata["source_split"] = "train"
        image = Path(row["image_path"])
        if not image.is_absolute():
            image = manifest.parent / image
        samples.append(VQASample(str(row["question_id"]), str(image.resolve()),
                                 row["question"], visual_entities=tuple(row.get("visual_entities", [])),
                                 metadata=metadata))
    ids = [s.question_id for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Raw manifest contains duplicate question IDs")
    identity = {"mode": "user_attested_label_free_train_development",
                "manifest_sha256": file_digest(manifest),
                "train_attestation_sha256": file_digest(args.train_development_attestation),
                "source_split": "train", "partition_role": "development_only",
                "upstream_split_authenticity_verified": False}
    return samples[:args.max_samples], identity

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--pilot-dir", help="Verified prepared 50-record TRAIN development folder")
    inputs.add_argument("--manifest", help="Alternative label-free manifest; explicit TRAIN attestation required")
    parser.add_argument("--train-development-attestation", help="JSON train/development-only attestation with complete manifest SHA256")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stage", choices=["train-development"], default="train-development")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--sources", default="wikipedia,wikidata,conceptnet")
    parser.add_argument("--retrieval-top-k", type=int, default=10)
    parser.add_argument("--filtered-top-k", type=int, default=5)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--evidence-jsonl", help="Optional frozen question-indexed evidence alternative")
    parser.add_argument("--source-ledger", help="Optional source provenance/licence ledger; SHA256 locked")
    parser.add_argument("--model-lock", default=str(ROOT / "configs/models_20260915.json"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-auto-entities", action="store_true")
    parser.add_argument("--plan", action="store_true", help="Print bounded plan without reading images/loading models")
    args = parser.parse_args(argv)
    if not 1 <= args.max_samples <= 50:
        parser.error("This development pilot is bounded to 1–50 TRAIN-derived records")
    if args.max_pixels < 28 * 28:
        parser.error("max-pixels must be at least one 28-by-28 model image block")
    sources = [x.strip().lower() for x in args.sources.split(",") if x.strip()]
    if not sources or len(sources) != len(set(sources)) or set(sources) - {"wikipedia", "wikidata", "conceptnet"}:
        parser.error("sources must be a unique subset of wikipedia,wikidata,conceptnet")
    config = ExecutionConfig(retrieval_top_k=args.retrieval_top_k,
                             filtered_top_k=args.filtered_top_k,
                             auto_extract_entities=not args.no_auto_entities,
                             prompt_version="evitrust-fixed-development-v1")
    config.validate()
    pins = json.loads(Path(args.model_lock).read_text())
    for name in ["generator", "qwen"]:
        revision = pins[name]["revision"]
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Pilot requires exact 40-character upstream model revisions")
    plan = {"stage": args.stage, "max_samples": args.max_samples,
            "source_mode": "static_evidence_alternative" if args.evidence_jsonl else "live_capture_once",
            "sources": sources, "models": {x: pins[x] for x in ["generator", "qwen"]},
            "max_pixels": args.max_pixels, "four_bit": not args.no_4bit,
            "execution_config": asdict(config), "base_source_commit": BASE_SOURCE_COMMIT,
            "benchmark_result": False, "calibration_implemented": False,
            "completion_gate": "Every requested input, candidate and critic record must validate",
            "provenance_boundary": "saved_context_not_licensed_immutable_knowledge_snapshot"}
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0

    samples, data_identity = load_pilot_inputs(args)
    if not samples:
        raise ValueError("Empty pilot manifest")
    for sample in samples:
        split = sample.metadata.get("source_split", sample.metadata.get("split"))
        if split != "train":
            raise ValueError("Manifest must declare source_split=train; held-out evaluation is prohibited")
    source_identity = {"mode": plan["source_mode"], "sources": sources,
                       "provider_config": {"retrieval_top_k": args.retrieval_top_k,
                                           "timeout_seconds": 15},
                       "upstream_revisions_verified": False}
    if args.source_ledger:
        source_identity["source_ledger_sha256"] = file_digest(args.source_ledger)
    output = Path(args.out_dir)
    if args.evidence_jsonl:
        rows = {}
        for line in Path(args.evidence_jsonl).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            assert_no_gold_keys(row, "static_evidence")
            if set(row) != {"question_id", "evidence"}:
                raise ValueError("Static evidence rows require ONLY question_id and evidence")
            qid = str(row["question_id"])
            if qid in rows:
                raise ValueError("Duplicate static evidence question ID")
            rows[qid] = [Evidence(**{**x, "entity_ids": tuple(x.get("entity_ids", []))})
                         for x in row["evidence"]]
        provider = FrozenQuestionEvidence(rows)
        source_identity["evidence_snapshot_sha256"] = file_digest(args.evidence_jsonl)
        source_identity["evidence_question_ids_sha256"] = digest(sorted(rows))
    else:
        from paper4_kbvqa.knowledge.http_providers import WikipediaProvider, WikidataProvider, ConceptNetProvider
        from paper4_kbvqa.knowledge.structured import StructuredKnowledgeProvider
        mapping = {"wikipedia": WikipediaProvider, "wikidata": WikidataProvider,
                   "conceptnet": ConceptNetProvider}
        provider = StructuredKnowledgeProvider([
            mapping[name](output / "source_cache" / name, timeout=15) for name in sources])

    session = CUDASessionResources(output)
    session_status = "FAILED"
    session_error = None
    pilot_status = None
    try:
        import torch
        session.begin(torch)
        if not torch.cuda.is_available():
            raise RuntimeError("A real CUDA GPU is required for this pilot; no model predictions were generated")
        from paper4_kbvqa.utils.seed import set_seed
        from paper4_kbvqa.utils.environment import collect_environment
        from paper4_kbvqa.filtering.relevance import RelevanceFilter
        from paper4_kbvqa.generation.qwen_vl import Qwen25VLGenerator
        from paper4_kbvqa.verification.llm_critic import QwenLabelLikelihoodCritic
        set_seed(args.seed)
        versions = {}
        for name in ["torch", "transformers", "accelerate", "bitsandbytes", "numpy", "scikit-learn", "pillow"]:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        runtime = {"python": sys.version, "packages": versions,
                   "gpu_name": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda}
        generator = Qwen25VLGenerator(
            model_name=pins["generator"]["model_id"], revision=pins["generator"]["revision"],
            load_in_4bit=not args.no_4bit, max_pixels=args.max_pixels)
        critic = QwenLabelLikelihoodCritic(
            model_id=pins["qwen"]["model_id"], revision=pins["qwen"]["revision"],
            load_in_4bit=not args.no_4bit, prompt_version="evitrust-critic-v1")
        run_config = {"plan": plan, "seed": args.seed,
                      "data_identity": data_identity,
                      "model_lock_sha256": file_digest(args.model_lock),
                      "code_identity": source_code_identity(), "runtime": runtime}
        # Dynamic disk/free-space measurements are diagnostics, not resume identity.
        from paper4_kbvqa.execution.fixed_candidate import durable_json, read_verified
        environment_path = output / "environment_initial.json"
        if environment_path.exists():
            read_verified(environment_path)
        else:
            durable_json(environment_path, collect_environment())
        pilot = FixedCandidatePilot(generator=generator, critic=critic, provider=provider,
                                    evidence_filter=RelevanceFilter(), execution_config=config,
                                    run_config=run_config, source_identity=source_identity)
        summary = pilot.run_manifest(samples, output_dir=output)
        print(json.dumps({k: v for k, v in summary.items() if k != "matched"}, indent=2))
        session_status = "COMPLETED_VALID" if summary["invalid"] == 0 else "COMPLETED_INVALID"
        pilot_status = summary["status"]
        return 0 if summary["invalid"] == 0 else 2
    except BaseException as error:
        session_error = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        session.finish(session_status, error=session_error, pilot_status=pilot_status)


if __name__ == "__main__":
    raise SystemExit(main())

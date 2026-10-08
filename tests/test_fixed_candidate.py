import json
from pathlib import Path

import pytest
from PIL import Image

from paper4_kbvqa.data.manifest import VQASample
from paper4_kbvqa.execution.controller import ExecutionConfig
from paper4_kbvqa.execution.fixed_candidate import (
    FixedCandidatePilot, read_verified,
)
from paper4_kbvqa.types import Evidence


class Generator:
    def __init__(self, result=None):
        self.answer_calls = 0
        self.grounding_calls = 0
        self.last_entity_record = None
        self.result = result or {"answer": "cutting", "parsed_ok": True,
            "parser_status": "valid", "parser_error": None,
            "raw_evidence_ids": ["ev1"], "valid_evidence_ids": ["ev1"],
            "raw_confidence": .8, "raw_text": '{"answer":"cutting","evidence_ids":["ev1"]}',
            "generated_token_logprobs": [-.2, -.3], "answer_token_confidence": None,
            "answer_token_confidence_status": "not_implemented_whole_response_only"}
        self.inputs = []

    def extract_visual_entities(self, image_path, question):
        self.grounding_calls += 1
        self.last_entity_record = {"entities": ["knife"], "raw_text": "knife", "parser_status": "valid"}
        return ["knife"]

    def generate(self, image_path, question, evidence):
        self.answer_calls += 1
        self.inputs.append({"image_path": image_path, "question": question,
                            "evidence": evidence})
        return self.result.copy()


class Provider:
    def __init__(self, evidence=None):
        self.calls = 0
        self.evidence = evidence if evidence is not None else [
            Evidence("ev1", "A knife is used for cutting.", "test-source", "https://example.test/knife", retrieval_score=.9)]

    def retrieve_for_question(self, question, entities, limit):
        self.calls += 1
        return list(self.evidence[:limit])


class Filter:
    def select(self, evidence, question, entities, top_k):
        return evidence[:top_k], {"ev1": {"score": .5}}


class Critic:
    def __init__(self, bad=False):
        self.calls = 0
        self.inputs = []
        self.bad = bad

    def verify(self, **kwargs):
        self.calls += 1
        self.inputs.append(kwargs)
        return {"supported": .8, "contradicted": .1,
                "insufficient": float("nan") if self.bad else .1,
                "label": "SUPPORTED", "model_id": "test-critic",
                "raw_label_log_scores": {"SUPPORTED": -.3, "CONTRADICTED": -2.4, "INSUFFICIENT": -2.6},
                "rendered_prompt": "archived critic prompt"}


@pytest.fixture
def setup(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (12, 8), "white").save(image)
    sample = VQASample("qid1", str(image), "What is the knife used for?",
                       answers=("SECRET_GOLD",), metadata={"source_split": "train", "image_id": 7})
    generator, critic, provider = Generator(), Critic(), Provider()
    return sample, generator, critic, provider, tmp_path / "run"


def runner(generator, critic, provider, config=None, run_config=None, source_identity=None):
    return FixedCandidatePilot(generator=generator, critic=critic, provider=provider,
        evidence_filter=Filter(), execution_config=config,
        run_config=run_config or {"model_revision": "pinned", "seed": 2026},
        source_identity=source_identity or {"mode": "live_capture_once", "provider_version": "v1"})


def test_same_candidate_context_single_generation_and_resume(setup):
    sample, generator, critic, provider, out = setup
    pilot = runner(generator, critic, provider)
    summary = pilot.run_manifest([sample], output_dir=out)
    assert summary["status"] == "COMPLETED_VALID_DEVELOPMENT_PILOT"
    assert generator.answer_calls == generator.grounding_calls == provider.calls == critic.calls == 1
    candidate = read_verified(next((out / "candidates").glob("*.json")))
    context = read_verified(next((out / "contexts").glob("*.json")))
    critique = read_verified(next((out / "critics").glob("*.json")))
    assert critique["B2"]["answer"] == critique["B3"]["answer"] == candidate["candidate"]["answer"]
    assert critique["B2"]["candidate_sha256"] == critique["B3"]["candidate_sha256"]
    assert critique["B2"]["context_sha256"] == critique["B3"]["context_sha256"] == context["context_sha256"]
    assert critic.inputs[0]["answer"] == generator.result["answer"]
    assert candidate["generator_result"]["generated_token_logprobs"] == [-.2, -.3]
    assert critique["critic_result"]["raw_label_log_scores"]["SUPPORTED"] == -.3
    assert context["visual_grounding_record"]["raw_text"] == "knife"
    assert "SECRET_GOLD" not in json.dumps([context, candidate, critique, generator.inputs], default=str)
    assert pilot.run_manifest([sample], output_dir=out) == summary
    assert generator.answer_calls == generator.grounding_calls == provider.calls == critic.calls == 1
    assert summary["benchmark_result"] is summary["calibration_implemented"] is False


def test_crash_after_candidate_archive_reuses_candidate_without_live_retrieval(setup, monkeypatch):
    from paper4_kbvqa.execution import fixed_candidate as module
    sample, generator, critic, provider, out = setup
    pilot = runner(generator, critic, provider)
    original = module.durable_json
    def fail_critic_write(path, record):
        if path.parent.name == "critics":
            raise OSError("simulated disconnect before critic archive")
        original(path, record)
    monkeypatch.setattr(module, "durable_json", fail_critic_write)
    with pytest.raises(OSError):
        pilot.run_manifest([sample], output_dir=out)
    assert generator.answer_calls == provider.calls == 1
    assert len(list((out / "candidates").glob("*.json"))) == 1
    monkeypatch.setattr(module, "durable_json", original)
    # Upstream source changes cannot change the already archived candidate context.
    provider.evidence = [Evidence("ev2", "Changed live source.", "test-source", "https://example.test/changed")]
    summary = pilot.run_manifest([sample], output_dir=out)
    assert summary["valid"] == 1
    assert generator.answer_calls == generator.grounding_calls == provider.calls == 1
    assert critic.inputs[-1]["evidence"][0].evidence_id == "ev1"


@pytest.mark.parametrize("target", ["contexts", "candidates", "critics"])
def test_tampered_archived_record_is_rejected(setup, target):
    sample, generator, critic, provider, out = setup
    pilot = runner(generator, critic, provider)
    pilot.run_manifest([sample], output_dir=out)
    path = next((out / target).glob("*.json"))
    obj = json.loads(path.read_text())
    obj["record"]["question_id"] = "tampered"
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="checksum"):
        pilot.run_manifest([sample], output_dir=out)
    assert generator.answer_calls == 1


@pytest.mark.parametrize("change", ["question", "image", "execution_config", "run_config", "source_config"])
def test_changed_input_or_configuration_rejects_resume(setup, change):
    sample, generator, critic, provider, out = setup
    runner(generator, critic, provider).run_manifest([sample], output_dir=out)
    config, run_config, source_identity = None, None, None
    if change == "question":
        sample = VQASample(sample.question_id, sample.image_path, "Changed question", metadata=sample.metadata)
    elif change == "image":
        Image.new("RGB", (12, 8), "red").save(sample.image_path)
    elif change == "execution_config":
        config = ExecutionConfig(generator_weight=.5)
    elif change == "run_config":
        run_config = {"model_revision": "different", "seed": 2026}
    else:
        source_identity = {"mode": "live_capture_once", "provider_version": "different"}
    with pytest.raises(ValueError, match="identity changed"):
        runner(generator, critic, provider, config, run_config, source_identity).run_manifest([sample], output_dir=out)
    assert generator.answer_calls == 1


@pytest.mark.parametrize("update", [
    {"parsed_ok": False, "parser_status": "invalid_json", "raw_text": "not JSON"},
    {"parsed_ok": False, "parser_status": "invalid_schema", "raw_text": '{"answer":42}'},
    {"answer": ""}, {"raw_confidence": float("nan")},
    {"raw_evidence_ids": [42]},
])
def test_invalid_generator_retained_completion_invalid_no_critic(setup, update):
    sample, generator, critic, provider, out = setup
    generator.result.update(update)
    summary = runner(generator, critic, provider).run_manifest([sample], output_dir=out)
    assert summary["invalid"] == 1 and summary["valid"] == 0
    assert summary["status"] == "COMPLETED_INVALID_DEVELOPMENT_PILOT"
    assert generator.answer_calls == 1 and critic.calls == 0
    record = read_verified(next((out / "candidates").glob("*.json")))
    assert "generator_result" in record and record["errors"]


def test_invalid_critic_retained_and_completion_invalid(setup):
    sample, generator, _, provider, out = setup
    critic = Critic(bad=True)
    summary = runner(generator, critic, provider).run_manifest([sample], output_dir=out)
    assert summary["invalid"] == 1
    rec = read_verified(next((out / "critics").glob("*.json")))
    assert rec["critic_result"]["insufficient"] == {"nonfinite_float": "nan"}
    assert rec["errors"] and rec.get("B3") is None


def test_gold_evidence_field_stops_model_calls_and_retains_source_error(setup):
    sample, generator, critic, provider, out = setup
    provider.evidence = [Evidence("ev1", "Safe text", "test-source", "https://example.test", metadata={"gold_answer": "SECRET"})]
    summary = runner(generator, critic, provider).run_manifest([sample], output_dir=out)
    assert summary["invalid"] == 1
    assert generator.answer_calls == critic.calls == 0
    context = read_verified(next((out / "contexts").glob("*.json")))
    assert "Forbidden gold/label" in context["errors"][0]["message"]


def test_duplicate_ids_and_nontrain_role_rejected_before_calls(setup):
    sample, generator, critic, provider, out = setup
    pilot = runner(generator, critic, provider)
    with pytest.raises(ValueError, match="Duplicate"):
        pilot.run_manifest([sample, sample], output_dir=out)
    other = VQASample("qid2", sample.image_path, sample.question, metadata={"source_split": "val"})
    with pytest.raises(ValueError, match="TRAIN"):
        pilot.run_manifest([other], output_dir=out)
    assert generator.answer_calls == critic.calls == provider.calls == 0


def test_provider_failure_stops_bounded_pilot_and_reports_unattempted(setup):
    sample, generator, critic, provider, out = setup
    def fail_source(*args, **kwargs):
        raise TimeoutError("source unavailable")
    provider.retrieve_for_question = fail_source
    second = VQASample("qid2", sample.image_path, sample.question, metadata=sample.metadata)
    summary = runner(generator, critic, provider).run_manifest([sample, second], output_dir=out)
    assert summary["attempted"] == 1 and summary["unattempted"] == 1
    assert summary["invalid"] == 1 and generator.answer_calls == critic.calls == 0
    rec = read_verified(next((out / "contexts").glob("*.json")))
    assert rec["errors"][0]["type"] == "TimeoutError"


def test_filter_cannot_invent_context_evidence(setup):
    sample, generator, critic, provider, out = setup
    class InventingFilter:
        def select(self, *args, **kwargs):
            return [Evidence("ev2", "Invented fact", "test-source", "https://example.test")], {}
    pilot = runner(generator, critic, provider)
    pilot.evidence_filter = InventingFilter()
    summary = pilot.run_manifest([sample], output_dir=out)
    assert summary["invalid"] == 1 and generator.answer_calls == critic.calls == 0


def test_cli_raw_manifest_needs_matching_explicit_train_attestation(tmp_path):
    import importlib.util
    from types import SimpleNamespace
    from paper4_kbvqa.execution.fixed_candidate import file_digest
    script = Path(__file__).resolve().parents[1] / "scripts/run_fixed_candidate_pilot.py"
    spec = importlib.util.spec_from_file_location("pilot_cli_test", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    manifest = tmp_path / "manifest.jsonl"
    row = {"question_id": "qid1", "question": "What is shown?", "image_path": "image.png"}
    manifest.write_text(json.dumps(row) + "\n")
    args = SimpleNamespace(pilot_dir=None, manifest=str(manifest), train_development_attestation=None, max_samples=50)
    with pytest.raises(ValueError, match="attestation"):
        cli.load_pilot_inputs(args)
    attestation = tmp_path / "attestation.json"
    attestation.write_text(json.dumps({"source_split": "train", "partition_role": "development_only", "manifest_sha256": file_digest(manifest)}))
    args.train_development_attestation = str(attestation)
    samples, identity = cli.load_pilot_inputs(args)
    assert samples[0].metadata["source_split"] == "train" and samples[0].answers == ()
    assert samples[0].image_path == str(tmp_path / "image.png")
    assert identity["upstream_split_authenticity_verified"] is False
    row["answers"] = ["SECRET_GOLD"]
    manifest.write_text(json.dumps(row) + "\n")
    attestation.write_text(json.dumps({"source_split": "train", "partition_role": "development_only", "manifest_sha256": file_digest(manifest)}))
    with pytest.raises(ValueError, match="Forbidden"):
        cli.load_pilot_inputs(args)


def test_malformed_visual_grounding_retains_raw_and_uses_empty_entity_fallback(setup):
    sample, generator, critic, provider, out = setup
    def malformed(image_path, question):
        generator.last_entity_record = {"entities": [], "raw_text": "bad JSON", "parser_status": "invalid_json"}
        return []
    generator.extract_visual_entities = malformed
    summary = runner(generator, critic, provider).run_manifest([sample], output_dir=out)
    assert summary["valid"] == 1 and generator.answer_calls == critic.calls == provider.calls == 1
    context = read_verified(next((out / "contexts").glob("*.json")))
    assert context["visual_grounding_record"]["raw_text"] == "bad JSON"
    assert context["grounding_fallback"] == "empty_entities_question_keywords_only"
    assert critic.inputs[0]["visual_entities"] == []


@pytest.mark.parametrize("ids,expected_valid,validity", [
    (["ev1", "UNKNOWN"], ["ev1"], .5),
    (["ev1", "ev1"], ["ev1", "ev1"], 1.),
    (["ev1", "UNKNOWN", "ev1"], ["ev1", "ev1"], .5),
    (["UNKNOWN"], [], 0.),
    ([], [], 0.),
])
def test_schema_valid_citation_diagnostics_do_not_drop_candidates(setup, ids, expected_valid, validity):
    sample, generator, critic, provider, out = setup
    generator.result["raw_evidence_ids"] = ids
    # Runner recomputes membership from the frozen context, not this stale field.
    generator.result["valid_evidence_ids"] = ["STAND_IN_STALE_VALID_ID"]
    summary = runner(generator, critic, provider).run_manifest([sample], output_dir=out)
    assert summary["valid"] == 1 and summary["invalid"] == 0
    assert summary["status"] == "COMPLETED_VALID_DEVELOPMENT_PILOT"
    assert generator.answer_calls == critic.calls == 1
    candidate = read_verified(next((out / "candidates").glob("*.json")))
    critique = read_verified(next((out / "critics").glob("*.json")))
    assert candidate["candidate"]["raw_evidence_ids"] == ids
    assert candidate["candidate"]["valid_evidence_ids"] == expected_valid
    assert critic.inputs[0]["cited_evidence_ids"] == expected_valid
    assert candidate["errors"] == []
    assert candidate["citation_diagnostics"]["distinct_raw_identifier_count"] == len(set(ids))
    assert candidate["citation_diagnostics"]["distinct_valid_identifier_count"] == len(set(expected_valid))
    assert critique["provenance"]["citation_validity"] == pytest.approx(validity)
    assert critique["B2"]["candidate_sha256"] == critique["B3"]["candidate_sha256"]
    assert critique["B2"]["answer"] == critique["B3"]["answer"]

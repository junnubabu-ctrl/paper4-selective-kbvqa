"""CPU/software contract checks; these do not establish Qwen GPU execution."""
from contextlib import nullcontext
from dataclasses import dataclass
import importlib.util
from importlib import metadata
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from paper4_kbvqa.generation import json_constraints as constraints
from paper4_kbvqa.generation import qwen_vl
from paper4_kbvqa.generation.qwen_vl import Qwen25VLGenerator
from paper4_kbvqa.types import Evidence


def test_schemas_are_exact_and_preserve_unknown_citation_diagnostics():
    entities = constraints.generation_schema("entities")
    assert entities["required"] == ["entities"]
    assert entities["additionalProperties"] is False
    assert entities["properties"]["entities"]["items"] == {"type": "string", "minLength": 1}
    answer = constraints.generation_schema("answer_with_evidence_ids")
    assert set(answer["properties"]) == {"answer", "evidence_ids"}
    assert answer["required"] == ["answer", "evidence_ids"]
    assert answer["additionalProperties"] is False
    assert answer["properties"]["evidence_ids"]["items"] == {"type": "string"}
    assert "enum" not in json.dumps(answer)
    parsed = Qwen25VLGenerator._parse_details('{"answer":"cat","evidence_ids":["unknown",""]}', ["e1"])
    assert parsed["parsed_ok"] and parsed["invalid_evidence_ids"] == ["unknown", ""]
    assert constraints.generation_schema("answer_only")["required"] == ["answer"]
    answer["properties"]["answer"]["minLength"] = 99
    assert constraints.generation_schema("answer_with_evidence_ids")["properties"]["answer"]["minLength"] == 1
    with pytest.raises(ValueError, match="Unknown generation"):
        constraints.generation_schema("anything")


def test_dependency_absent_is_an_explicit_opt_in_error(monkeypatch):
    def absent(name):
        raise metadata.PackageNotFoundError(name)
    monkeypatch.setattr(constraints.metadata, "version", absent)
    with pytest.raises(RuntimeError, match=r"lm-format-enforcer==0\.11\.3"):
        Qwen25VLGenerator(constrained_json=True)._load()
    assert Qwen25VLGenerator().constrained_json is False


def test_unsupported_dependency_version_fails_before_loading_vlm(monkeypatch):
    monkeypatch.setattr(constraints.metadata, "version", lambda name: "0.10.0")
    with pytest.raises(RuntimeError, match="found 0.10.0"):
        constraints.require_json_constraints()


class Tokenizer:
    name_or_path = "TEST_ONLY_LOCAL_TOKENIZER"
    is_fast = True
    eos_token_id = 0
    pad_token_id = 0
    all_special_ids = [0]
    special_tokens_map = {"eos_token": "<eos>"}
    init_kwargs = {"revision": "test-revision", "_commit_hash": "test-hash"}
    clean_up_tokenization_spaces = True
    backend_tokenizer = SimpleNamespace(to_str=lambda: '{"TEST_ONLY_BACKEND":true}')

    def get_vocab(self):
        return {"<eos>": 0, "{": 1, "}": 2}

    def get_added_vocab(self):
        return {"<eos>": 0}


def test_builder_uses_tokenizer_and_fresh_parser_and_records_effective_config(monkeypatch):
    @dataclass
    class Config:
        alphabet: str = "TOKENIZER_ALPHABET"
        max_consecutive_whitespaces: int = 12
        force_json_field_order: bool = False
        max_json_array_length: int = 20
    built = []
    class Parser:
        def __init__(self, schema):
            self.schema = schema
            self.config = Config(alphabet="BEFORE_BUILDER")
    def builder(tokenizer, parser):
        parser.config = Config()
        fn = SimpleNamespace(token_enforcer=SimpleNamespace(root_parser=parser))
        built.append((tokenizer, parser, fn))
        return fn
    data_built = []
    def data_builder(tokenizer):
        data = SimpleNamespace(source_tokenizer=tokenizer)
        data_built.append(data)
        return data
    monkeypatch.setattr(constraints, "require_json_constraints", lambda: (Parser, builder, "0.11.3", data_builder))
    tokenizer = Tokenizer()
    first, identity = constraints.build_json_constraint(tokenizer, "answer_only")
    second, again = constraints.build_json_constraint(tokenizer, "answer_only")
    assert built[0][0] is tokenizer and built[1][0] is tokenizer
    assert first is not second and built[0][1] is not built[1][1]
    assert tokenizer.clean_up_tokenization_spaces is False
    assert identity == again
    assert identity["schema_version"] == constraints.CONSTRAINT_SCHEMA_VERSION
    assert identity["schema"] == constraints.generation_schema("answer_only")
    assert identity["library"]["version"] == "0.11.3"
    assert identity["parser_config"]["alphabet"] == "TOKENIZER_ALPHABET"
    assert identity["parser_config"]["max_json_array_length"] == 20
    assert identity["tokenizer"]["clean_up_tokenization_spaces"] is False
    assert len(identity["tokenizer"]["vocab_sha256"]) == 64
    assert len(identity["tokenizer"]["backend_tokenizer_sha256"]) == 64
    tokenizer.get_vocab = lambda: {"<eos>": 0, "different": 1}
    _, changed = constraints.build_json_constraint(tokenizer, "answer_only")
    assert changed["tokenizer"]["vocab_sha256"] != identity["tokenizer"]["vocab_sha256"]
    with pytest.raises(RuntimeError, match="processor.tokenizer"):
        constraints.build_json_constraint(None, "answer_only")
    cache = {}
    first_cached, _ = constraints.build_json_constraint(tokenizer, "answer_only", tokenizer_data_cache=cache)
    second_cached, _ = constraints.build_json_constraint(tokenizer, "entities", tokenizer_data_cache=cache)
    assert first_cached is not second_cached
    assert len(data_built) == 1
    assert built[-1][0] is built[-2][0] is data_built[0]
    assert built[-1][1] is not built[-2][1]
    tokenizer.get_vocab = lambda: {"<eos>": 0, "changed_again": 1}
    constraints.build_json_constraint(tokenizer, "answer_only", tokenizer_data_cache=cache)
    assert len(data_built) == 2
    # A distinct tokenizer with identical metadata must not share a live decoder.
    identical = Tokenizer()
    identical.get_vocab = tokenizer.get_vocab
    constraints.build_json_constraint(identical, "answer_only", tokenizer_data_cache=cache)
    assert len(data_built) == 3
    identical.backend_tokenizer = None
    constraints.build_json_constraint(identical, "answer_only", tokenizer_data_cache=cache)
    assert built[-1][0] is identical  # unsupported content fingerprints are never cached
    assert len(data_built) == 3


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    class Inputs(dict):
        def __getattr__(self, key):
            return self[key]
        def to(self, device):
            return self
    class Scores:
        def __init__(self, values):
            self.values = values
        def __getitem__(self, key):
            return self
        def float(self):
            return np.array(self.values, dtype=float)
    class Processor:
        image_processor = SimpleNamespace(patch_size=14, to_dict=lambda: {"patch_size": 14})
        tokenizer = Tokenizer()
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return json.dumps(messages, default=lambda value: "TEST_ONLY_IMAGE", sort_keys=True)
        def __call__(self, **kwargs):
            return Inputs(input_ids=np.array([[1, 2]]), image_grid_thw=np.array([[1, 2, 3]]))
        def decode(self, tokens, **kwargs):
            assert kwargs == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
            return self.raw
    class Model:
        device = "TEST_ONLY_CPU_STAND_IN"
        config = SimpleNamespace(_commit_hash="a" * 40, vocab_size=4, to_dict=lambda: {"TEST_ONLY_CONFIG": True})
        generation_config = SimpleNamespace(eos_token_id=0, to_dict=lambda: {"repetition_penalty": 1.05, "eos_token_id": 0})
        omit_logits = False
        shorten_logits = False
        generation_tokens = [2, 3]
        def generate(self, **kwargs):
            self.kwargs = kwargs
            seqs = np.array([[1, 2] + self.generation_tokens])
            if not kwargs.get("return_dict_in_generate"):
                return seqs
            # A grammar would renormalize these selected-token scores to 1.0.
            masked = [Scores([0. if index == token else -math.inf for index in range(4)]) for token in self.generation_tokens]
            raw = [Scores([0., 1., 2., 3.]) for token in self.generation_tokens]
            logits = None if self.omit_logits else raw[:1] if self.shorten_logits else raw
            return SimpleNamespace(sequences=seqs, scores=masked if kwargs.get("output_logits") else raw,
                                   logits=logits)
    def log_softmax(values, dim):
        return values - np.log(np.exp(values).sum())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext,
        initial_seed=lambda: 2026, log_softmax=log_softmax))
    image_path = tmp_path / "test_only.png"
    Image.new("RGB", (42, 28)).save(image_path)
    calls = []
    def build(tokenizer, kind, *, tokenizer_data_cache=None):
        callback = lambda batch_id, prefix: [2, 3]
        identity = {"schema_name": kind, "schema": constraints.generation_schema(kind),
                    "identity_version": constraints.CONSTRAINT_IDENTITY_VERSION}
        calls.append((tokenizer, kind, callback))
        return callback, identity
    monkeypatch.setattr(qwen_vl, "build_json_constraint", build)
    def make(raw, constrained_json=False, include_evidence_ids=True):
        generator = Qwen25VLGenerator(revision="a" * 40, constrained_json=constrained_json,
                                     include_evidence_ids=include_evidence_ids)
        generator._loaded = True
        generator.processor, generator.model = Processor(), Model()
        generator.processor.raw = raw
        generator.loading_identity = {"TEST_ONLY_STAND_IN": True}
        return generator
    return SimpleNamespace(make=make, image=str(image_path), evidence=[Evidence("e1", "Test evidence", "fixture")],
                           constraint_calls=calls)


def test_constrained_confidence_uses_raw_logits_not_grammar_masked_scores(adapter):
    generator = adapter.make('{"answer":"cat","evidence_ids":["unknown","e1"]}', True)
    result = generator.generate(adapter.image, "Test?", adapter.evidence)
    expected = np.array([2., 3.]) - np.log(np.exp([0., 1., 2., 3.]).sum())
    assert result["generated_token_logprobs"] == pytest.approx(expected)
    assert result["raw_confidence"] == pytest.approx(math.exp(float(expected.mean())))
    assert result["raw_confidence"] != 1.0  # grammar-masked score confidence
    assert result["invalid_evidence_ids"] == ["unknown"]
    assert result["generated_token_ids"] == [2, 3]
    assert result["raw_text"] == generator.processor.raw == result["raw_output"]
    assert result["likelihood_score_source"] == constraints.LIKELIHOOD_SOURCE
    decoding = result["generation_identity"]["decoding"]
    assert decoding["json_constraint"]["schema_name"] == "answer_with_evidence_ids"
    assert decoding["likelihood_score_source"] == constraints.LIKELIHOOD_SOURCE
    assert result["generation_identity"]["generation_config"]["repetition_penalty"] == 1.05
    assert generator.model.kwargs["output_scores"] is False
    assert generator.model.kwargs["output_logits"] is True
    assert generator.model.kwargs["return_dict_in_generate"] is True
    assert generator.model.kwargs["prefix_allowed_tokens_fn"] is adapter.constraint_calls[0][2]
    assert adapter.constraint_calls[0][0] is generator.processor.tokenizer


@pytest.mark.parametrize("raw,status", [
    ('{"answer":"cat","evidence_ids":["e1"', "invalid_json"),
    ('```json\n{"answer":"cat","evidence_ids":[]}\n```', "invalid_json"),
    ('{"answer":" ","evidence_ids":[]}', "invalid_schema"),
    ('{"answer":"cat","evidence_ids":[],"other":0}', "invalid_schema"),
    ('{"answer":"cat","answer":"dog","evidence_ids":[]}', "invalid_json"),
])
def test_constrained_mode_never_repairs_invalid_answer_outputs(adapter, raw, status):
    generator = adapter.make(raw, True)
    result = generator.generate(adapter.image, "Test?", adapter.evidence)
    assert result["raw_text"] == result["raw_output"] == raw
    assert result["generated_token_ids"] == [2, 3]
    assert result["parser_status"] == status and result["parsed_ok"] is False
    assert result["answer"] == "" and result["raw_evidence_ids"] == []


@pytest.mark.parametrize("raw,status", [
    ('{"entities":["cat"]}', "valid"),
    ('{"entities":["cat"', "invalid_json"),
    ('```json\n{"entities":[]}\n```', "invalid_json"),
    ('{"entities":[" "]}', "invalid_schema"),
])
def test_constrained_entity_generation_archives_raw_tokens_and_strict_status(adapter, raw, status):
    generator = adapter.make(raw, True)
    entities = generator.extract_visual_entities(adapter.image, "Test?")
    record = generator.last_entity_record
    assert record["parser_status"] == status
    assert entities == (["cat"] if status == "valid" else [])
    assert record["raw_text"] == record["raw_output"] == raw
    assert record["generated_token_ids"] == [2, 3]
    assert len(record["generated_token_logprobs"]) == 2
    assert record["raw_confidence"] < 1.0
    assert record["decoding"]["json_constraint"]["schema_name"] == "entities"
    assert record["likelihood_score_source"] == constraints.LIKELIHOOD_SOURCE
    assert generator.model.kwargs["output_scores"] is False


@pytest.mark.parametrize("entities", [False, True])
@pytest.mark.parametrize("problem", ["omit_logits", "shorten_logits"])
def test_constrained_mode_fails_without_all_unprocessed_logits(adapter, problem, entities):
    generator = adapter.make('{"answer":"cat","evidence_ids":[]}', True)
    setattr(generator.model, problem, True)
    with pytest.raises(RuntimeError, match="unprocessed generation logits"):
        if entities:
            generator.extract_visual_entities(adapter.image, "Test?")
        else:
            generator.generate(adapter.image, "Test?", adapter.evidence)


def test_opt_in_changes_decoding_identity_but_preserves_prompts_and_defaults(adapter, monkeypatch):
    raw = '{"answer":"cat","evidence_ids":[]}'
    default = adapter.make(raw)
    explicit_default = adapter.make(raw, False)
    constrained = adapter.make(raw, True)
    one = default.generate(adapter.image, "Test?", adapter.evidence)
    two = explicit_default.generate(adapter.image, "Test?", adapter.evidence)
    three = constrained.generate(adapter.image, "Test?", adapter.evidence)
    assert one == two
    assert one["prediction_key"] != three["prediction_key"]
    assert one["generation_identity"]["rendered_prompt"] == three["generation_identity"]["rendered_prompt"]
    assert one["generation_identity"]["prompt_version"] == three["generation_identity"]["prompt_version"]
    assert one["generation_identity"]["decoding"] == {"max_new_tokens": 192, "do_sample": False}
    assert "likelihood_score_source" not in one
    assert "prefix_allowed_tokens_fn" not in default.model.kwargs
    assert "output_logits" not in default.model.kwargs
    assert default.model.kwargs["output_scores"] is True
    def forbidden(*args, **kwargs):
        pytest.fail("The default path must not initialize JSON constraints")
    monkeypatch.setattr(qwen_vl, "build_json_constraint", forbidden)
    default.generate(adapter.image, "Test?", adapter.evidence)
    default.processor.raw = '{"entities":["cat"]}'
    assert default.extract_visual_entities(adapter.image, "Test?") == ["cat"]
    assert default.last_entity_record["decoding"] == {"max_new_tokens": 64, "do_sample": False}
    assert "return_dict_in_generate" not in default.model.kwargs


def test_each_request_gets_fresh_constraint_state_and_answer_only_schema(adapter):
    generator = adapter.make('{"answer":"cat"}', True, False)
    generator.generate(adapter.image, "Test?", [])
    generator.generate(adapter.image, "Again?", [])
    generator.processor.raw = '{"entities":[]}'
    generator.extract_visual_entities(adapter.image, "Test?")
    assert [call[1] for call in adapter.constraint_calls] == ["answer_only", "answer_only", "entities"]
    assert len({id(call[2]) for call in adapter.constraint_calls}) == 3


def test_tokenizer_model_compatibility_accepts_padded_logits_width_and_checks_eos_and_ids():
    tokenizer = Tokenizer()
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=3),
        generation_config=SimpleNamespace(eos_token_id=[0, 6]),
        get_output_embeddings=lambda: SimpleNamespace(weight=np.zeros((8, 1))))
    tokenizer.get_vocab = lambda: {"<eos>": 0, "highest_known_token": 7}
    identity = qwen_vl._constraint_tokenizer_compatibility(model, tokenizer)
    assert identity["model_output_vocab_size"] == 8
    assert identity["tokenizer_max_token_id"] == 7
    assert identity["constraint_eos_token_ids"] == [0]
    tokenizer.get_vocab = lambda: {"<eos>": 0, "out_of_range": 8}
    with pytest.raises(RuntimeError, match="fit the model output vocabulary"):
        qwen_vl._constraint_tokenizer_compatibility(model, tokenizer)
    model.generation_config.eos_token_id = [6]
    with pytest.raises(RuntimeError, match="EOS must be accepted"):
        qwen_vl._constraint_tokenizer_compatibility(model, tokenizer)


@pytest.mark.parametrize("entity_mode,budget", [(False, 192), (True, 64)])
@pytest.mark.parametrize("terminal_eos", [False, True])
def test_length_limit_requires_terminal_eos_even_for_complete_json(adapter, entity_mode, budget, terminal_eos):
    raw = '{"entities":["cat"]}' if entity_mode else '{"answer":"cat","evidence_ids":["e1","unknown"]}'
    generator = adapter.make(raw, True)
    generator.model.generation_tokens = [2] * (budget - 1) + [0 if terminal_eos else 2]
    if entity_mode:
        entities = generator.extract_visual_entities(adapter.image, "Test?")
        result = generator.last_entity_record
        assert entities == (["cat"] if terminal_eos else [])
    else:
        result = generator.generate(adapter.image, "Test?", adapter.evidence)
        assert result["parsed_ok"] is terminal_eos
        assert result["answer"] == ("cat" if terminal_eos else "")
        assert result["raw_evidence_ids"] == ["e1", "unknown"]
        assert result["valid_evidence_ids"] == ["e1"]
        assert result["invalid_evidence_ids"] == ["unknown"]
        assert result["supporting_evidence_ids"] == (["e1", "unknown"] if terminal_eos else [])
    assert result["strict_parser_status"] == "valid"
    assert result["parser_status"] == ("valid" if terminal_eos else "length_limit_without_eos")
    assert result["generation_termination"]["status"] == ("eos" if terminal_eos else "length_limit_without_eos")
    assert result["raw_text"] == raw == result["raw_output"]
    assert len(result["generated_token_ids"]) == len(result["generated_token_logprobs"]) == budget


@pytest.mark.skipif(importlib.util.find_spec("lmformatenforcer") is None, reason="optional LMFE dependency absent")
@pytest.mark.parametrize("kind,raw,accepted", [
    ("entities", '{"entities":[]}', True),
    ("entities", '{"entities":["cat","dog"]}', True),
    ("entities", '{"entities":[""]}', False),
    ("entities", '{"entities":[1]}', False),
    ("answer_only", '{"answer":"cat"}', True),
    ("answer_only", '{"answer":"cat","other":"dog"}', False),
    ("answer_only", '{"answer":"cat","answer":"dog"}', False),
    ("answer_only", '{"answer":""}', False),
    ("answer_with_evidence_ids", '{"answer":"cat","evidence_ids":["unknown",""]}', True),
    ("answer_with_evidence_ids", '{"evidence_ids":[],"answer":"cat"}', True),
    ("answer_with_evidence_ids", '{"answer":"cat","evidence_ids":[1]}', False),
    ("answer_with_evidence_ids", '{"answer":"cat"}', False),
])
def test_real_lmfe_token_enforcer_with_local_character_tokenizer(kind, raw, accepted):
    """Real LMFE state transitions; no downloads, model, GPU, or HF stub needed."""
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer, TokenEnforcerTokenizerData
    assert metadata.version("lm-format-enforcer") == constraints.LMFE_VERSION
    chars = [chr(value) for value in range(32, 127)] + ["\n", "\r", "\t"]
    pieces = {index + 1: char for index, char in enumerate(chars)}
    encode = {char: token for token, char in pieces.items()}
    data = TokenEnforcerTokenizerData([(token, char, False) for token, char in pieces.items()],
        lambda tokens: "".join(pieces.get(token, "") for token in tokens), 0, False, len(pieces) + 1)
    enforcer = TokenEnforcer(data, JsonSchemaParser(constraints.generation_schema(kind)))
    prefix = [0]  # first call marks prompt boundary; subsequent tokens are response
    valid = True
    for char in raw:
        token = encode[char]
        if token not in enforcer.get_allowed_tokens(prefix).allowed_tokens:
            valid = False
            break
        prefix.append(token)
    if valid:
        valid = 0 in enforcer.get_allowed_tokens(prefix).allowed_tokens
    assert valid is accepted


@pytest.mark.skipif(not all(importlib.util.find_spec(name) is not None for name in
    ["lmformatenforcer", "transformers", "tokenizers", "torch"]), reason="optional HF/Torch integration dependencies absent")
def test_real_hf_prefix_callback_with_locally_constructed_tokenizer():
    """Exercise the production builder too when VLM extras are installed."""
    import torch
    from tokenizers import Tokenizer as BackendTokenizer
    from tokenizers.decoders import Fuse
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast
    chars = [chr(value) for value in range(32, 127)] + ["\n", "\r", "\t"]
    vocab = {"<eos>": 0, "<unk>": 1, **{char: index + 2 for index, char in enumerate(chars)}}
    backend = BackendTokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.decoder = Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>",
                                       pad_token="<eos>", unk_token="<unk>")
    callback, identity = constraints.build_json_constraint(tokenizer, "answer_with_evidence_ids")
    other, other_identity = constraints.build_json_constraint(tokenizer, "answer_with_evidence_ids")
    assert callback.token_enforcer is not other.token_enforcer
    assert identity == other_identity
    raw = '{"answer":"cat","evidence_ids":["unknown"]}'
    tokens = tokenizer.encode(raw, add_special_tokens=False)
    assert tokenizer.decode(tokens) == raw
    prefix = [tokenizer.eos_token_id]
    for token in tokens:
        assert token in callback(0, torch.tensor(prefix))
        prefix.append(token)
    assert tokenizer.eos_token_id in callback(0, torch.tensor(prefix))
    assert identity["tokenizer"]["clean_up_tokenization_spaces"] is False
    assert identity["parser_config"]["max_json_array_length"] == 20

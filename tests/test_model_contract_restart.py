"""Software contracts only; loader/score stand-ins are not GPU evidence."""
import math
import sys
from types import SimpleNamespace
import pytest
from paper4_kbvqa.generation.qwen_vl import Qwen25VLGenerator, _quantization_identity
from paper4_kbvqa.verification.llm_critic import QwenLabelLikelihoodCritic, _normalise_log_scores

@pytest.mark.parametrize('raw,status', [
    ('cat', 'invalid_json'),
    ('```json\n{"answer":"cat","evidence_ids":[]}\n```', 'invalid_json'),
    ('{"answer":"cat","answer":"dog","evidence_ids":[]}', 'invalid_json'),
    ('{"answer":"cat","evidence_ids":[],"score":NaN}', 'invalid_json'),
    ('{"answer":42,"evidence_ids":[]}', 'invalid_schema'),
    ('{"answer":" ","evidence_ids":[]}', 'invalid_schema'),
    ('{"answer":"cat","evidence_ids":[42]}', 'invalid_schema'),
    ('{"answer":"cat"}', 'invalid_schema'),
    ('{"answer":"cat","evidence_ids":[],"other":0}', 'invalid_schema'),
    ('[]', 'invalid_schema'),
])
def test_strict_parser_never_promotes_failed_output(raw, status):
    record = Qwen25VLGenerator._parse_details(raw, ['e1'])
    assert record['parser_status'] == status
    assert not record['parsed_ok'] and record['answer'] == ''
    assert record['raw_evidence_ids'] == [] and record['parser_error']


def test_raw_unknown_repeated_ids_survive_and_valid_subset_is_separate():
    raw = ' \n{"answer":"cat", "evidence_ids":["e1","unknown","e1"]}\n '
    record = Qwen25VLGenerator._parse_details(raw, ['e1'])
    assert record['parsed_ok']
    assert record['raw_evidence_ids'] == ['e1', 'unknown', 'e1']
    assert record['valid_evidence_ids'] == ['e1', 'e1']
    assert record['invalid_evidence_ids'] == ['unknown']
    assert Qwen25VLGenerator._parse_output(raw, ['e1']) == ('cat', ['e1', 'unknown', 'e1'])


def test_answer_only_schema_is_distinct():
    assert Qwen25VLGenerator._parse_details('{"answer":"cat"}', [], False)['parsed_ok']
    record = Qwen25VLGenerator._parse_details('{"answer":"cat","evidence_ids":[]}', [], False)
    assert record['parser_status'] == 'invalid_schema' and record['answer'] == ''


@pytest.mark.parametrize('raw', ['cat', '{"entities":[1]}', '{"entities":["cat"],"answer":"cat"}',
    '{"entities":["cat"],"entities":["dog"]}', '{"entities":[""]}'])
def test_entity_schema_fails_empty(raw):
    entities, status, error = Qwen25VLGenerator._parse_entities(raw)
    assert entities == [] and status != 'valid' and error


def test_entities_clean_only_after_schema_validation():
    assert Qwen25VLGenerator._parse_entities('{"entities":["cat","CAT","dog"]}', 1) == (['cat'], 'valid', None)


@pytest.mark.parametrize('values', [[math.nan, -1, -2], [-math.inf]*3, [0, math.inf, -1]])
def test_nonfinite_critic_scores_fail_instead_of_fabricating_insufficiency(values):
    with pytest.raises(ValueError, match='finite'):
        _normalise_log_scores(values)


def test_quantization_identity_checks_actual_nf4():
    stub = SimpleNamespace(is_loaded_in_4bit=False, config=SimpleNamespace(quantization_config={}))
    with pytest.raises(RuntimeError, match='not confirmed'):
        _quantization_identity(stub, True)
    stub.is_loaded_in_4bit = True
    stub.config.quantization_config = {'bnb_4bit_quant_type': 'fp4'}
    with pytest.raises(RuntimeError, match='not confirmed'):
        _quantization_identity(stub, True)
    stub.config.quantization_config['bnb_4bit_quant_type'] = 'nf4'
    assert _quantization_identity(stub, True)['quantization_mode'] == '4bit_nf4'


def _fake_loader_modules(monkeypatch, *, cuda=True, fail_quant=False, actual_4bit=True,
                         quant_type='nf4', resolved='a'*40, device='cuda:0'):
    calls = []
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda,
        is_bf16_supported=lambda: False, get_device_name=lambda: 'TEST_ONLY_FAKE_CUDA'), float16='float16')
    class Model:
        is_loaded_in_4bit = actual_4bit
        config = SimpleNamespace(_commit_hash=resolved, quantization_config={'bnb_4bit_quant_type': quant_type})
        hf_device_map = {'': device}
        def parameters(self):
            return iter([SimpleNamespace(device=device)])
        def eval(self):
            calls.append('eval')
    class Tokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append('tokenizer')
            return cls()
    class Loader:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(('model', kwargs))
            return Model()
    class Bits:
        def __init__(self, **kwargs):
            calls.append(('quant', kwargs))
            if fail_quant:
                raise ValueError('TEST_ONLY_QUANTIZATION_FAILURE')
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=Tokenizer,
        AutoModelForCausalLM=Loader, BitsAndBytesConfig=Bits))
    return calls


def test_critic_cuda_blocker_precedes_model_loading(monkeypatch):
    calls = _fake_loader_modules(monkeypatch, cuda=False)
    critic = QwenLabelLikelihoodCritic(revision='a'*40)
    with pytest.raises(RuntimeError, match='CUDA GPU'):
        critic._load()
    assert calls == [] and critic._model is None


def test_requested_quantization_has_no_silent_fallback(monkeypatch):
    calls = _fake_loader_modules(monkeypatch, fail_quant=True)
    critic = QwenLabelLikelihoodCritic(revision='a'*40)
    with pytest.raises(RuntimeError, match='4-bit mode requested'):
        critic._load()
    assert not any(isinstance(x, tuple) and x[0] == 'model' for x in calls)
    assert critic._model is None


@pytest.mark.parametrize('actual,quant_type,device', [(False, 'nf4', 'cuda:0'),
    (True, 'fp4', 'cuda:0'), (True, 'nf4', 'cpu')])
def test_loaded_critic_must_match_actual_quantization_and_cuda(monkeypatch, actual, quant_type, device):
    _fake_loader_modules(monkeypatch, actual_4bit=actual, quant_type=quant_type, device=device)
    critic = QwenLabelLikelihoodCritic(revision='a'*40)
    with pytest.raises(RuntimeError):
        critic._load()
    assert critic._model is None


def test_loaded_revision_and_mode_are_archived(monkeypatch):
    _fake_loader_modules(monkeypatch)
    critic = QwenLabelLikelihoodCritic(revision='a'*40)
    critic._load()
    assert critic.loading_identity['resolved_revision'] == 'a'*40
    assert critic.loading_identity['quantization_mode'] == '4bit_nf4'
    assert critic.loading_identity['cuda_device'] == 'TEST_ONLY_FAKE_CUDA'


def test_checkpoint_revision_mismatch_fails_before_loaded_state(monkeypatch):
    _fake_loader_modules(monkeypatch, resolved='b'*40)
    critic = QwenLabelLikelihoodCritic(revision='a'*40)
    with pytest.raises(RuntimeError, match='pinned revision'):
        critic._load()
    assert critic._model is None


def test_critic_result_archives_raw_scores_and_common_prompt(monkeypatch):
    critic = QwenLabelLikelihoodCritic(revision='a'*40)
    monkeypatch.setattr(critic, '_chat_prefix', lambda prompt: 'TEST_ONLY_PREFIX\n' + prompt)
    scores = {'SUPPORTED': -.1, 'CONTRADICTED': -2., 'INSUFFICIENT': -1.}
    monkeypatch.setattr(critic, '_continuation_logprob', lambda prefix, label: scores[label])
    result = critic.verify(question='Test question', answer='Test answer', visual_entities=[],
        evidence=[], cited_evidence_ids=[]).as_dict()
    assert result['raw_label_log_scores'] == scores
    assert result['rendered_prompt'].startswith('TEST_ONLY_PREFIX')
    assert result['requested_revision'] == 'a'*40
    assert 'critic_request_sha256' in result['loading_identity']
    assert result['resolved_revision'] is None  # stand-in is not checkpoint evidence

@pytest.mark.parametrize('raw,valid,include_evidence_ids', [
    (' \n{"answer":"cat","evidence_ids":["e1","missing","e1"]}\n ', True, True),
    (' cat \n', False, True),
    ('{"answer":"cat"}', True, False),
    ('cat', False, False),
])
def test_generation_archives_exact_raw_tokens_grid_and_parser_status(monkeypatch, tmp_path, raw, valid, include_evidence_ids):
    from contextlib import nullcontext
    import numpy as np
    from PIL import Image
    from paper4_kbvqa.types import Evidence
    class Inputs(dict):
        def __getattr__(self, key):
            return self[key]
        def to(self, device):
            return self
    class Scores:
        def __getitem__(self, index):
            return self
        def float(self):
            return np.array([0., 1., 2., 3.])
    class Processor:
        image_processor = SimpleNamespace(patch_size=14, to_dict=lambda: {'patch_size': 14})
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return 'TEST_ONLY_RENDERED_PROMPT ' + messages[-1]['content'][1]['text']
        def __call__(self, **kwargs):
            return Inputs(input_ids=np.array([[1, 2]]), image_grid_thw=np.array([[1, 2, 3]]))
        def decode(self, tokens, **kwargs):
            return raw
    class Model:
        device = 'TEST_ONLY_DEVICE'
        config = SimpleNamespace(_commit_hash='a'*40, to_dict=lambda: {'TEST_ONLY_CONFIG': True})
        def generate(self, **kwargs):
            return SimpleNamespace(sequences=np.array([[1, 2, 2, 3]]), scores=[Scores(), Scores()])
    def log_softmax(values, dim):
        return values - np.log(np.exp(values).sum())
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=nullcontext,
        initial_seed=lambda: 2026, log_softmax=log_softmax))
    image_path = tmp_path / 'test_only.png'
    Image.new('RGB', (42, 28)).save(image_path)
    generator = Qwen25VLGenerator(revision='a'*40, include_evidence_ids=include_evidence_ids)
    generator._loaded = True
    generator.model, generator.processor = Model(), Processor()
    generator.loading_identity = {'TEST_ONLY_STAND_IN': True}
    result = generator.generate(str(image_path), 'Test only?', [Evidence('e1', 'Test evidence', 'fixture')])
    assert result['raw_text'] == raw == result['raw_output']
    assert result['generated_token_ids'] == [2, 3]
    assert len(result['generated_token_logprobs']) == 2
    assert result['parsed_ok'] is valid
    assert result['generation_identity']['image_grid_thw'] == [[1, 2, 3]]
    assert result['generation_identity']['processed_image_dimensions'] == [[42, 28]]
    assert result['generation_identity']['image_dimensions_passed_to_processor'] == [42, 28]
    assert [m['role'] for m in generator.processor.messages] == ['system', 'user']
    system = generator.processor.messages[0]['content']
    user_text = generator.processor.messages[-1]['content'][1]['text']
    if include_evidence_ids:
        assert 'exactly the keys answer and evidence_ids' in system
        assert user_text.endswith('[e1] Test evidence')
        assert result['generation_identity']['prompt_version'] == 'evitrust-answer-evidence-json-v2'
    else:
        assert 'exactly the key answer.' in system and 'evidence_ids' not in system
        assert user_text.endswith('Test evidence') and '[e1]' not in user_text
        assert result['generation_identity']['prompt_version'] == 'evitrust-answer-only-json-v2'
    assert result['generation_identity']['decoding'] == {'max_new_tokens': 48, 'do_sample': False}
    assert result['answer_token_confidence'] is None
    if valid and include_evidence_ids:
        assert result['supporting_evidence_ids'] == ['e1', 'missing', 'e1']
        assert result['valid_evidence_ids'] == ['e1', 'e1']
    elif valid:
        assert result['answer'] == 'cat' and result['supporting_evidence_ids'] == []
    else:
        assert result['answer'] == '' and result['supporting_evidence_ids'] == []


@pytest.mark.parametrize('raw,valid', [
    ('{"entities":["surfboard"]}', True),
    ('The man has finished surfing and is walking back to the beach.', False),
])
def test_entity_prompt_contract_preserves_schema_failure_and_raw_model_output(monkeypatch, tmp_path, raw, valid):
    from contextlib import nullcontext
    import numpy as np
    from PIL import Image
    class Inputs(dict):
        def __getattr__(self, key):
            return self[key]
        def to(self, device):
            return self
    class Processor:
        image_processor = SimpleNamespace(patch_size=14)
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return 'TEST_ONLY_ENTITY_PROMPT ' + messages[0]['content'] + messages[-1]['content'][1]['text']
        def __call__(self, **kwargs):
            return Inputs(input_ids=np.array([[1, 2]]), image_grid_thw=np.array([[1, 2, 3]]))
        def decode(self, tokens, **kwargs):
            return raw
    class Model:
        device = 'TEST_ONLY_DEVICE'
        def generate(self, **kwargs):
            self.decoding = {k: kwargs[k] for k in ['max_new_tokens', 'do_sample']}
            return np.array([[1, 2, 2, 3]])
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=nullcontext))
    image_path = tmp_path / 'test_only.png'
    Image.new('RGB', (42, 28)).save(image_path)
    generator = Qwen25VLGenerator(revision='a'*40)
    generator._loaded = True
    generator.model, generator.processor = Model(), Processor()
    generator.loading_identity = {'TEST_ONLY_STAND_IN': True}
    question = 'Why is he carrying the surfboard?'
    entities = generator.extract_visual_entities(str(image_path), question)
    messages = generator.processor.messages
    assert [m['role'] for m in messages] == ['system', 'user']
    assert 'visual grounding extractor' in messages[0]['content']
    assert 'Do not answer the question or infer intentions' in messages[0]['content']
    user_text = messages[-1]['content'][1]['text']
    assert '{"entities":[]}' in user_text and '[...]' not in user_text
    assert user_text.endswith('Question: ' + question)
    record = generator.last_entity_record
    assert record['raw_text'] == raw
    assert record['generated_token_ids'] == [2, 3]
    assert record['prompt_version'] == 'evitrust-visible-entities-json-v2'
    assert record['decoding'] == generator.model.decoding == {'max_new_tokens': 64, 'do_sample': False}
    assert record['question'] == question
    assert record['image_dimensions_passed_to_processor'] == [42, 28]
    if valid:
        assert entities == record['entities'] == ['surfboard']
        assert record['parser_status'] == 'valid' and record['parser_error'] is None
    else:
        assert entities == record['entities'] == []
        assert record['parser_status'] == 'invalid_json' and record['parser_error']

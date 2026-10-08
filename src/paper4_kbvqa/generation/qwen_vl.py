from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

from paper4_kbvqa.generation.base import AnswerGenerator, build_prompt_payload
from paper4_kbvqa.types import Evidence


def _strict_json(text: str):
    """Decode one entire JSON value; duplicate keys and nonfinite values are invalid."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    return json.loads(text.strip(), object_pairs_hook=pairs, parse_constant=constant)


def _quantization_identity(model, requested_4bit: bool) -> dict:
    config = getattr(getattr(model, "config", None), "quantization_config", None)
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    config = dict(config) if isinstance(config, dict) else {}
    actual_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
    actual_8bit = bool(getattr(model, "is_loaded_in_8bit", False))
    quant_type = config.get("bnb_4bit_quant_type")
    if requested_4bit and (not actual_4bit or quant_type != "nf4"):
        raise RuntimeError("Requested NF4 4-bit loading was not confirmed by the loaded model")
    if not requested_4bit and (actual_4bit or actual_8bit):
        raise RuntimeError("Unrequested quantized loading was reported by the loaded model")
    return {"requested_4bit": bool(requested_4bit), "actual_4bit": actual_4bit,
            "quantization_mode": "4bit_nf4" if actual_4bit else "unquantized",
            "quantization_config": config}


def _validate_loaded_revision(model, requested: str):
    resolved = getattr(getattr(model, "config", None), "_commit_hash", None)
    if len(requested) == 40 and all(x in "0123456789abcdef" for x in requested.lower()):
        if resolved != requested:
            raise RuntimeError("Loaded model did not confirm the requested pinned revision")
    return resolved


class Qwen25VLGenerator(AnswerGenerator):
    """Frozen Qwen adapter; software tests do not establish genuine GPU execution.

    ``raw_confidence`` is whole-response token likelihood, including structure and
    citation tokens. It is not an answer-only probability or correctness confidence.
    """
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", revision="main", load_in_4bit=True, max_pixels=None, include_evidence_ids=True):
        self.model_name = model_name
        self.revision = revision
        self.load_in_4bit = load_in_4bit
        self.max_pixels = max_pixels
        self.include_evidence_ids = include_evidence_ids
        self._loaded = False
        self.loading_identity = None
        self.last_entity_record = None

    def _load(self):
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Install VLM extras in a CUDA environment") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen VLM benchmark execution requires a CUDA GPU")
        quant = None
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
            except Exception as exc:
                raise RuntimeError("4-bit mode requested but compatible quantization is unavailable") from exc
        proc_kwargs = {"max_pixels": int(self.max_pixels)} if self.max_pixels is not None else {}
        processor = AutoProcessor.from_pretrained(self.model_name, revision=self.revision,
                                                  trust_remote_code=False, **proc_kwargs)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_name,
            revision=self.revision, device_map="auto", quantization_config=quant,
            torch_dtype="auto", trust_remote_code=False)
        resolved = _validate_loaded_revision(model, self.revision)
        quant_identity = _quantization_identity(model, self.load_in_4bit)
        device_map = {key: str(value) for key, value in getattr(model, "hf_device_map", {}).items()}
        device = str(getattr(model, "device", "unknown"))
        if not (device.startswith("cuda") or any(v.startswith("cuda") or v.isdigit() for v in device_map.values())):
            raise RuntimeError("Loaded generator has no confirmed CUDA placement")
        model.eval()
        self.processor, self.model = processor, model
        self.loading_identity = {"model_id": self.model_name, "requested_revision": self.revision,
            "resolved_revision": resolved, "device": device, "device_map": device_map,
            "cuda_device": torch.cuda.get_device_name(), **quant_identity}
        self._loaded = True

    @staticmethod
    def _parse_details(text: str, supplied_ids: list[str], include_evidence_ids: bool = True) -> dict:
        schema = "answer_with_evidence_ids" if include_evidence_ids else "answer_only"
        result = {"answer": "", "parsed_ok": False, "parser_status": "invalid_json",
            "parser_error": None, "parser_schema": schema, "raw_evidence_ids": [],
            "valid_evidence_ids": [], "invalid_evidence_ids": []}
        try:
            obj = _strict_json(text)
        except (ValueError, TypeError) as exc:
            result["parser_error"] = str(exc)
            return result
        keys = {"answer", "evidence_ids"} if include_evidence_ids else {"answer"}
        if (not isinstance(obj, dict) or set(obj) != keys or not isinstance(obj.get("answer"), str)
                or not obj["answer"].strip() or (include_evidence_ids and
                (not isinstance(obj.get("evidence_ids"), list)
                 or any(not isinstance(value, str) for value in obj["evidence_ids"])))):
            result.update(parser_status="invalid_schema", parser_error="Expected exactly the declared answer schema")
            return result
        raw_ids = list(obj["evidence_ids"]) if include_evidence_ids else []
        allowed = set(supplied_ids)
        result.update(answer=obj["answer"].strip(), parsed_ok=True, parser_status="valid",
            parser_error=None, raw_evidence_ids=raw_ids,
            valid_evidence_ids=[eid for eid in raw_ids if eid in allowed],
            invalid_evidence_ids=[eid for eid in raw_ids if eid not in allowed])
        return result

    @staticmethod
    def _parse_output(text: str, fallback_ids: list[str], include_evidence_ids: bool = True) -> tuple[str, list[str]]:
        """Compatibility tuple API; parsing failure produces no answer or citations."""
        parsed = Qwen25VLGenerator._parse_details(text, fallback_ids, include_evidence_ids)
        return parsed["answer"], parsed["raw_evidence_ids"]

    @staticmethod
    def _parse_entities(text: str, max_entities: int = 8) -> tuple[list[str], str, str | None]:
        try:
            obj = _strict_json(text)
        except (ValueError, TypeError) as exc:
            return [], "invalid_json", str(exc)
        if (not isinstance(obj, dict) or set(obj) != {"entities"} or not isinstance(obj["entities"], list)
                or any(not isinstance(value, str) or not value.strip() for value in obj["entities"])):
            return [], "invalid_schema", "Expected exactly an entities list of nonempty strings"
        clean, seen = [], set()
        for value in obj["entities"]:
            value = value.strip()
            if value.lower() not in seen:
                clean.append(value)
                seen.add(value.lower())
        return clean[:max(0, int(max_entities))], "valid", None

    def _image_record(self, image_path, image, inputs) -> dict:
        grid = inputs.get("image_grid_thw")
        values = grid.tolist() if grid is not None else None
        patch = getattr(self.processor.image_processor, "patch_size", None)
        dims = [[row[2] * patch, row[1] * patch] for row in values] if values and patch else None
        return {"image_sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
            "image_dimensions_passed_to_processor": list(image.size),
            "image_grid_thw": values, "processor_patch_size": patch,
            "processed_image_dimensions": dims, "prompt_token_count": int(inputs.input_ids.shape[1])}

    def extract_visual_entities(self, image_path: str, question: str, max_entities: int = 8) -> list[str]:
        if not self._loaded:
            self._load()
        import torch
        from PIL import Image
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        prompt_version = "evitrust-visible-entities-json-v2"
        system = ("You are a visual grounding extractor. Return exactly one valid JSON object and nothing else. "
            "The object must have exactly the key entities, whose value is an array of nonempty strings. "
            "Describe only directly visible entities. Do not answer the question or infer intentions, causes, "
            "past events, or invisible facts. Do not include Markdown, commentary, or additional keys.")
        prompt = ("Identify only visible entities in the image that are relevant to the question. "
            "Use at most " + str(int(max_entities)) + " short noun phrases. "
            "The JSON schema is illustrated by {\"entities\":[]}; populate the array only with visible entities. "
            "Return an empty array when no relevant entities are visible.\nQuestion: " + question)
        messages = [{"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        gen = out[0][inputs.input_ids.shape[1]:]
        decoded = self.processor.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        entities, status, error = self._parse_entities(decoded, max_entities)
        self.last_entity_record = {"entities": entities, "raw_text": decoded,
            "generated_token_ids": gen.tolist(), "parser_status": status, "parser_error": error,
            "rendered_prompt": text, "prompt_version": prompt_version,
            "decoding": {"max_new_tokens": 64, "do_sample": False},
            "question": question, "loading_identity": self.loading_identity,
            **self._image_record(image_path, image, inputs)}
        return entities

    def generate(self, image_path: str, question: str, evidence: list[Evidence]) -> dict:
        if not self._loaded:
            self._load()
        import torch
        from PIL import Image
        payload = build_prompt_payload(question, evidence)
        evidence_text = "\n".join(f"[{x['id']}] {x['text']}" for x in payload['evidence']) or "[NONE] No external evidence supplied."
        prompt = ("Answer the visual question using the image and only relevant supplied evidence. "
            "Do not invent evidence. Return strict JSON with exactly keys answer and evidence_ids. "
            "answer must be a nonempty concise string. evidence_ids must be a list of strings containing only supplied IDs that directly support the answer.\n"
            f"Question: {question}\nEvidence:\n{evidence_text}")
        if not self.include_evidence_ids:
            evidence_text = "\n".join(x['text'] for x in payload['evidence']) or "No external evidence supplied."
            prompt = ("Answer the visual question using the image and only relevant supplied evidence. "
                "Return strict JSON with exactly the key answer; answer must be a nonempty concise string.\n"
                f"Question: {question}\nEvidence:\n{evidence_text}")
        prompt_version = ("evitrust-answer-evidence-json-v2" if self.include_evidence_ids
                          else "evitrust-answer-only-json-v2")
        system = ("You are a visual question answering assistant. Return exactly one valid JSON object and nothing else. "
            "Do not include Markdown, commentary, or additional keys. The answer must be a nonempty concise string. ")
        if self.include_evidence_ids:
            system += ("The object must have exactly the keys answer and evidence_ids. "
                "evidence_ids must be an array of strings containing only supplied evidence IDs that directly support "
                "the answer. Do not invent or alter IDs. Use an empty array when no supplied evidence supports the answer.")
        else:
            system += "The object must have exactly the key answer."
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        messages = [{"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        # The first live JSON answer exhausted 48 tokens mid-citation. Keep a
        # bounded budget with more room for the declared answer/citation schema.
        answer_max_new_tokens = 192
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=answer_max_new_tokens, do_sample=False,
                return_dict_in_generate=True, output_scores=True)
        prompt_len = int(inputs.input_ids.shape[1])
        gen_ids = out.sequences[0][prompt_len:]
        decoded = self.processor.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        token_logps = [float(torch.log_softmax(logits[0].float(), dim=-1)[int(token)].item())
                       for token, logits in zip(gen_ids, out.scores)]
        finite = len(token_logps) == len(gen_ids) and bool(token_logps) and all(math.isfinite(x) for x in token_logps)
        raw_conf = float(math.exp(sum(token_logps) / len(token_logps))) if finite else float("nan")
        parsed = self._parse_details(decoded, [e.evidence_id for e in evidence], self.include_evidence_ids)
        identity = {**self._image_record(image_path, image, inputs), "question": question,
            "ordered_evidence": [dict(id=e.evidence_id, text=e.text, source=e.source, uri=e.uri) for e in evidence],
            "rendered_prompt": text, "prompt_version": prompt_version,
            "model_id": self.model_name, "requested_revision": self.revision,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "model_config": self.model.config.to_dict(), "processor_config": self.processor.image_processor.to_dict(),
            "decoding": {"max_new_tokens": answer_max_new_tokens, "do_sample": False}, "seed": int(torch.initial_seed()),
            "four_bit": self.load_in_4bit, "max_pixels": self.max_pixels, "loading_identity": self.loading_identity,
            "generation_config": (self.model.generation_config.to_dict()
                if hasattr(getattr(self.model, "generation_config", None), "to_dict") else None)}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()
        return {**parsed, "raw_confidence": raw_conf, "raw_confidence_status": "finite" if finite else "invalid_token_scores",
            "supporting_evidence_ids": parsed["raw_evidence_ids"], "raw_text": decoded, "raw_output": decoded,
            "raw_confidence_interpretation": "uncalibrated_whole_response_token_likelihood",
            "generated_token_ids": gen_ids.tolist(), "generated_token_logprobs": token_logps,
            "answer_token_confidence": None, "answer_token_confidence_status": "not_implemented_whole_response_only",
            "prediction_key": key, "generation_identity": identity}

"""Opt-in JSON token constraints; strict output parsing remains authoritative.

The callback is local to one ``generate`` call. No model method is replaced and
no completed or truncated model output is repaired.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
from importlib import metadata
import json


LMFE_VERSION = "0.11.3"
CONSTRAINT_IDENTITY_VERSION = "evitrust-json-constraints-v1"
CONSTRAINT_SCHEMA_VERSION = "evitrust-generation-json-schema-v1"
LIKELIHOOD_SOURCE = "transformers_generate_unprocessed_logits_v1"


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), default=str).encode()).hexdigest()


def generation_schema(kind: str) -> dict:
    """Return a new exact-key schema, without constraining citation membership.

    ``minLength`` excludes empty strings, but whitespace-only strings still need
    the existing strict parser's strip-based validation. Unknown evidence IDs
    must reach that parser so citation-integrity diagnostics are preserved.
    """
    if kind == "entities":
        properties = {"entities": {"type": "array", "items": {"type": "string", "minLength": 1}}}
    elif kind in {"answer_only", "answer_with_evidence_ids"}:
        properties = {"answer": {"type": "string", "minLength": 1}}
        if kind == "answer_with_evidence_ids":
            properties["evidence_ids"] = {"type": "array", "items": {"type": "string"}}
    else:
        raise ValueError(f"Unknown generation constraint schema: {kind}")
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def require_json_constraints():
    """Load only the explicitly supported integration, with a useful failure."""
    try:
        version = metadata.version("lm-format-enforcer")
        if version != LMFE_VERSION:
            raise RuntimeError(f"JSON constrained decoding requires lm-format-enforcer=={LMFE_VERSION}; "
                f"found {version}")
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_token_enforcer_tokenizer_data, build_transformers_prefix_allowed_tokens_fn,
        )
    except (ImportError, metadata.PackageNotFoundError) as exc:
        raise RuntimeError("JSON constrained decoding requires lm-format-enforcer==0.11.3 "
            "and the VLM extras; install the vlm extra") from exc
    return JsonSchemaParser, build_transformers_prefix_allowed_tokens_fn, version, build_token_enforcer_tokenizer_data


def _tokenizer_identity(tokenizer) -> dict:
    vocab = tokenizer.get_vocab()
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_json = backend.to_str() if backend is not None and hasattr(backend, "to_str") else None
    init = getattr(tokenizer, "init_kwargs", {})
    return {"identity_version": "evitrust-constraint-tokenizer-v1",
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "requested_revision": init.get("revision"), "resolved_revision": init.get("_commit_hash"),
        "init_kwargs_sha256": _json_hash(init),
        "is_fast": bool(getattr(tokenizer, "is_fast", False)), "vocab_size": len(vocab),
        "vocab_sha256": _json_hash(vocab),
        "backend_tokenizer_sha256": hashlib.sha256(backend_json.encode()).hexdigest() if backend_json else None,
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "all_special_ids": list(getattr(tokenizer, "all_special_ids", [])),
        "special_tokens_map": dict(getattr(tokenizer, "special_tokens_map", {})),
        "added_vocab_sha256": _json_hash(tokenizer.get_added_vocab())
            if hasattr(tokenizer, "get_added_vocab") else None,
        "clean_up_tokenization_spaces": tokenizer.clean_up_tokenization_spaces}


def build_json_constraint(tokenizer, kind: str, *, tokenizer_data_cache: dict | None = None):
    """Build a fresh LMFE parser/callback and its effective decoding identity."""
    JsonSchemaParser, builder, version, data_builder = require_json_constraints()
    if tokenizer is None:
        raise RuntimeError("JSON constrained decoding requires processor.tokenizer")
    # LMFE 0.11.3 calls tokenizer.decode(tokens) without a cleanup argument.
    # Match the adapter's raw response decoding rather than collapsing spaces.
    tokenizer.clean_up_tokenization_spaces = False
    schema = generation_schema(kind)
    tokenizer_identity = _tokenizer_identity(tokenizer)
    tokenizer_input = tokenizer
    if tokenizer_data_cache is not None and tokenizer_identity["backend_tokenizer_sha256"] is not None:
        # Only reusable tokenizer preprocessing is retained. LMFE's documented
        # data object contains no prompt/prefix/parser state; its free-text
        # memoization depends only on token strings and requested string lengths.
        # Fingerprinting on every call also detects tokenizer changes.
        cache_key = _json_hash({"tokenizer": tokenizer_identity, "library_version": version})
        cached = tokenizer_data_cache.get("entry")
        if cached is None or cached[0] is not tokenizer or cached[1] != cache_key:
            cached = (tokenizer, cache_key, data_builder(tokenizer))
            tokenizer_data_cache["entry"] = cached
        tokenizer_input = cached[2]
    prefix_fn = builder(tokenizer_input, JsonSchemaParser(schema))
    # TokenEnforcer replaces the parser's initial config with one whose alphabet
    # comes from tokenizer data. Record that effective config, including limits.
    config = prefix_fn.token_enforcer.root_parser.config
    config_dict = asdict(config) if is_dataclass(config) else dict(vars(config))
    identity = {"identity_version": CONSTRAINT_IDENTITY_VERSION,
        "mode": "json_schema_prefix_allowed_tokens",
        "schema_version": CONSTRAINT_SCHEMA_VERSION, "schema_name": kind,
        "schema": schema, "schema_sha256": _json_hash(schema),
        "library": {"identity_version": "evitrust-constraint-library-v1",
                    "name": "lm-format-enforcer", "version": version},
        "integration": "build_transformers_prefix_allowed_tokens_fn",
        "parser_config_version": "lm-format-enforcer-0.11.3-character-parser-config-v1",
        "parser_config": config_dict, "parser_config_sha256": _json_hash(config_dict),
        "tokenizer": tokenizer_identity,
        "tokenizer_data_cache_policy": "reuse_only_same_tokenizer_and_full_backend_fingerprint_v1",
        "token_enforcer_vocab_size": getattr(prefix_fn.token_enforcer, "vocab_size", None),
        "state_scope": "fresh_parser_and_token_enforcer_per_generate_call"}
    return prefix_fn, identity

#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "src"))

import argparse, json, hashlib, subprocess, time
from dataclasses import asdict
from pathlib import Path

from paper4_kbvqa.data.manifest import load_jsonl
from paper4_kbvqa.execution.controller import ExecutionConfig
from paper4_kbvqa.execution.variants import MatchedVariantRunner
from paper4_kbvqa.filtering.relevance import RelevanceFilter
from paper4_kbvqa.generation.qwen_vl import Qwen25VLGenerator
from paper4_kbvqa.knowledge.combined import CombinedKnowledgeProvider
from paper4_kbvqa.knowledge.http_providers import (
    ConceptNetProvider, WikidataProvider, WikipediaProvider,
)
from paper4_kbvqa.verification.llm_critic import QwenLabelLikelihoodCritic


def make_provider(sources, cache_dir):
    mapping = {
        "wikipedia": lambda: WikipediaProvider(Path(cache_dir)/"wikipedia"),
        "wikidata": lambda: WikidataProvider(Path(cache_dir)/"wikidata"),
        "conceptnet": lambda: ConceptNetProvider(Path(cache_dir)/"conceptnet"),
    }
    providers=[]
    for s in sources:
        s=s.strip().lower()
        if s not in mapping:
            raise ValueError(f"Unknown source: {s}")
        providers.append(mapping[s]())
    from paper4_kbvqa.knowledge.structured import StructuredKnowledgeProvider
    return StructuredKnowledgeProvider(providers)


def main():
    ap=argparse.ArgumentParser(description="Run clean matched EviTrust B0-B3 variants.")
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--variant",choices=["B0","B1","B2","B3"],required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--generator-model",default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--generator-revision",default="main")
    ap.add_argument("--critic-revision",default="main")
    ap.add_argument("--critic-backend",choices=["qwen","deberta"],default="qwen")
    ap.add_argument("--filter-backend",choices=["transparent","semantic"],default="transparent")
    ap.add_argument("--reranker-revision",default="main")
    ap.add_argument("--corruption",default="clean")
    ap.add_argument("--distractor-predictions")
    ap.add_argument("--seed",type=int,default=2026)
    ap.add_argument("--provenance-weight",type=float,default=.25)
    ap.add_argument("--contradiction-penalty",type=float,default=.35)
    ap.add_argument("--critic-model",default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--sources",default="wikipedia,wikidata,conceptnet")
    ap.add_argument("--cache-dir",default="cache/knowledge")
    ap.add_argument("--retrieval-top-k",type=int,default=10)
    ap.add_argument("--filtered-top-k",type=int,default=5)
    ap.add_argument("--max-samples",type=int)
    ap.add_argument("--max-pixels",type=int)
    ap.add_argument("--no-evidence-ids",action="store_true")
    ap.add_argument("--no-4bit",action="store_true")
    args=ap.parse_args()

    samples=load_jsonl(args.manifest)
    if args.max_samples is not None:
        samples=samples[:args.max_samples]

    from paper4_kbvqa.execution.study import freeze_json, CorruptionTransform
    from paper4_kbvqa.utils.environment import collect_environment
    from paper4_kbvqa.utils.seed import set_seed
    set_seed(args.seed)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; no benchmark output was generated")
    identity={"arguments":vars(args), "manifest_sha256":hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
              "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=_ROOT,text=True).strip()}
    freeze_json(Path(args.out).with_suffix(".identity.json"), identity)
    t0=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    generator=Qwen25VLGenerator(
        model_name=args.generator_model,
        revision=args.generator_revision,
        load_in_4bit=not args.no_4bit,
        max_pixels=args.max_pixels,
        include_evidence_ids=not args.no_evidence_ids,
    )

    provider=None
    evidence_filter=None
    critic=None
    if args.variant != "B0":
        provider=make_provider(args.sources.split(","),args.cache_dir)
    if args.variant in {"B2","B3"}:
        if args.filter_backend == "semantic":
            from paper4_kbvqa.filtering.semantic_reranker import SentenceTransformerReranker
            evidence_filter=SentenceTransformerReranker(revision=args.reranker_revision)
        else:
            evidence_filter=RelevanceFilter()
    if args.variant=="B3":
        if args.critic_backend == "deberta":
            from paper4_kbvqa.verification.nli_verifier import DeBERTaNLIVerifier
            critic=DeBERTaNLIVerifier(model_id=args.critic_model,revision=args.critic_revision,device="cpu")
        else:
            critic=QwenLabelLikelihoodCritic(model_id=args.critic_model,
                revision=args.critic_revision,load_in_4bit=not args.no_4bit)

    runner=MatchedVariantRunner(
        variant=args.variant,
        generator=generator,
        provider=provider,
        evidence_filter=evidence_filter,
        critic=critic,
        corruption=CorruptionTransform(args.corruption,args.seed,args.distractor_predictions),
        config=ExecutionConfig(
            retrieval_top_k=args.retrieval_top_k,
            filtered_top_k=args.filtered_top_k,
            provenance_weight=args.provenance_weight,
            contradiction_penalty=args.contradiction_penalty,
        ),
    )
    state=runner.run_manifest(
        samples,
        output_jsonl=args.out,
        checkpoint_json=args.checkpoint,
    )
    state.update({"session_elapsed_s":time.perf_counter()-t0,
                  "session_peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                  "session_peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                  "environment":collect_environment(),"identity":identity})
    audit=Path(args.out).with_suffix(".sessions.jsonl")
    with audit.open("a",encoding="utf-8") as f: f.write(json.dumps(state)+"\n")
    print(json.dumps(state,indent=2))
    if state["completed"] != len(samples):
        raise RuntimeError("Incomplete inference; results are not eligible for evaluation")

if __name__=="__main__":
    main()

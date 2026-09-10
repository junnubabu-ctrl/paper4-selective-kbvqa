#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "src"))

import argparse, json
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
    return CombinedKnowledgeProvider(providers) if len(providers)>1 else providers[0]


def main():
    ap=argparse.ArgumentParser(description="Run clean matched EviTrust B0-B3 variants.")
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--variant",choices=["B0","B1","B2","B3"],required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--generator-model",default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--generator-revision",default="main")
    ap.add_argument("--critic-model",default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--sources",default="wikipedia,wikidata,conceptnet")
    ap.add_argument("--cache-dir",default="cache/knowledge")
    ap.add_argument("--retrieval-top-k",type=int,default=10)
    ap.add_argument("--filtered-top-k",type=int,default=5)
    ap.add_argument("--max-samples",type=int)
    ap.add_argument("--max-pixels",type=int)
    ap.add_argument("--no-4bit",action="store_true")
    args=ap.parse_args()

    samples=load_jsonl(args.manifest)
    if args.max_samples is not None:
        samples=samples[:args.max_samples]

    generator=Qwen25VLGenerator(
        model_name=args.generator_model,
        revision=args.generator_revision,
        load_in_4bit=not args.no_4bit,
        max_pixels=args.max_pixels,
    )

    provider=None
    evidence_filter=None
    critic=None
    if args.variant != "B0":
        provider=make_provider(args.sources.split(","),args.cache_dir)
    if args.variant in {"B2","B3"}:
        evidence_filter=RelevanceFilter()
    if args.variant=="B3":
        critic=QwenLabelLikelihoodCritic(
            model_id=args.critic_model,
            load_in_4bit=not args.no_4bit,
        )

    runner=MatchedVariantRunner(
        variant=args.variant,
        generator=generator,
        provider=provider,
        evidence_filter=evidence_filter,
        critic=critic,
        config=ExecutionConfig(
            retrieval_top_k=args.retrieval_top_k,
            filtered_top_k=args.filtered_top_k,
        ),
    )
    state=runner.run_manifest(
        samples,
        output_jsonl=args.out,
        checkpoint_json=args.checkpoint,
    )
    print(json.dumps(state,indent=2))

if __name__=="__main__":
    main()

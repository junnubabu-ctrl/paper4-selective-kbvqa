from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path: _sys.path.insert(0, str(_ROOT / "src"))
import json, time, traceback
from pathlib import Path
from paper4_kbvqa.data.manifest import load_jsonl
from paper4_kbvqa.generation.qwen_vl import Qwen25VLGenerator
from paper4_kbvqa.knowledge.http_providers import WikipediaProvider, WikidataProvider, ConceptNetProvider
from paper4_kbvqa.knowledge.combined import CombinedKnowledgeProvider
from paper4_kbvqa.pipeline import KBVQAPipeline
from paper4_kbvqa.utils.checkpoint import CheckpointManager, CheckpointState


def make_provider(sources: list[str], cache_dir: str):
    mapping={
        "wikipedia": lambda: WikipediaProvider(Path(cache_dir)/"wikipedia"),
        "wikidata": lambda: WikidataProvider(Path(cache_dir)/"wikidata"),
        "conceptnet": lambda: ConceptNetProvider(Path(cache_dir)/"conceptnet"),
    }
    providers=[]
    for s in sources:
        if s not in mapping: raise ValueError(f"Unknown source {s!r}")
        providers.append(mapping[s]())
    return CombinedKnowledgeProvider(providers) if len(providers)>1 else providers[0]


def run(args, mode: str):
    samples=load_jsonl(args.manifest)
    if args.max_samples is not None: samples=samples[:args.max_samples]
    provider=None if mode=="no_knowledge" else make_provider(args.sources.split(","), args.cache_dir)
    generator=Qwen25VLGenerator(model_name=args.model,revision=args.revision,load_in_4bit=not args.no_4bit,max_pixels=args.max_pixels)
    pipeline=KBVQAPipeline(generator,provider=provider)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    ckpt=CheckpointManager(args.checkpoint)
    state=ckpt.load() or CheckpointState(run_id=args.run_id)
    done=set(state.completed_question_ids)
    processed=0
    with out.open("a",encoding="utf-8") as f:
        for sample in samples:
            if sample.question_id in done: continue
            t0=time.perf_counter()
            try:
                inference=sample.inference_view()
                result=pipeline.run_one(
                    **inference,
                    mode=mode,
                    retrieval_limit=args.retrieval_limit,
                    top_k=args.top_k,
                    auto_extract_entities=not args.no_auto_entities,
                )
                result["latency_s"]=time.perf_counter()-t0
                f.write(json.dumps(result,ensure_ascii=False)+"\n"); f.flush()
                state.completed_question_ids.append(sample.question_id)
            except Exception as e:
                state.errors.append({"question_id":sample.question_id,"error":repr(e),"traceback":traceback.format_exc()[-4000:]})
                ckpt.save(state)
                if args.fail_fast: raise
            processed += 1
            if processed % args.checkpoint_every == 0: ckpt.save(state)
    ckpt.save(state)
    return state

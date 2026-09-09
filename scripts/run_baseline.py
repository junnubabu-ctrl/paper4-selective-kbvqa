#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path: _sys.path.insert(0, str(_ROOT / "src"))
import argparse
from _runner import run

def main():
    ap=argparse.ArgumentParser(description="Run B0 (VLM only) or B1 (VLM + raw retrieved knowledge)")
    ap.add_argument("--manifest",required=True); ap.add_argument("--baseline",choices=["B0","B1"],required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--checkpoint",required=True); ap.add_argument("--run-id",default="baseline")
    ap.add_argument("--model",default="Qwen/Qwen2.5-VL-3B-Instruct"); ap.add_argument("--revision",default="main")
    ap.add_argument("--sources",default="wikipedia,wikidata,conceptnet"); ap.add_argument("--cache-dir",default="cache/knowledge")
    ap.add_argument("--retrieval-limit",type=int,default=10); ap.add_argument("--top-k",type=int,default=5); ap.add_argument("--max-samples",type=int)
    ap.add_argument("--checkpoint-every",type=int,default=10); ap.add_argument("--max-pixels",type=int); ap.add_argument("--no-auto-entities",action="store_true"); ap.add_argument("--no-4bit",action="store_true"); ap.add_argument("--fail-fast",action="store_true")
    args=ap.parse_args(); state=run(args,"no_knowledge" if args.baseline=="B0" else "raw_knowledge")
    print(f"completed={len(state.completed_question_ids)} errors={len(state.errors)}")
if __name__=="__main__": main()

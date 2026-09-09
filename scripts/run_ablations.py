#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path: _sys.path.insert(0, str(_ROOT / "src"))

import argparse
import json
import subprocess
from pathlib import Path


def build_commands(args) -> list[dict]:
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    common = [
        "--manifest", args.manifest,
        "--model", args.model,
        "--revision", args.revision,
        "--cache-dir", args.cache_dir,
        "--checkpoint-every", str(args.checkpoint_every),
    ]
    if args.max_samples is not None:
        common += ["--max-samples", str(args.max_samples)]
    if args.no_4bit:
        common += ["--no-4bit"]

    matrix: list[dict] = []
    for baseline in ("B0", "B1"):
        name = baseline.lower()
        cmd = [_sys.executable, str(_ROOT / "scripts" / "run_baseline.py"), *common, "--baseline", baseline, "--sources", args.sources, "--out", str(root / f"{name}.jsonl"), "--checkpoint", str(root / f"{name}.checkpoint.json"), "--run-id", name]
        matrix.append({"name": name, "kind": "baseline", "command": cmd})
    source_sets = [[s] for s in args.sources.split(",") if s.strip()]
    source_sets.append([s for s in args.sources.split(",") if s.strip()])
    seen = set()
    for sources in source_sets:
        key = "+".join(sources)
        if key in seen: continue
        seen.add(key)
        for top_k in args.top_k:
            name = f"full_sources-{key}_topk-{top_k}"
            cmd = [_sys.executable, str(_ROOT / "scripts" / "run_proposed.py"), *common, "--sources", ",".join(sources), "--top-k", str(top_k), "--out", str(root / f"{name}.jsonl"), "--checkpoint", str(root / f"{name}.checkpoint.json"), "--run-id", name]
            matrix.append({"name": name, "kind": "source_topk", "command": cmd})
    return matrix


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate and optionally execute a controlled Paper-4 ablation matrix")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default="results/ablations")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--sources", default="wikipedia,wikidata,conceptnet")
    ap.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--cache-dir", default="cache/knowledge")
    ap.add_argument("--max-samples", type=int)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--execute", action="store_true", help="Actually run commands; requires a suitable GPU for VLM experiments")
    args = ap.parse_args()
    matrix = build_commands(args)
    manifest = Path(args.out_dir) / "ablation_matrix.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    print(f"Wrote {len(matrix)} controlled runs to {manifest}")
    if not args.execute:
        print("Dry run only. Pass --execute on a CUDA machine after the development gate passes.")
        return
    for item in matrix:
        print(f"[RUN] {item['name']}")
        subprocess.run(item["command"], check=True)

if __name__ == "__main__": main()

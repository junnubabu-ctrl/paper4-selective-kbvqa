#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "src"))

import argparse, json, random
from pathlib import Path
from paper4_kbvqa.data.manifest import load_jsonl


def write(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        for s in rows:
            payload={
                "question_id":s.question_id,
                "image_path":s.image_path,
                "question":s.question,
                "answers":list(s.answers),
                "visual_entities":list(s.visual_entities),
                "metadata":s.metadata,
            }
            f.write(json.dumps(payload,ensure_ascii=False)+"\n")


def main():
    ap=argparse.ArgumentParser(description="Deterministically split a labelled manifest into calibration and evaluation partitions.")
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--calibration-out",required=True)
    ap.add_argument("--evaluation-out",required=True)
    ap.add_argument("--calibration-fraction",type=float,default=0.5)
    ap.add_argument("--seed",type=int,default=2026)
    args=ap.parse_args()
    if not 0 < args.calibration_fraction < 1:
        raise ValueError("calibration-fraction must be in (0,1)")
    rows=load_jsonl(args.manifest)
    ids=list(range(len(rows)))
    random.Random(args.seed).shuffle(ids)
    n=max(1,min(len(rows)-1,round(len(rows)*args.calibration_fraction)))
    cal_idx=set(ids[:n])
    cal=[x for i,x in enumerate(rows) if i in cal_idx]
    ev=[x for i,x in enumerate(rows) if i not in cal_idx]
    write(args.calibration_out,cal); write(args.evaluation_out,ev)
    print(json.dumps({"seed":args.seed,"n_total":len(rows),"n_calibration":len(cal),"n_evaluation":len(ev)},indent=2))


if __name__=="__main__":
    main()

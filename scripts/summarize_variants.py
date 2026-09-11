#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    ap=argparse.ArgumentParser(description="Assemble machine-readable B0-B5 result summary from evaluator outputs.")
    ap.add_argument("--b0",required=True); ap.add_argument("--b1",required=True)
    ap.add_argument("--b2",required=True); ap.add_argument("--b3",required=True)
    ap.add_argument("--b4",required=True); ap.add_argument("--b5",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    payload={}
    for name,path in [("B0",args.b0),("B1",args.b1),("B2",args.b2),("B3",args.b3),("B4",args.b4),("B5",args.b5)]:
        x=load(path)
        row={
            "n":x.get("n_evaluated"),
            "vqa_score":x.get("soft_accuracy_percent"),
            "full_credit_accuracy":x.get("full_credit_accuracy_percent"),
        }
        cal=x.get("calibration") or {}
        sel=x.get("selective") or {}
        row.update({
            "ece":cal.get("ece"),
            "brier":cal.get("brier"),
            "aurc":cal.get("aurc"),
            "coverage":sel.get("coverage",1.0 if name!="B5" else None),
            "risk":sel.get("selective_risk"),
            "selective_soft_accuracy":sel.get("soft_accuracy_percent"),
        })
        payload[name]=row
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2,allow_nan=True),encoding="utf-8")
    print(json.dumps(payload,indent=2,allow_nan=True))


if __name__=="__main__":
    main()

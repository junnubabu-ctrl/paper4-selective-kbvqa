#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path: _sys.path.insert(0, str(_ROOT / "src"))
import argparse,json
from pathlib import Path
from paper4_kbvqa.calibration.calibrators import PlattCalibrator
from paper4_kbvqa.selective.decision import selective_decision

def main():
    ap=argparse.ArgumentParser(description="Apply a frozen validation-fitted calibration/threshold policy to held-out predictions")
    ap.add_argument("--policy",required=True); ap.add_argument("--predictions",required=True); ap.add_argument("--out",required=True); args=ap.parse_args()
    policy=json.loads(Path(args.policy).read_text(encoding="utf-8")); cal=PlattCalibrator.from_dict(policy["calibrator"]); t=float(policy["threshold"])
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    with open(args.predictions,encoding="utf-8") as src, open(args.out,"w",encoding="utf-8") as dst:
        for line in src:
            if not line.strip(): continue
            x=json.loads(line); c=float(cal.predict([float(x["raw_confidence"])])[0]); answer=selective_decision(c,t)
            x["calibrated_confidence"]=c; x["threshold"]=t; x["abstain"]=not answer; x["selective_answer"]=x["answer"] if answer else "<ABSTAIN>"
            dst.write(json.dumps(x,ensure_ascii=False)+"\n")
if __name__=="__main__": main()

#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path: _sys.path.insert(0, str(_ROOT / "src"))
import argparse, json
from pathlib import Path
from paper4_kbvqa.calibration.calibrators import PlattCalibrator
from paper4_kbvqa.data.manifest import load_jsonl
from paper4_kbvqa.evaluation.aokvqa import direct_answer_score
from paper4_kbvqa.selective.decision import choose_threshold_for_target_risk
from paper4_kbvqa.evaluation.metrics import expected_calibration_error,brier_score,aurc


def load_predictions(path):
    d={}
    with open(path,encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x=json.loads(line); d[str(x["question_id"])]=x
    return d

def main():
    ap=argparse.ArgumentParser(description="Fit Platt calibration + selective threshold on validation data only")
    ap.add_argument("--manifest",required=True); ap.add_argument("--predictions",required=True); ap.add_argument("--out",required=True); ap.add_argument("--target-risk",type=float,default=.05)
    args=ap.parse_args()
    manifest=load_jsonl(args.manifest); preds=load_predictions(args.predictions)
    from paper4_kbvqa.execution.study import validate_predictions
    validate_predictions(args.manifest,args.predictions)
    scores=[]; correct=[]; used=[]
    for s in manifest:
        if s.metadata.get("difficult_direct_answer",False): continue
        p=preds.get(s.question_id)
        if p is None or not s.answers: continue
        # Full-credit event: prediction receives maximal A-OKVQA/VQA-style agreement credit.
        soft=direct_answer_score(str(p["answer"]), list(s.answers))
        scores.append(float(p["raw_confidence"])); correct.append(int(soft>=1.0)); used.append(s.question_id)
    if len(set(correct))<2: raise RuntimeError("Validation labels need both correct and incorrect examples for calibration")
    cal=PlattCalibrator().fit(scores,correct); calibrated=cal.predict(scores)
    policy=choose_threshold_for_target_risk(calibrated,correct,target_risk=args.target_risk)
    payload={
        "calibrator":cal.to_dict(),"threshold":policy["threshold"],"validation_coverage":policy["coverage"],"validation_risk":policy["risk"],
        "target_risk":args.target_risk,"n_validation":len(scores),"correctness_event":"full_credit",
        "validation_ece":expected_calibration_error(calibrated,correct),"validation_brier":brier_score(calibrated,correct),"validation_aurc":aurc(calibrated,correct),
        "question_ids":used,
    }
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in payload.items() if k not in {"question_ids","calibrator"}},indent=2))
if __name__=="__main__": main()

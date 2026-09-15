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
    ap.add_argument("--dataset",choices=["aokvqa","okvqa"],default="aokvqa")
    args=ap.parse_args()
    manifest=load_jsonl(args.manifest); preds=load_predictions(args.predictions)
    from paper4_kbvqa.execution.study import validate_predictions
    validate_predictions(args.manifest,args.predictions)
    official_scores={}
    if args.dataset=='okvqa':
        from paper4_kbvqa.evaluation.okvqa import score_predictions
        official_scores=score_predictions(manifest,preds)
    scores=[]; correct=[]; used=[]
    for s in manifest:
        if s.metadata.get("difficult_direct_answer",False): continue
        p=preds.get(s.question_id)
        if p is None or not s.answers: continue
        # Full-credit event: prediction receives maximal A-OKVQA/VQA-style agreement credit.
        soft=official_scores[s.question_id] if args.dataset=="okvqa" else direct_answer_score(str(p["answer"]), list(s.answers))
        scores.append(float(p["raw_confidence"])); correct.append(int(soft>=1.0)); used.append(s.question_id)
    from paper4_kbvqa.data.partitions import partition_groups, image_group
    metadata={s.question_id:{"question_id":s.question_id,"image_path":s.image_path,"metadata":s.metadata} for s in manifest}
    indexed=[{**metadata[qid],"score":score,"correct":label} for qid,score,label in zip(used,scores,correct)]
    fit_rows,threshold_rows=partition_groups(indexed,seed=2026)
    fit_scores=[x["score"] for x in fit_rows]; fit_correct=[x["correct"] for x in fit_rows]
    if len(set(fit_correct))<2: raise RuntimeError("Fit partition requires both correct and incorrect examples; increase calibration sample size")
    cal=PlattCalibrator().fit(fit_scores,fit_correct)
    calibrated=cal.predict([x["score"] for x in threshold_rows])
    threshold_correct=[x["correct"] for x in threshold_rows]
    policy=choose_threshold_for_target_risk(calibrated,threshold_correct,target_risk=args.target_risk)
    payload={
        "dataset":args.dataset,"calibrator":cal.to_dict(),"threshold":policy["threshold"],"validation_coverage":policy["coverage"],"validation_risk":policy["risk"],
        "target_risk":args.target_risk,"n_validation":len(scores),"correctness_event":"full_credit",
        "validation_ece":expected_calibration_error(calibrated,threshold_correct),"validation_brier":brier_score(calibrated,threshold_correct),"validation_aurc":aurc(calibrated,threshold_correct),
        "fit_question_ids":[x["question_id"] for x in fit_rows],
        "threshold_question_ids":[x["question_id"] for x in threshold_rows],
        "calibration_image_groups":sorted({image_group(x) for x in indexed}),
        "partition_seed":2026,"n_fit":len(fit_rows),"n_threshold":len(threshold_rows),
        "risk_interpretation":"empirical threshold-partition full-credit error; no finite-sample guarantee",
        "question_ids":used,
    }
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in payload.items() if k not in {"question_ids","calibrator"}},indent=2))
if __name__=="__main__": main()

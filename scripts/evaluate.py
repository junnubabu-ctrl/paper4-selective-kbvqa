from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path: _sys.path.insert(0, str(_ROOT / "src"))

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from paper4_kbvqa.data.manifest import load_jsonl
from paper4_kbvqa.evaluation.aokvqa import direct_answer_score
from paper4_kbvqa.evaluation.metrics import (
    aurc,
    brier_score,
    expected_calibration_error,
    selective_metrics,
)


def _norm(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return " ".join(text.split())


def _vqa_consensus_aux(pred: str, answers: list[str]) -> float:
    """Auxiliary raw-string consensus only; not a replacement for official OK-VQA evaluation."""
    p = _norm(pred)
    matches = sum(_norm(a) == p for a in answers)
    return min(matches / 3.0, 1.0)


def _load_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    out = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["question_id"])
            if qid in out:
                raise ValueError(f"Duplicate prediction question_id: {qid}")
            out[qid] = row
    return out


def evaluate_rows(manifest_path: str, prediction_path: str, *, dataset: str, confidence_field: str, threshold: float | None) -> dict[str, Any]:
    samples = load_jsonl(manifest_path)
    preds = _load_predictions(prediction_path)
    per_question: list[dict[str, Any]] = []
    missing = []

    for s in samples:
        if not s.answers:
            continue
        p = preds.get(s.question_id)
        if p is None:
            missing.append(s.question_id)
            continue
        answer = str(p.get("answer", ""))
        if dataset == "aokvqa":
            soft = direct_answer_score(answer, list(s.answers))
            metric_name = "A-OKVQA direct-answer official formula"
        else:
            soft = _vqa_consensus_aux(answer, list(s.answers))
            metric_name = "auxiliary raw-string VQA consensus (run official OK-VQA evaluator for publication)"
        confidence = p.get(confidence_field)
        per_question.append({
            "question_id": s.question_id,
            "answer": answer,
            "soft_score": float(soft),
            "correct_full_credit": int(soft >= 1.0),
            "confidence": None if confidence is None else float(confidence),
        })

    if missing:
        raise RuntimeError(f"Missing {len(missing)} labelled predictions; first={missing[:5]}")
    if not per_question:
        raise RuntimeError("No labelled samples were evaluated")

    soft_accuracy = 100.0 * sum(x["soft_score"] for x in per_question) / len(per_question)
    payload: dict[str, Any] = {
        "dataset": dataset,
        "metric": metric_name,
        "n_evaluated": len(per_question),
        "soft_accuracy_percent": soft_accuracy,
        "full_credit_accuracy_percent": 100.0 * sum(x["correct_full_credit"] for x in per_question) / len(per_question),
        "confidence_field": confidence_field,
        "threshold": threshold,
        "per_question": per_question,
    }

    usable = [x for x in per_question if x["confidence"] is not None]
    if usable:
        conf = [x["confidence"] for x in usable]
        correct = [x["correct_full_credit"] for x in usable]
        payload["calibration"] = {
            "n": len(usable),
            "ece": expected_calibration_error(conf, correct),
            "brier": brier_score(conf, correct),
            "aurc": aurc(conf, correct),
        }
        if threshold is not None:
            payload["selective"] = selective_metrics(conf, correct, threshold)
            answered = [x for x in usable if x["confidence"] >= threshold]
            payload["selective"]["soft_accuracy_percent"] = (
                100.0 * sum(x["soft_score"] for x in answered) / len(answered)
                if answered else float("nan")
            )

    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate recorded Paper-4 predictions; never generates results itself")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--dataset", choices=["aokvqa", "okvqa"], required=True)
    ap.add_argument("--confidence-field", default="raw_confidence")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payload = evaluate_rows(
        args.manifest,
        args.predictions,
        dataset=args.dataset,
        confidence_field=args.confidence_field,
        threshold=args.threshold,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    summary = {k: v for k, v in payload.items() if k != "per_question"}
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

def direct_answer_score(prediction: str, direct_answers: list[str] | tuple[str, ...]) -> float:
    """A-OKVQA direct-answer score used by the official evaluator: min(matches/3, 1)."""
    matches = sum(str(prediction) == str(a) for a in direct_answers)
    return min(1.0, matches / 3.0)

def direct_answer_accuracy(predictions: dict[str, str], dataset: list[dict], skip_difficult: bool = True) -> float:
    scores=[]
    for row in dataset:
        if skip_difficult and row.get("difficult_direct_answer", False):
            continue
        qid=str(row["question_id"])
        pred=str(predictions.get(qid, ""))
        scores.append(direct_answer_score(pred, row.get("direct_answers", [])))
    return 100.0 * sum(scores) / len(scores) if scores else float("nan")

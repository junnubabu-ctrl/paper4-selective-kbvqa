"""Adapter to the upstream VQA evaluator linked by the official OK-VQA site."""
from __future__ import annotations
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from types import SimpleNamespace
from paper4_kbvqa.evaluation.vendor.vqa_eval import VQAEval


def score_predictions(samples,predictions):
    qa={}; results={}
    for sample in samples:
        qid=str(sample.question_id)
        if qid not in predictions: raise ValueError('Missing OK-VQA prediction: '+qid)
        annotation=deepcopy(sample.metadata.get('official_annotation'))
        if not annotation:
            raise ValueError('OK-VQA manifest lacks official_annotation; rebuild using the v1.1 annotations')
        if not annotation.get('answers'): raise ValueError('No reference answers: '+qid)
        qa[qid]=annotation
        results[qid]={'answer':str(predictions[qid]['answer'])}
    if not qa: raise ValueError('No OK-VQA samples')
    ground_truth=SimpleNamespace(qa=qa,getQuesIds=lambda:list(qa))
    result=SimpleNamespace(qa=results)
    # Preserve precision for bootstrap and full-credit classification. The
    # upstream default rounds per-question scores to two percentage decimals.
    evaluator=VQAEval(ground_truth,result,n=12)
    with redirect_stdout(StringIO()): evaluator.evaluate()
    return {str(qid):float(score)/100 for qid,score in evaluator.evalQA.items()}

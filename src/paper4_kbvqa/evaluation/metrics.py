from __future__ import annotations
import numpy as np

def brier_score(confidence, correct):
    p=np.asarray(confidence,float); y=np.asarray(correct,float); return float(np.mean((p-y)**2))

def expected_calibration_error(confidence, correct, bins: int=10):
    p=np.asarray(confidence,float); y=np.asarray(correct,float); total=len(p)
    if total==0: return float('nan')
    edges=np.linspace(0,1,bins+1); ece=0.0
    for i in range(bins):
        lo,hi=edges[i],edges[i+1]; m=(p>=lo)&((p<hi) if i<bins-1 else (p<=hi))
        if m.any(): ece += m.mean()*abs(float(y[m].mean())-float(p[m].mean()))
    return float(ece)

def selective_metrics(confidence, correct, threshold):
    p=np.asarray(confidence,float); y=np.asarray(correct,bool); m=p>=threshold
    coverage=float(m.mean()) if len(m) else 0.0
    if not m.any(): return {"coverage":coverage,"selective_accuracy":float('nan'),"selective_risk":float('nan'),"answered":0}
    acc=float(y[m].mean()); return {"coverage":coverage,"selective_accuracy":acc,"selective_risk":1-acc,"answered":int(m.sum())}

def _validated_risk_inputs(confidence, correct):
    p = np.asarray(confidence)
    y = np.asarray(correct)
    if p.ndim != 1 or y.ndim != 1:
        raise ValueError("confidence and correct must be one-dimensional")
    if len(p) != len(y):
        raise ValueError("confidence and correct must have equal length")
    if p.dtype.kind not in "biuf":
        raise ValueError("confidence must contain only real numeric values")
    p = p.astype(float)
    if not np.all(np.isfinite(p)):
        raise ValueError("confidence must contain only finite values")
    if y.dtype.kind not in "biuf" or not np.all((y == 0) | (y == 1)):
        raise ValueError("correct must contain only numeric or boolean binary values (0 or 1)")
    return p, y.astype(bool)


def risk_coverage_curve(confidence, correct):
    """Return risks at complete equal-confidence groups in descending order.

    Every point corresponds to the selection rule ``confidence >= threshold``.
    Ties are never split, so results do not depend on the input order.
    Inputs must use real numeric dtypes; strings, objects and complex scores
    are rejected rather than silently converted.
    """
    p, y = _validated_risk_inputs(confidence, correct)
    if len(y) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    order = np.argsort(-p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    group_ends = np.r_[np.flatnonzero(sorted_p[:-1] != sorted_p[1:]) + 1, len(y)]
    coverage = group_ends / len(y)
    risk = np.cumsum(~sorted_y)[group_ends - 1] / group_ends
    return coverage.astype(float), risk.astype(float)

def aurc(confidence, correct):
    """Compute the whole-tie, right-endpoint risk integral from coverage zero."""
    coverage, risk = risk_coverage_curve(confidence, correct)
    if len(coverage) == 0:
        return float('nan')
    widths = np.diff(np.r_[0.0, coverage])
    return float(np.sum(widths * risk))

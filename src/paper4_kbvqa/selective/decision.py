from __future__ import annotations
import numpy as np

def selective_decision(confidence: float, threshold: float) -> bool:
    """Return True when the system should answer, False when it should abstain."""
    return float(confidence) >= float(threshold)

def choose_threshold_for_target_risk(confidences, correct, target_risk: float=0.05) -> dict:
    """Validation-only threshold selection maximizing coverage subject to empirical risk <= target."""
    c=np.asarray(confidences,float); y=np.asarray(correct,bool)
    if len(c) == 0 or len(c) != len(y) or not np.isfinite(c).all():
        raise ValueError("Nonempty, finite, equally sized inputs are required")
    if not 0 <= target_risk <= 1 or ((c < 0) | (c > 1)).any():
        raise ValueError("Confidence and target risk must be in [0, 1]")
    # A threshold of 1 still answers predictions with confidence exactly 1.
    reject_all=float(np.nextafter(1.0, np.inf))
    candidates=np.unique(np.concatenate(([0.0],c,[reject_all])))
    feasible=[]
    for t in candidates:
        m=c>=t; cov=float(m.mean())
        if not m.any(): risk=0.0
        else: risk=float((~y[m]).mean())
        if risk<=target_risk: feasible.append((cov,-float(t),float(t),risk))
    if not feasible: return {"threshold":reject_all,"coverage":0.0,"risk":0.0}
    cov,_,t,risk=max(feasible)
    return {"threshold":t,"coverage":cov,"risk":risk}

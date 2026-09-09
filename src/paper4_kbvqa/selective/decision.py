from __future__ import annotations
import numpy as np

def selective_decision(confidence: float, threshold: float) -> bool:
    """Return True when the system should answer, False when it should abstain."""
    return float(confidence) >= float(threshold)

def choose_threshold_for_target_risk(confidences, correct, target_risk: float=0.05) -> dict:
    """Validation-only threshold selection maximizing coverage subject to empirical risk <= target."""
    c=np.asarray(confidences,float); y=np.asarray(correct,bool)
    candidates=np.unique(np.concatenate(([0.0],c,[1.0])))
    feasible=[]
    for t in candidates:
        m=c>=t; cov=float(m.mean())
        if not m.any(): risk=0.0
        else: risk=float((~y[m]).mean())
        if risk<=target_risk: feasible.append((cov,-float(t),float(t),risk))
    if not feasible: return {"threshold":1.0,"coverage":0.0,"risk":0.0}
    cov,_,t,risk=max(feasible)
    return {"threshold":t,"coverage":cov,"risk":risk}

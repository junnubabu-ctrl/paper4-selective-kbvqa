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

def risk_coverage_curve(confidence, correct):
    p=np.asarray(confidence,float); y=np.asarray(correct,bool)
    order=np.argsort(-p); y=y[order]
    if len(y)==0: return np.array([]),np.array([])
    k=np.arange(1,len(y)+1); coverage=k/len(y); risk=np.cumsum(~y)/k
    return coverage.astype(float),risk.astype(float)

def aurc(confidence, correct):
    coverage,risk=risk_coverage_curve(confidence,correct)
    if len(coverage)<2: return float(risk[0]) if len(risk) else float('nan')
    return float(np.trapezoid(risk,coverage))

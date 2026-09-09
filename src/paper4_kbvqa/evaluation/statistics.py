from __future__ import annotations
import numpy as np

def paired_bootstrap_accuracy(a_correct,b_correct,n_boot=2000,seed=42):
    a=np.asarray(a_correct,float); b=np.asarray(b_correct,float)
    if len(a)!=len(b): raise ValueError("Paired samples must have equal length")
    rng=np.random.default_rng(seed); n=len(a)
    diffs=np.empty(n_boot)
    for i in range(n_boot):
        idx=rng.integers(0,n,n); diffs[i]=(a[idx]-b[idx]).mean()
    return {"difference":float((a-b).mean()),"ci95_low":float(np.quantile(diffs,.025)),"ci95_high":float(np.quantile(diffs,.975))}

def mcnemar_counts(a_correct,b_correct):
    a=np.asarray(a_correct,bool); b=np.asarray(b_correct,bool)
    return {"a_correct_b_wrong":int(np.sum(a & ~b)),"a_wrong_b_correct":int(np.sum(~a & b))}

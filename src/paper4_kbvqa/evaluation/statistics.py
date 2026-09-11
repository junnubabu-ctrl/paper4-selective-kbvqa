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


def paired_selective_bootstrap(a_conf,a_correct,a_threshold,b_conf,b_correct,b_threshold,n_boot=2000,seed=2026):
    """Matched-question intervals; undefined zero-coverage risks remain null."""
    from paper4_kbvqa.evaluation.metrics import aurc
    arrays=[np.asarray(x) for x in [a_conf,a_correct,b_conf,b_correct]]
    n=len(arrays[0])
    if not n or any(len(x)!=n for x in arrays): raise ValueError('Nonempty paired inputs required')
    if n_boot<1: raise ValueError('n_boot must be positive')
    def calculate(index):
        ac,ay,bc,by=[x[index] for x in arrays]
        am=ac>=a_threshold; bm=bc>=b_threshold
        return [float(am.mean()-bm.mean()),
                float((1-ay[am].mean())-(1-by[bm].mean())) if am.any() and bm.any() else float('nan'),
                aurc(ac,ay)-aurc(bc,by)]
    observed=calculate(np.arange(n)); rng=np.random.default_rng(seed)
    draws=np.asarray([calculate(rng.integers(0,n,n)) for _ in range(n_boot)])
    out={}
    for j,name in enumerate(['coverage_difference','selective_risk_difference','aurc_difference']):
        finite=draws[:,j][np.isfinite(draws[:,j])]
        out[name]={'difference':observed[j] if np.isfinite(observed[j]) else None,
                   'ci95_low':float(np.quantile(finite,.025)) if len(finite) else None,
                   'ci95_high':float(np.quantile(finite,.975)) if len(finite) else None,
                   'valid_bootstrap_replicates':len(finite),'total_bootstrap_replicates':n_boot}
    return out

from __future__ import annotations
import numpy as np

def _blocks(n, groups=None):
    if not n: raise ValueError('Nonempty paired inputs required')
    if groups is None: return [np.array([i]) for i in range(n)]
    if len(groups)!=n: raise ValueError('One image group required per question')
    buckets={}
    for i,group in enumerate(groups):
        if group is None: raise ValueError('Missing image group')
        buckets.setdefault(str(group),[]).append(i)
    if len(buckets)<2: raise ValueError('At least two image groups required for inference')
    return [np.asarray(v,dtype=int) for v in buckets.values()]


def _resample_indices(n,n_boot,seed,groups=None):
    if n_boot<1: raise ValueError('n_boot must be positive')
    blocks=_blocks(n,groups); rng=np.random.default_rng(seed)
    for _ in range(n_boot):
        if groups is None:
            yield rng.integers(0,n,n)
        else:
            yield np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))])


def paired_bootstrap_accuracy(a_correct,b_correct,n_boot=2000,seed=42,groups=None):
    """Question-weighted paired mean; optional resampling of entire image groups."""
    a=np.asarray(a_correct,float); b=np.asarray(b_correct,float)
    if len(a)!=len(b): raise ValueError("Paired samples must have equal length")
    if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('Finite scores required')
    diffs=np.asarray([(a[idx]-b[idx]).mean() for idx in _resample_indices(len(a),n_boot,seed,groups)])
    return {"difference":float((a-b).mean()),"ci95_low":float(np.quantile(diffs,.025)),"ci95_high":float(np.quantile(diffs,.975)),
            "resampling_unit":"image" if groups is not None else "question",
            "n_clusters":len(_blocks(len(a),groups)),"question_weighted":True}


def paired_cluster_swap_test(a_scores,b_scores,groups,n_permutations=19999,seed=2026):
    """Two-sided image-block label swap under within-image system exchangeability.

    Exact enumeration for <=16 image groups; otherwise a plus-one Monte Carlo
    p value. This is conditional on fixed predictions, not on calibration fitting.
    """
    import itertools
    a=np.asarray(a_scores,float); b=np.asarray(b_scores,float)
    if len(a)!=len(b): raise ValueError('Paired samples must have equal length')
    if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('Finite scores required')
    blocks=_blocks(len(a),groups)
    if n_permutations<1: raise ValueError('n_permutations must be positive')
    sums=np.asarray([(a[index]-b[index]).sum() for index in blocks])
    observed=abs(sums.sum()); extreme=0; g=len(blocks)
    if not np.any(sums):
        return {'p_value':1.0,'mode':'invariant_under_all_swaps','n_assignments':0,
                'n_image_groups':g,'seed':seed,
                'null_assumption':'System labels are exchangeable jointly within each independent image group.'}
    if g<=16:
        total=2**g
        for signs in itertools.product((-1,1),repeat=g):
            extreme+=abs(np.dot(signs,sums))>=observed-1e-12
        p=float(extreme/total); mode='exact_enumeration'
    else:
        total=n_permutations; rng=np.random.default_rng(seed)
        for start in range(0,total,256):
            signs=rng.integers(0,2,size=(min(256,total-start),g))*2-1
            extreme+=int(np.sum(np.abs(signs@sums)>=observed-1e-12))
        p=float((extreme+1)/(total+1)); mode='monte_carlo_plus_one'
    return {'p_value':p,'mode':mode,'n_assignments':total,'n_image_groups':g,'seed':seed,
            'null_assumption':'System labels are exchangeable jointly within each independent image group.'}

def mcnemar_counts(a_correct,b_correct):
    a=np.asarray(a_correct,bool); b=np.asarray(b_correct,bool)
    return {"a_correct_b_wrong":int(np.sum(a & ~b)),"a_wrong_b_correct":int(np.sum(~a & b))}


def paired_selective_bootstrap(a_conf,a_correct,a_threshold,b_conf,b_correct,b_threshold,n_boot=2000,seed=2026,groups=None):
    """Paired question/image-block intervals; zero-coverage risks remain null."""
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
    observed=calculate(np.arange(n))
    draws=np.asarray([calculate(index) for index in _resample_indices(n,n_boot,seed,groups)])
    out={}
    for j,name in enumerate(['coverage_difference','selective_risk_difference','aurc_difference']):
        finite=draws[:,j][np.isfinite(draws[:,j])]
        out[name]={'difference':observed[j] if np.isfinite(observed[j]) else None,
                   'ci95_low':float(np.quantile(finite,.025)) if len(finite) else None,
                   'ci95_high':float(np.quantile(finite,.975)) if len(finite) else None,
                   'valid_bootstrap_replicates':len(finite),'total_bootstrap_replicates':n_boot}
    return out

#!/usr/bin/env python
"""Generate statistics, CSV tables and 600-dpi figures from recorded metrics."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.stats import binomtest
from paper4_kbvqa.execution.study import write_json, read_rows
from paper4_kbvqa.evaluation.statistics import paired_bootstrap_accuracy, mcnemar_counts
from paper4_kbvqa.evaluation.metrics import risk_coverage_curve


def aligned(a,b):
    a={x['question_id']:x for x in a['per_question']}; b={x['question_id']:x for x in b['per_question']}
    if set(a)!=set(b): raise ValueError('Paired statistics require identical question IDs')
    keys=sorted(a)
    return [a[k] for k in keys],[b[k] for k in keys]


def holm(pvalues):
    order=np.argsort(pvalues); adjusted=[0.0]*len(order); previous=0.0
    for rank,i in enumerate(order):
        previous=max(previous,min(1.0,(len(order)-rank)*pvalues[i])); adjusted[int(i)]=previous
    return adjusted


def main():
    p=argparse.ArgumentParser(); p.add_argument('--results',required=True); args=p.parse_args()
    root=Path(args.results); out=root/'report'; out.mkdir(exist_ok=True)
    metrics={f.stem:json.loads(f.read_text()) for f in sorted((root/'metrics').glob('*.json'))
             if f.name!='aokvqa_B0_B5_summary.json'}
    table=[]
    for name,m in metrics.items():
        cal=m.get('calibration',{}); sel=m.get('selective',{})
        table.append({'experiment':name,'n':m['n_evaluated'],'soft_accuracy_percent':m['soft_accuracy_percent'],
                      'full_credit_accuracy_percent':m['full_credit_accuracy_percent'],
                      'coverage':sel.get('coverage',1),'selective_risk':sel.get('selective_risk'),
                      'ece':cal.get('ece'),'brier':cal.get('brier'),'aurc':cal.get('aurc')})
    with (out/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    pairs=[('B1','B0'),('B2','B1'),('B3','B2'),('B3','B0')]
    tests=[]
    for av,bv in pairs:
        a,b=aligned(metrics['aokvqa_val_'+av],metrics['aokvqa_val_'+bv])
        diff=paired_bootstrap_accuracy([x['soft_score'] for x in a],[x['soft_score'] for x in b],n_boot=2000,seed=2026)
        counts=mcnemar_counts([x['correct_full_credit'] for x in a],[x['correct_full_credit'] for x in b])
        discordant=sum(counts.values())
        pv=float(binomtest(counts['a_correct_b_wrong'],discordant,.5).pvalue) if discordant else 1.0
        tests.append({'a':av,'b':bv,'soft_accuracy_difference_fraction':diff,
                      'mcnemar_full_credit':counts,'p_value_exact':pv})
    for row,pv in zip(tests,holm([x['p_value_exact'] for x in tests])): row['p_value_holm']=pv
    efficiency={}; support={}
    for path in sorted((root/'predictions').glob('*.jsonl')):
        if path.name.endswith('.sessions.jsonl') or path.name.endswith('_calibrated.jsonl') or path.stem in {'aokvqa_val_B4','aokvqa_val_B5'}: continue
        rows=read_rows(path)
        if not rows: continue
        latency=np.asarray([x['latency_s'] for x in rows])
        sessions=path.with_suffix('.sessions.jsonl')
        hardware=[json.loads(x) for x in sessions.read_text().splitlines()] if sessions.exists() else []
        efficiency[path.stem]={'n':len(rows),'median_latency_s':float(np.median(latency)),
            'p95_latency_s':float(np.quantile(latency,.95)),
            'latency_note':'End-to-end per-question latency; includes retrieval and first-use model loading.',
            'session_peak_allocated_bytes':[x['session_peak_allocated_bytes'] for x in hardware]}
        evidence_counts={}
        for row in rows:
            for e in row.get('retrieved_evidence',[]): evidence_counts[e['source']]=evidence_counts.get(e['source'],0)+1
        support[path.stem]={'cited_prediction_fraction':sum(bool(x.get('supporting_evidence_ids')) for x in rows)/len(rows),
            'empty_retrieval_fraction':sum(not x.get('retrieved_evidence') for x in rows)/len(rows),
            'retrieved_items_by_source':evidence_counts,
            'note':'Citation/source diagnostics, not independently measured grounding accuracy.'}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4))
    for v in ['B0','B1','B2','B3','B4']:
        rows=metrics['aokvqa_val_'+v]['per_question']
        coverage,risk=risk_coverage_curve([x['confidence'] for x in rows],[x['correct_full_credit'] for x in rows])
        ax.plot(coverage,risk,label=v)
    ax.set(xlabel='Coverage',ylabel='Full-credit error rate',xlim=(0,1),ylim=(0,1)); ax.legend(); fig.tight_layout()
    fig.savefig(out/'risk_coverage.png',dpi=600); fig.savefig(out/'risk_coverage.pdf'); plt.close(fig)
    report={'statistics':tests,'bootstrap_replicates':2000,'seed':2026,'efficiency':efficiency,'evidence_diagnostics':support,
            'scope':'A-OKVQA; observed results only',
            'limitations':['Calibration target risk is empirical, not a certified risk bound.',
                          'Corruption injections are controlled perturbations, not natural contradiction labels.',
                          'Independent evidence support, external baselines and official OK-VQA evaluation remain necessary.']}
    write_json(out/'study_report.json',report)
    files={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
           for folder in ['predictions','metrics','policies','manifests','report']
           for f in (root/folder).rglob('*') if f.is_file()}
    write_json(root/'artifact_checksums.json',files)
    print('Generated',out)

if __name__=='__main__': main()

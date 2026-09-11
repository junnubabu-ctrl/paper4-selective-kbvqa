import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import pytest
from paper4_kbvqa.execution.study import freeze_json, validate_predictions, CorruptionTransform
from paper4_kbvqa.execution.variants import MatchedVariantRunner
from paper4_kbvqa.data.manifest import VQASample
from paper4_kbvqa.selective.decision import choose_threshold_for_target_risk
from paper4_kbvqa.generation.qwen_vl import Qwen25VLGenerator
from paper4_kbvqa.types import Evidence
ROOT=Path(__file__).resolve().parents[1]

class Generator:
    def generate(self,**kw): return {'answer':'cat','raw_confidence':.8,'supporting_evidence_ids':[]}


def test_resume_uses_predictions_not_stale_checkpoint(tmp_path):
    runner=MatchedVariantRunner(variant='B0',generator=Generator())
    samples=[VQASample('a','x','What?')]
    out=tmp_path/'p.jsonl'; cp=tmp_path/'cp.json'
    runner.run_manifest(samples,output_jsonl=out,checkpoint_json=cp)
    cp.write_text('{"completed_question_ids":[]}')
    with out.open('ab') as f: f.write(b'{"interrupted":')
    runner.run_manifest(samples,output_jsonl=out,checkpoint_json=cp)
    assert len(out.read_text().splitlines())==1


def test_corrupt_complete_prediction_rejected(tmp_path):
    r=MatchedVariantRunner(variant='B0',generator=Generator()); s=[VQASample('a','x','?')]
    out=tmp_path/'p'; cp=tmp_path/'cp'; r.run_manifest(s,output_jsonl=out,checkpoint_json=cp)
    out.write_text(out.read_text().replace('cat','dog'))
    with pytest.raises(ValueError): r.run_manifest(s,output_jsonl=out,checkpoint_json=cp)


def test_failed_inference_stops_without_false_success(tmp_path):
    class Broken:
        def generate(self,**kw): raise RuntimeError('GPU failed')
    r=MatchedVariantRunner(variant='B0',generator=Broken())
    with pytest.raises(RuntimeError): r.run_manifest([VQASample('a','x','?')],output_jsonl=tmp_path/'p',checkpoint_json=tmp_path/'cp')


def test_freeze_refuses_changed_configuration(tmp_path):
    p=tmp_path/'id'; freeze_json(p,{'seed':1}); freeze_json(p,{'seed':1})
    with pytest.raises(ValueError): freeze_json(p,{'seed':2})


def test_reject_all_handles_confidence_one():
    p=choose_threshold_for_target_risk([1.,1.],[0,0],.05)
    assert p['threshold']>1 and p['coverage']==0


def test_parse_failure_does_not_invent_citations():
    assert Qwen25VLGenerator._parse_output('cat',['e1'])[1]==[]


def test_prediction_gate_rejects_extra_ids(tmp_path):
    m=tmp_path/'m'; p=tmp_path/'p'
    m.write_text('{"question_id":"1"}\n')
    p.write_text('{"question_id":"2","answer":"x","raw_confidence":0.5}\n')
    with pytest.raises(ValueError): validate_predictions(m,p)


def test_corruption_precedes_generation_and_is_repeatable():
    ev=[Evidence('e1','A cat is an animal.','wikipedia')]
    corrupt=CorruptionTransform('contradiction:1',2026)
    first=corrupt(ev,'1'); assert first==corrupt(ev,'1')
    assert first[0].metadata['synthetic'] and first[1]==ev[0]


def test_official_difficult_filter_in_cli(tmp_path):
    spec=importlib.util.spec_from_file_location('evaluate_cli',ROOT/'scripts/evaluate.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    m=tmp_path/'m'; p=tmp_path/'p'
    m.write_text('\n'.join(json.dumps({'question_id':str(i),'image_path':'x','question':'?',
        'answers':['cat']*3,'metadata':{'difficult_direct_answer':i==2}}) for i in [1,2]))
    p.write_text('\n'.join(json.dumps({'question_id':str(i),'answer':'cat' if i==1 else 'dog','raw_confidence':.5}) for i in [1,2]))
    result=module.evaluate_rows(str(m),str(p),dataset='aokvqa',confidence_field='raw_confidence',threshold=None)
    assert result['n_evaluated']==1 and result['soft_accuracy_percent']==100


def test_all_scripts_and_notebook_cells_compile():
    for f in (ROOT/'scripts').glob('*.py'): compile(f.read_text(),str(f),'exec')
    nb=json.loads((ROOT/'notebooks/Paper4_Final_Free_GPU_Execution.ipynb').read_text())
    for c in nb['cells']:
        if c['cell_type']=='code': compile(''.join(c['source']),'notebook','exec')


def test_plan_requires_no_gpu():
    output=subprocess.check_output([sys.executable,str(ROOT/'scripts/run_full_study.py'),'--plan'],text=True)
    plan=json.loads(output)
    assert len(plan['experiments'])==38
    assert all(x['variant'] in {'B0','B1','B2','B3'} for x in plan['experiments'])


def test_cpu_postprocessing_end_to_end(tmp_path):
    """Synthetic fixture validates plumbing only; all files remain in tmp_path."""
    cal=tmp_path/'cal.jsonl'; val=tmp_path/'val.jsonl'; cp=tmp_path/'cal_pred.jsonl'
    def samples(prefix):
        return [{'question_id':f'{prefix}{i}','image_path':'x','question':'?',
                 'answers':['cat']*3,'metadata':{'difficult_direct_answer':False}} for i in range(8)]
    def preds(prefix):
        return [{'question_id':f'{prefix}{i}','answer':'cat' if i%2 else 'dog',
                 'raw_confidence':.9 if i%2 else .1,'latency_s':.01,'retrieved_evidence':[],
                 'supporting_evidence_ids':[]} for i in range(8)]
    def write(path,rows):
        path.parent.mkdir(parents=True,exist_ok=True); path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    write(cal,samples('c')); write(val,samples('v')); write(cp,preds('c'))
    def call(script,*args):
        subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],check=True,capture_output=True,text=True)
    policy=tmp_path/'policies/p.json'
    call('fit_selective_policy.py','--manifest',cal,'--predictions',cp,'--out',policy)
    for v in ['B0','B1','B2','B3']:
        pred=tmp_path/f'predictions/aokvqa_val_{v}.jsonl'; write(pred,preds('v'))
        call('evaluate.py','--manifest',val,'--predictions',pred,'--dataset','aokvqa','--out',tmp_path/f'metrics/aokvqa_val_{v}.json')
    applied=tmp_path/'predictions/applied_calibrated.jsonl'
    call('apply_selective_policy.py','--policy',policy,'--predictions',pred,'--out',applied)
    for v in ['B4','B5']:
        extra=['--threshold',json.loads(policy.read_text())['threshold']] if v=='B5' else []
        call('evaluate.py','--manifest',val,'--predictions',applied,'--dataset','aokvqa',
             '--confidence-field','calibrated_confidence',*extra,'--out',tmp_path/f'metrics/aokvqa_val_{v}.json')
    args=[]
    for v in ['B0','B1','B2','B3','B4','B5']: args+=['--'+v.lower(),tmp_path/f'metrics/aokvqa_val_{v}.json']
    summary=tmp_path/'metrics/aokvqa_B0_B5_summary.json'
    call('summarize_variants.py',*args,'--out',summary)
    assert json.loads(summary.read_text())['B5']['risk']==0
    call('report_study.py','--results',tmp_path)
    report=json.loads((tmp_path/'report/study_report.json').read_text())
    assert len(report['statistics'])==4
    assert (tmp_path/'report/risk_coverage.png').stat().st_size>0
    assert all(x['p_value_holm']==1 for x in report['statistics'])
    # A calibrator cannot be applied to its own labelled fitting IDs as held-out data.
    with pytest.raises(subprocess.CalledProcessError):
        call('apply_selective_policy.py','--policy',policy,'--predictions',cp,'--out',tmp_path/'bad.jsonl')


def test_entity_indexed_sources_receive_entities():
    from paper4_kbvqa.knowledge.structured import StructuredKnowledgeProvider
    class Source:
        name='conceptnet'
        def __init__(self): self.queries=[]
        def retrieve(self,query,limit): self.queries.append(query); return []
    source=Source()
    StructuredKnowledgeProvider([source]).retrieve_for_question('What is this animal doing?',['cat'],10)
    assert source.queries==['cat']

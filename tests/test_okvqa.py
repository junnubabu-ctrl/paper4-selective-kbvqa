from dataclasses import replace
from pathlib import Path
import json
import subprocess
import sys
import pytest
from paper4_kbvqa.data.manifest import VQASample,load_jsonl
from paper4_kbvqa.evaluation.okvqa import score_predictions
from paper4_kbvqa.evaluation.statistics import paired_selective_bootstrap
ROOT=Path(__file__).resolve().parents[1]


def sample(answers):
    annotation={'question_id':1,'question_type':'other','answer_type':'other',
                'answers':[{'answer':v,'answer_id':i+1} for i,v in enumerate(answers)]}
    return VQASample('1','x','?',answers=tuple(answers),metadata={'official_annotation':annotation})


def test_okvqa_leave_one_out_not_naive_consensus():
    s=sample(['cat']*3+['dog']*7)
    score=score_predictions([s],{'1':{'answer':'cat'}})['1']
    assert score==pytest.approx(.9)
    assert score_predictions([sample(['cat']*10)],{'1':{'answer':'cat'}})['1']==1


def test_okvqa_upstream_normalization():
    s=sample(['two']*4+['three']*6)
    assert score_predictions([s],{'1':{'answer':'The two!'}})['1']==1
    assert s.metadata['official_annotation']['answers'][0]['answer']=='two'


def test_official_annotations_required():
    with pytest.raises(ValueError): score_predictions([VQASample('1','x','?')],{'1':{'answer':'cat'}})


def test_manifest_preserves_annotation_and_coco2014_filename(tmp_path):
    q=tmp_path/'questions.json'; a=tmp_path/'annotations.json'; out=tmp_path/'manifest.jsonl'
    q.write_text(json.dumps({'questions':[{'question_id':1,'image_id':42,'question':'?'}]}))
    annotation=sample(['cat']*10).metadata['official_annotation']; annotation['image_id']=42
    a.write_text(json.dumps({'annotations':[annotation]}))
    subprocess.run([sys.executable,str(ROOT/'scripts/build_manifest.py'),'okvqa','--questions',str(q),
        '--annotations',str(a),'--coco-dir',str(tmp_path),'--out',str(out)],check=True)
    rows=load_jsonl(out)
    assert rows[0].image_path.endswith('val2014/COCO_val2014_000000000042.jpg')
    assert rows[0].metadata['official_annotation']==annotation


def test_selective_intervals_handle_zero_coverage():
    output=paired_selective_bootstrap([.5,.5],[0,1],1.1,[.5,.5],[0,1],0,n_boot=20)
    assert output['selective_risk_difference']['difference'] is None
    assert output['selective_risk_difference']['valid_bootstrap_replicates']==0
    assert output['coverage_difference']['difference']==-1


def test_selective_intervals_identical_predictions():
    output=paired_selective_bootstrap([.2,.8],[0,1],.5,[.2,.8],[0,1],.5,n_boot=20)
    assert output['aurc_difference']['difference']==0
    assert output['selective_risk_difference']['ci95_high']==0


def test_two_dataset_plans_and_three_corruption_seeds():
    for dataset in ['aokvqa','okvqa']:
        plan=json.loads(subprocess.check_output([sys.executable,str(ROOT/'scripts/run_full_study.py'),
            '--dataset',dataset,'--plan'],text=True))
        assert plan['dataset']==dataset
        assert len(plan['experiments'])==87
        assert {x['corruption_seed'] for x in plan['experiments'] if x.get('corruption')}=={42,123,2026}

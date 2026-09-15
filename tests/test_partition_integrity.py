from paper4_kbvqa.data.partitions import partition_groups,image_group
from paper4_kbvqa.selective.decision import choose_threshold_for_target_risk
from paper4_kbvqa.generation.qwen_vl import Qwen25VLGenerator
import pytest


def test_same_image_stays_in_one_partition():
    rows=[{'question_id':str(i),'metadata':{'image_id':i//2}} for i in range(20)]
    a,b=partition_groups(rows)
    assert {image_group(x) for x in a}.isdisjoint({image_group(x) for x in b})
    assert len(a)+len(b)==len(rows)
    assert partition_groups(list(reversed(rows)))[0][::-1]==a


def test_single_cluster_cannot_be_calibrated_independently():
    with pytest.raises(ValueError):partition_groups([{'metadata':{'image_id':1}}]*5)


def test_reject_all_has_undefined_risk_even_at_confidence_one():
    p=choose_threshold_for_target_risk([1.,1.],[0,0],0.)
    assert p['coverage']==0 and p['risk'] is None and p['threshold']>1


def test_unstructured_answer_does_not_invent_citations():
    assert Qwen25VLGenerator._parse_output('cat',['e1','e2'])==('cat',[])

@pytest.mark.parametrize('change',['question','image'])
def test_resume_rejects_changed_input(tmp_path,change):
    from paper4_kbvqa.execution.variants import MatchedVariantRunner
    from paper4_kbvqa.data.manifest import VQASample
    from dataclasses import replace
    class G:
        def generate(self,**kwargs):return {'answer':'cat','raw_confidence':.8}
    image=tmp_path/'image';image.write_bytes(b'first')
    sample=VQASample('q',str(image),'what?',visual_entities=('cat',))
    runner=MatchedVariantRunner(variant='B0',generator=G())
    kwargs={'output_jsonl':tmp_path/'out','checkpoint_json':tmp_path/'ckpt'}
    runner.run_manifest([sample],**kwargs)
    if change=='question':sample=replace(sample,question='where?')
    else:image.write_bytes(b'second')
    with pytest.raises(ValueError,match='input changed'):runner.run_manifest([sample],**kwargs)

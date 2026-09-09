from paper4_kbvqa.utils.checkpoint import CheckpointManager, CheckpointState

def test_checkpoint_roundtrip(tmp_path):
    p=tmp_path/'checkpoint.json'; m=CheckpointManager(p)
    s=CheckpointState(run_id='r1',completed_question_ids=['q1'])
    m.save(s); x=m.load()
    assert x.run_id=='r1' and x.completed_question_ids==['q1'] and x.updated_at

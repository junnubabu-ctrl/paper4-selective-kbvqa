import pytest
from paper4_kbvqa.data.leakage import assert_prompt_no_ground_truth, check_split_overlap

def test_ground_truth_rejected_from_prompt():
    with pytest.raises(ValueError): assert_prompt_no_ground_truth({"question":"q","answer":"secret"})

def test_duplicate_question_id_detected():
    r=check_split_overlap([{"question_id":1,"image_id":1,"question":"a"}],[],[{"question_id":1,"image_id":9,"question":"z"}])
    assert not r.clean and "1" in r.duplicate_ids

from paper4_kbvqa.data.leakage import check_split_overlap

def test_split_overlap_clean():
    r=check_split_overlap([{"question_id":1,"image_id":1,"question":"a"}],[{"question_id":2,"image_id":2,"question":"b"}],[{"question_id":3,"image_id":3,"question":"c"}])
    assert r.clean

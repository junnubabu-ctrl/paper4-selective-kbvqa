from paper4_kbvqa.evaluation.aokvqa import direct_answer_score, direct_answer_accuracy

def test_direct_answer_score_matches_official_formula():
    assert direct_answer_score("cab", ["cab","cab","cab","taxi"]) == 1.0
    assert direct_answer_score("cab", ["cab","cab","taxi","taxi"]) == 2/3

def test_direct_answer_accuracy_skip_difficult():
    ds=[
        {"question_id":"1","direct_answers":["a","a","a"],"difficult_direct_answer":False},
        {"question_id":"2","direct_answers":["b","b","b"],"difficult_direct_answer":True},
    ]
    assert direct_answer_accuracy({"1":"a","2":"x"}, ds) == 100.0

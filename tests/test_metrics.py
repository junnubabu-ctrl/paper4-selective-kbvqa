from paper4_kbvqa.evaluation.metrics import expected_calibration_error,brier_score,selective_metrics,aurc

def test_metrics_ranges():
    p=[.1,.4,.8,.9]; y=[0,0,1,1]
    assert 0<=expected_calibration_error(p,y)<=1
    assert 0<=brier_score(p,y)<=1
    assert 0<=aurc(p,y)<=1
    assert selective_metrics(p,y,.8)["coverage"]==.5

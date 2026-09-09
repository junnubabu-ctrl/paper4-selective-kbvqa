from paper4_kbvqa.selective.decision import selective_decision, choose_threshold_for_target_risk

def test_decision(): assert selective_decision(.8,.7) and not selective_decision(.6,.7)
def test_threshold_validation_logic():
    r=choose_threshold_for_target_risk([.1,.4,.8,.9],[0,0,1,1],0.0); assert r["coverage"]==.5

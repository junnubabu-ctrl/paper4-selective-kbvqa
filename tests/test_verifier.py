from paper4_kbvqa.types import Evidence
from paper4_kbvqa.verification.verifier import EvidenceVerifier

def test_verifier_bounds():
    x=EvidenceVerifier().verify("Paris",[Evidence("e","Eiffel Tower is in Paris","wiki",retrieval_score=.8)],.9)
    assert 0 <= x["verification_score"] <= 1

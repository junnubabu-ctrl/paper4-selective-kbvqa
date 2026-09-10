from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "src"))

import json
from pathlib import Path
import numpy as np

from paper4_kbvqa.types import Evidence
from paper4_kbvqa.retrieval.tfidf import TfidfRetriever
from paper4_kbvqa.filtering.relevance import RelevanceFilter, RelevanceWeights
from paper4_kbvqa.verification.verifier import EvidenceVerifier
from paper4_kbvqa.calibration.calibrators import PlattCalibrator
from paper4_kbvqa.evaluation.metrics import expected_calibration_error, brier_score, aurc
from paper4_kbvqa.selective.decision import choose_threshold_for_target_risk

question = "In which city is the Eiffel Tower located?"
base_evidence = [
    Evidence(
        evidence_id="e1",
        text="The Eiffel Tower is a wrought-iron tower in Paris, France.",
        source="wikipedia",
        uri="https://en.wikipedia.org/wiki/Eiffel_Tower",
        entity_ids=("Eiffel Tower", "Paris"),
    ),
    Evidence(
        evidence_id="e2",
        text="The Colosseum is an amphitheatre in Rome, Italy.",
        source="wikipedia",
        uri="https://en.wikipedia.org/wiki/Colosseum",
        entity_ids=("Colosseum", "Rome"),
    ),
    Evidence(
        evidence_id="e3",
        text="Paris is the capital and most populous city of France.",
        source="wikidata",
        uri="https://www.wikidata.org/wiki/Q90",
        entity_ids=("Paris", "France"),
    ),
]

retriever = TfidfRetriever([e.text for e in base_evidence])
ranked = []
for idx, score in retriever.search(question, top_k=3):
    e = base_evidence[idx]
    ranked.append(
        Evidence(
            evidence_id=e.evidence_id,
            text=e.text,
            source=e.source,
            uri=e.uri,
            entity_ids=e.entity_ids,
            retrieval_score=score,
            metadata=e.metadata,
        )
    )

flt = RelevanceFilter(
    RelevanceWeights(
        semantic=0.35,
        question=0.25,
        entity=0.15,
        visual=0.10,
        source=0.10,
        redundancy=0.05,
    ),
    min_score=0.05,
)
selected, filter_details = flt.select(
    ranked,
    question,
    ["Eiffel Tower", "Paris"],
    top_k=2,
)

verification = EvidenceVerifier().verify(
    "Paris",
    selected,
    visual_consistency=0.8,
)

raw = np.array([0.20, 0.35, 0.45, 0.58, 0.65, 0.72, 0.80, 0.90])
y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
cal = PlattCalibrator().fit(raw, y).predict(raw)
policy = choose_threshold_for_target_risk(cal, y, target_risk=0.20)

out = {
    "status": "SYNTHETIC_SMOKE_ONLY",
    "retrieved": [e.evidence_id for e in ranked],
    "selected": [e.evidence_id for e in selected],
    "filter_details": filter_details,
    "verification": verification,
    "calibration": {
        "ece": expected_calibration_error(cal, y),
        "brier": brier_score(cal, y),
    },
    "selective": {
        "threshold": policy["threshold"],
        "coverage": policy["coverage"],
        "risk": policy["risk"],
    },
    "aurc": aurc(cal, y),
}

Path("results/metrics").mkdir(parents=True, exist_ok=True)
Path("results/logs").mkdir(parents=True, exist_ok=True)
Path("results/metrics/dev_smoke.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
Path("results/logs/dev_smoke_latest.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))

# EXPERIMENT PLAN — LOCK CANDIDATE v0.1

## Research question
Can explicit multi-source evidence provenance and contradiction-aware verification, followed by validation-only confidence calibration and risk-controlled selective answering, reduce unsupported KB-VQA answers while retaining useful coverage?

## Datasets
1. **A-OKVQA** — primary selective-evaluation benchmark because it explicitly requires outside/world knowledge and has an official evaluation package.
2. **OK-VQA** — complementary established external-knowledge VQA benchmark.

Official splits and official metrics must be preserved. Dataset counts/licensing will be recorded from official sources when materialized; do not hard-code remembered counts into results.

## Proposed baseline progression
- B0: VLM, no external knowledge.
- B1: VLM + raw multi-source retrieved evidence.
- B2: B1 + relevance filtering.
- B3: B2 + evidence/provenance verifier.
- B4: B3 + validation-only calibration.
- B5: B4 + selective answering/abstention.

## Core ablations (P0/P1)
- no retrieval; no filtering; no verifier; no calibration; no abstention
- source: Wikipedia / Wikidata / ConceptNet / combined
- top-k: 1, 3, 5, 10
- verifier: semantic-only vs provenance+contradiction-aware
- confidence: raw generation vs verifier-fused vs calibrated
- threshold / target-risk sensitivity

## Leakage protocol
- Ground-truth answers never enter query construction, retrieval, filtering, generation, verification, or test-time calibration.
- Calibration model and operating threshold selected on validation/development only.
- Final test is one-way evaluation; no threshold or weight changes after inspecting test outcomes.

## Primary metrics
Official VQA metric per dataset, plus coverage, selective accuracy, selective risk, ECE, Brier score, risk-coverage curve, AURC. Retrieval Recall@k/MRR only when genuine gold evidence exists; otherwise use explicitly labeled relevance analysis.

## Statistics
For paired model predictions: bootstrap 95% CI for paired accuracy difference and McNemar discordant-pair analysis. Significance claims require actual p-values; none are pre-written.

## Seeds
42, 123, 2026 for stochastic components where repeats are scientifically meaningful. Deterministic VLM outputs may be cached rather than rerun wastefully.

## Development gate
Run 50–200 samples end-to-end first. A full experiment cannot begin until: no-leakage tests pass, prompt schema is validated, cache/resume works, official evaluator works, and one complete baseline/proposed prediction record is produced.

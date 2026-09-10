# EXPERIMENT PLAN — LOCK CANDIDATE v0.2

## Research question
Can explicit multi-source evidence provenance and contradiction-aware verification, followed by validation-only confidence calibration and target-risk selective answering, reduce unsupported KB-VQA answers under clean and deliberately corrupted external evidence while retaining useful answer coverage?

## Datasets
1. **A-OKVQA** — primary benchmark because it requires outside/world knowledge and provides an official evaluation package.
2. **OK-VQA** — complementary established external-knowledge VQA benchmark.

Official splits and official metrics must be preserved. Dataset counts and licensing must be recorded from official sources when materialized; do not hard-code remembered counts into result claims.

## Model policy
Use one fixed open VLM family/revision across B0–B5 wherever technically possible so improvements are attributable to the reliability pipeline rather than model swapping. Record revision, precision, quantization, prompt version, decoding parameters, and hardware.

## Baseline progression
- **B0**: VLM, no external knowledge.
- **B1**: VLM + raw multi-source retrieved evidence.
- **B2**: B1 + relevance filtering.
- **B3**: B2 + evidence/provenance verifier with contradiction signal.
- **B4**: B3 + validation-only confidence calibration.
- **B5**: B4 + target-risk selective answering/abstention.

## Mandatory external comparison families
Paper-4 must not compare only against ordinary KB-VQA models.
- **Selective-VQA reference**: Learning From Your Peers (CVPR 2023) or a protocol-matched learned selection baseline where feasible.
- **Black-box selective reference**: neighborhood-consistency/uncertainty style selection (CVPR 2024) conceptually or experimentally where feasible.
- **Evidence-rescue reference**: ReCoVERR-style verification/rescue comparison where implementation and protocol allow.
- **Retrieval-strength reference**: a strong recent RAG/KB-VQA retrieval baseline such as ReAuSE; for datasets outside OK-VQA/A-OKVQA, ReAG is contextual rather than directly protocol-matched.

Do not report cross-paper numbers as controlled superiority unless model, split, metric, knowledge corpus, and inference protocol are comparable.

## Core ablations (P0/P1)
- no retrieval; no filtering; no verifier; no calibration; no abstention
- source: Wikipedia / Wikidata / ConceptNet / combined
- top-k: 1, 3, 5, 10
- verifier: semantic-only vs provenance-aware vs provenance+contradiction-aware
- confidence: raw generation vs verifier-fused vs calibrated
- threshold / target-risk sensitivity
- answer generation with evidence IDs removed vs retained

## Evidence-corruption stress test — P0 novelty test
Construct controlled perturbations from the retrieved evidence **without using ground-truth answers**:
1. **irrelevant injection** — add semantically plausible but question-irrelevant passages;
2. **source dropout** — remove one evidence source at a time;
3. **contradictory injection** — add evidence that conflicts with another retrieved statement using a predefined transformation/provenance protocol;
4. **ranking corruption** — move lower-ranked evidence into top-k positions;
5. **evidence scarcity** — restrict k to 0/1/3.

For each corruption level, report answer accuracy, coverage, selective risk, AURC, calibration error, verifier score separation, and abstention behavior. The corruption generator and random seed must be saved so every perturbation is reproducible.

## Provenance/evidence-support evaluation
Because answer accuracy alone cannot validate the proposed contribution, additionally measure where feasible:
- fraction of answered predictions with at least one cited evidence item;
- answer–evidence support score using the frozen verifier protocol;
- contradiction-detection AUROC/AUPRC when synthetic contradiction labels are available;
- source attribution distribution and source-level failure rates;
- verifier false-positive and false-negative cases.

If no independently annotated gold evidence exists, do **not** label these metrics as retrieval Recall@k or factual-grounding accuracy.

## Leakage protocol
- Ground-truth answers never enter query construction, retrieval, filtering, generation, verification, corruption construction, or test-time calibration.
- Calibration model and operating threshold are selected on validation/development data only.
- Corruption severity and target-risk operating points are frozen before final test evaluation.
- Final test is one-way evaluation; no threshold, verifier weight, prompt, source mixture, or corruption definition changes after inspecting test outcomes.

## Primary metrics
Official VQA metric per dataset, plus coverage, selective accuracy, selective risk, C@target-risk (including C@1% and/or C@5% where statistically meaningful), ECE, Brier score, risk-coverage curve, and AURC. Retrieval Recall@k/MRR only when genuine gold evidence exists; otherwise use explicitly labeled relevance/support analysis.

## Statistics
For paired model predictions: bootstrap 95% CI for paired accuracy/risk differences and McNemar discordant-pair analysis. For corruption curves, use paired bootstrap over question IDs at matched corruption levels. Report effect sizes and multiplicity handling when many pairwise hypotheses are tested. Significance claims require actually calculated p-values; none are pre-written.

## Seeds
42, 123, 2026 for stochastic components where repeats are scientifically meaningful. Deterministic VLM outputs may be cached rather than rerun wastefully. Corruption seeds must be persisted with each run.

## Development gate
Run 50–200 samples end-to-end first. A full experiment cannot begin until: no-leakage tests pass, prompt schema is validated, cache/resume works, official evaluator works, one complete baseline/proposed prediction record is produced, and the corruption generator passes reproducibility tests.

## Promotion gate to manuscript results
A result may enter the manuscript only when its experiment record contains: git commit, config, dataset split, sample count, model revision, seed, hardware, cached predictions, evaluator output, and status=`COMPLETED`. Partial/development outputs remain clearly labeled and are excluded from headline claims.

# EviTrust-VQA — LLM Execution Requirements

## Required runtime architecture

1. Visual grounding / answer generator: `Qwen/Qwen2.5-VL-3B-Instruct`.
2. External knowledge: Wikipedia + Wikidata + ConceptNet with immutable evidence IDs, source, URI, cache state and retrieval score.
3. Relevance filtering before generation. The transparent filter is the primary auditable baseline; a sentence-transformer reranker is available as a stronger retrieval-quality ablation.
4. Separate evidence critic: `Qwen/Qwen2.5-1.5B-Instruct`.
   - Inputs: question, candidate answer, visual entities, evidence, source/URI and cited evidence IDs.
   - Forbidden input: gold/reference answer.
   - Outputs: normalized likelihoods for `SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT`.
5. Cross-family verifier control: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, used only as an NLI ablation to test same-family critic dependence.
6. Provenance decomposition:
   - citation validity,
   - source/URI traceability,
   - source diversity.
   The primary composite is fixed at 0.60/0.30/0.10 respectively.
7. Primary reliability fusion: generation likelihood + critic support + provenance - contradiction penalty - insufficiency penalty.
   A standalone visual-consistency scalar is excluded from the primary fusion until an independently validated estimator is available.
8. Validation-only calibration using Platt, isotonic and temperature methods; freeze the selected mapping before final test.
9. Selective answering using validation-selected thresholds and report risk-coverage/AURC.
10. Checkpoint/resume with completed question IDs.
11. Audit model IDs/revisions, prompt versions, evidence IDs, latency and SHA-256 record hashes.

## Baselines

- B0 — MLLM only.
- B1 — MLLM + raw multi-source retrieval.
- B2 — B1 + relevance filtering.
- B3 — B2 + separate LLM evidence critic.
- B4 — B3 + validation-only calibration.
- B5 — B4 + risk-controlled selective answering.

## Mandatory ablations

- no retrieval;
- raw vs filtered evidence;
- transparent vs sentence-transformer reranking;
- heuristic verifier vs Qwen critic vs cross-family DeBERTa NLI verifier;
- support-only vs support+contradiction+insufficiency;
- citation-validity-only vs full provenance composite;
- individual sources vs combined;
- top-k = 1, 3, 5, 10;
- generation likelihood vs critic-fused vs calibrated confidence;
- no calibration;
- no abstention;
- target-risk sensitivity.

## Robustness

Evaluate irrelevant evidence, contradictory evidence, source dropout, evidence scarcity and ranking corruption at controlled severities. Save corruption seed and perturbed evidence IDs. A manually audited corruption subset should be used to verify that synthetic contradiction and irrelevance interventions are semantically valid before they support paper claims.

## Statistics

Use matched question IDs. Report 95% confidence intervals, 10,000 paired bootstrap resamples where appropriate, McNemar analysis, effect sizes and Holm correction for prespecified multiple comparisons. Report verifier discrimination and contradiction sensitivity separately from downstream VQA accuracy.

## Efficiency

Report generator, retrieval/reranking, verifier and total latency; hardware; quantization; peak VRAM; exact sample count; and throughput. This is mandatory because B3-B5 introduce an additional verification model.

## Publication rule

All manuscript numerical claims must be generated from saved evaluation artifacts. Development-subset metrics must retain exact split and N and must never be promoted to final benchmark claims.

# EviTrust-VQA — LLM Execution Requirements

## Required runtime architecture

1. Visual grounding / answer generator: `Qwen/Qwen2.5-VL-3B-Instruct`.
2. External knowledge: Wikipedia + Wikidata + ConceptNet with evidence IDs, source, URI, cache state and retrieval score.
3. Relevance filtering before generation.
4. Independent evidence critic: `Qwen/Qwen2.5-1.5B-Instruct`.
   - Inputs: question, candidate answer, visual entities, evidence, source/URI and cited evidence IDs.
   - Forbidden input: gold/reference answer.
   - Outputs: normalized likelihoods for `SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT`.
5. Reliability fusion: generation likelihood + critic support + provenance + visual consistency - contradiction penalty.
6. Validation-only calibration using Platt, isotonic and temperature methods; freeze the selected mapping before final test.
7. Selective answering using validation-selected thresholds and report risk-coverage/AURC.
8. Checkpoint/resume with completed question IDs.
9. Audit model IDs/revisions, prompt versions, evidence IDs, latency and SHA-256 record hashes.

## Baselines

- B0 — MLLM only.
- B1 — MLLM + raw multi-source retrieval.
- B2 — B1 + relevance filtering.
- B3 — B2 + independent LLM critic.
- B4 — B3 + validation-only calibration.
- B5 — B4 + risk-controlled selective answering.

## Mandatory ablations

No retrieval; raw vs filtered evidence; heuristic verifier vs LLM critic; semantic-only vs provenance+contradiction-aware critic; individual sources vs combined; top-k 1/3/5/10; raw vs critic-fused vs calibrated confidence; no calibration; no abstention; target-risk sensitivity.

## Robustness

Evaluate irrelevant evidence, contradictory evidence, source dropout, evidence scarcity and ranking corruption at controlled severities. Save corruption seed and perturbed evidence IDs.

## Statistics

Use matched question IDs. Report 95% confidence intervals, 10,000 paired bootstrap resamples where appropriate, McNemar analysis, effect sizes and Holm correction for multiple comparisons.

## Publication rule

All manuscript numerical claims must be generated from saved evaluation artifacts. Development-subset metrics must retain exact split and N and must never be promoted to final benchmark claims.

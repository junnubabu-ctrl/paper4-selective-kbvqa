## Two-dataset execution update

The pipeline now supports official-format A-OKVQA and OK-VQA v1.1 scoring, COCO 2014/2017 images, 87 extended configurations per dataset, three corruption seeds, evidence-ID-removal controls and paired selective intervals. CI installs reporting dependencies. These are implementation changes; GPU results remain absent.

## September 11 execution update

A resumable A-OKVQA study runner and persistent Colab workflow are implemented. CPU regression and fixture-based postprocessing checks pass; GPU inference and live APIs are unverified. No scientific results were generated. See `docs/AUTOMATED_STUDY.md` for exact scope and outstanding requirements. Historical implementation notes below do not constitute benchmark evidence.

# IMPLEMENTATION STATUS — 2026-09-09

## COMPLETED
- dedicated Paper-4 repository scaffold prepared locally
- CPU environment audit
- leakage-safe inference manifest with label-isolated `inference_view`
- modular Wikipedia/Wikidata/ConceptNet providers + combined-provider deduplication
- rank-derived retrieval confidence baseline
- relevance-aware filtering with minimum-relevance floor
- Qwen2.5-VL 3B lazy GPU adapter with deterministic generation and token-likelihood confidence
- question-conditioned visual-entity extraction path in VLM adapter
- evidence/provenance verification baseline
- Platt/isotonic/temperature calibration implementations
- serializable Platt policy
- validation-only target-risk threshold selection
- selective metrics, ECE, Brier, AURC, paired-bootstrap utilities
- checkpoint/resume utilities
- A-OKVQA direct-answer evaluator matching official `min(matches/3,1)` rule
- A-OKVQA/OK-VQA manifest builder
- B0/B1 baseline runner
- B2/B3 filtered+verified runner
- validation-policy fit/apply scripts for B4/B5
- recorded-prediction evaluator with official-formula A-OKVQA direct-answer scoring
- controlled source/top-k ablation-matrix generator
- manuscript table and risk-coverage asset generator driven only by recorded metrics
- executable free-GPU development notebook
- PhD Paper 1→4 progression audit using retrieved library material
- novelty/experimental-design drafts

## TESTED
- CPU unit/integration suite: **17/17 PASSED** at latest checkpoint
- all main CLI entry points load successfully
- evaluator/asset pipeline executed on a synthetic fixture only (engineering validation; not a benchmark result)
- synthetic end-to-end engineering smoke test passes

## EXECUTED EXPERIMENTS
None on real OK-VQA/A-OKVQA VLM benchmark data in this ChatGPT container.

## EXECUTED ENGINEERING CHECKS
- CPU pytest suite
- synthetic retrieval → filtering → verification → calibration → selective-decision smoke path
- environment audit

## PARTIAL
- contradiction handling: transparent heuristic baseline exists; pretrained NLI verifier remains a planned ablation
- model selection: GPU fit not yet measured
- live knowledge-provider integration: implemented but external-provider experiments not executed here
- manuscript material: methodology scaffolding can be drafted, but results sections must wait for real experiments

## NOT EXECUTED
- real CUDA VLM inference
- full B0–B5 benchmark comparison
- source/top-k/verifier/calibration ablations on benchmark data
- statistical significance on real predictions
- final risk-coverage plots and reliability diagrams
- publication result tables

## BLOCKERS
- current execution container: CPU-only (`torch 2.10.0+cpu`, CUDA unavailable)

## NEXT HIGHEST-PRIORITY ACTION
Execute `notebooks/Paper4_End_to_End_Free_GPU.ipynb` on a genuine free CUDA GPU, beginning with DEV_MODE and preserving the locked validation/test protocol.

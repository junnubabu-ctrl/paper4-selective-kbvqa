# NOVELTY AUDIT — updated 2026-09-10

## Result
The original high-level idea **"verification + selective answering" is not sufficient novelty by itself**.

## Closest method families verified
1. **Dancette et al., CVPR 2023 — Improving Selective Visual Question Answering by Learning From Your Peers.** Establishes selective prediction/abstention as a VQA research problem and explicitly evaluates risk/coverage, including coverage at 1% risk.
2. **Khan and Fu, CVPR 2024 — Consistency and Uncertainty: Identifying Unreliable Responses From Black-Box Vision-Language Models for Selective Visual Question Answering.** Directly studies unreliable VLM responses using neighborhood consistency for black-box selective prediction.
3. **Srinivasan et al., Findings ACL 2024 — ReCoVERR.** Particularly close to the reliability objective: low-confidence answers are checked using additional evidence to reduce unnecessary abstention; calibration is part of the reliability protocol.
4. **Adjali et al., EMNLP 2024 — Multi-Level Information Retrieval Augmented Generation for KB-VQA.** Couples entity/passage retrieval and answer generation.
5. **Long et al., AAAI 2025 — ReAuSE.** Integrates retrieval into a generative multimodal model and includes retrieval calibration from relevance feedback; evaluated on OK-VQA and A-OKVQA.
6. **Compagnoni et al., CVPR 2026 — ReAG.** Combines coarse/fine retrieval with a critic that filters irrelevant passages before answer generation; demonstrates that retrieval filtering and evidence-grounded reasoning are already strong contemporary KB-VQA directions.
7. **Recent 2025–2026 retrieval/filtering work** further reduces novelty space for generic claims around evidence ranking, knowledge filtering, or calibration.

## Defensible Paper-4 delta to test
Do **not** claim novelty for retrieval, filtering, verification, calibration, selective prediction, or abstention individually.

The paper should test one integrated reliability hypothesis:

> **source-traceable multi-source external evidence → relevance filtering → answer/evidence provenance verification with an explicit contradiction signal → confidence fusion → validation-only calibration → target-risk selective answering, evaluated under controlled external-evidence corruption.**

The strongest defensible contribution, if supported by experiments, is therefore **risk-controlled factual reliability of externally grounded KB-VQA under noisy, missing, mis-ranked, and contradictory evidence**, with an auditable provenance trail.

## P0 novelty stress tests
1. **Selective baseline** — compare against a strong selective-VQA method or protocol-matched learned selection baseline; ordinary KB-VQA baselines alone are insufficient.
2. **Evidence-rescue baseline** — include a ReCoVERR-style verification/rescue comparison where feasible.
3. **Strong retrieval baseline** — include or discuss a strong recent RAG/KB-VQA retrieval method such as ReAuSE under compatible datasets/protocols.
4. **Controlled evidence corruption** — evaluate irrelevant injection, source dropout, contradiction injection, ranking corruption, and evidence scarcity.
5. **Provenance/support evaluation** — report cited-evidence coverage, source-level failures, verifier separation, and contradiction detection where labels are available.
6. **Two-dataset check** — show calibration/risk-coverage behavior on both A-OKVQA and OK-VQA if compute permits.
7. **No cross-paper superiority claim without protocol matching** — published numbers from different models, knowledge corpora, splits, or evaluation code are contextual comparisons only.

## Novelty decision rule
Paper-4 should proceed to a strong journal claim only if at least the following are demonstrated from real experiments:
- B3 improves unsupported-answer detection or verifier discrimination over B2;
- B4 measurably improves calibration over raw/verifier-fused confidence;
- B5 achieves a better risk-coverage trade-off at one or more predeclared target-risk levels;
- the advantage does not disappear under at least two evidence-corruption types;
- provenance/contradiction-aware verification contributes beyond semantic-only verification in ablation.

If these conditions fail, the method or paper framing must be revised before manuscript submission rather than inflating novelty language.

## Scientific positioning
ReAG demonstrates that critic-based filtering of noisy retrieved passages is already current KB-VQA methodology. ReAuSE demonstrates that retrieval calibration is also already active. Selective VQA and black-box reliability were established before this project. Consequently, Paper-4 must be positioned as an **integrated, risk-controlled, provenance-audited reliability framework**, not as the first verifier, first calibration method, first abstention method, or first retrieval filter for VQA.

## Status
- Selective-VQA prior-art verification: **VERIFIED** against official CVPR pages.
- ReAuSE verification: **VERIFIED** against the AAAI proceedings page.
- ReAG verification: **VERIFIED** against the official CVPR 2026 page.
- ReCoVERR and remaining related-work full-text extraction: **PARTIAL**; must be completed before final novelty wording.
- Exhaustive systematic review: **NOT CLAIMED**.

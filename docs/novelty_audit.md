# NOVELTY AUDIT — 2026-09-09

## Result
The original high-level idea **"verification + selective answering" is not sufficient novelty by itself**.

### Closest method families found
1. **Dancette et al., CVPR 2023 — Improving Selective Visual Question Answering by Learning From Your Peers.** Establishes selective prediction/abstention as a VQA research problem and evaluates risk/coverage.
2. **Khan et al., CVPR 2024 — Consistency and Uncertainty: Identifying Unreliable Responses From Black-Box Vision-Language Models for Selective VQA.** Directly studies unreliable VLM responses and selection scores.
3. **Srinivasan et al., Findings ACL 2024 — ReCoVERR.** Especially close: on A-OKVQA and VQAv2, low-confidence predictions are checked using additional visual evidence to reduce unnecessary abstention; calibration is also studied.
4. **Adjali et al., EMNLP 2024 — Multi-Level Information Retrieval Augmented Generation for KB-VQA.** Couples entity/passage retrieval and answer generation.
5. **Long et al., AAAI 2025 — ReAuSE.** Integrates retrieval and generation and includes retrieval calibration from relevance feedback on OK-VQA/A-OKVQA.
6. **Hong et al., 2025 — Wiki-PRF.** Processing/retrieval/filtering pipeline targeting irrelevant retrieved knowledge.
7. **Compagnoni et al., CVPR 2026 — ReAG.** KB-VQA RAG with a critic model that filters irrelevant passages and strengthens reasoning over evidence.
8. **Ma et al., Findings ACL 2026 — Ground Then Rank.** Training-free entity identification followed by evidence re-ranking; emphasizes entity- and fact-level grounding.
9. **Deng et al., Knowledge-Based Systems 2026 — collaborative parametric knowledge calibration for retrieval-augmented VQA.** Shows calibration terminology is already active within KB-VQA retrieval/generation.
10. **Wang et al., JVCIR 2026 — CKCR.** Context-aware knowledge construction/retrieval addresses missing knowledge, semantic gaps, and heterogeneous-source fusion.

## Defensible Paper-4 delta to test
Do not claim novelty for retrieval, filtering, verification, calibration, or abstention individually. Test the **joint reliability formulation**:

> source-traceable multi-source external evidence → relevance filtering → answer/evidence provenance verification with explicit contradiction signal → confidence fusion → held-out calibration → target-risk selective answering.

The strongest novelty claim, if experiments support it, should be about **risk-controlled factual reliability of externally grounded KB-VQA under noisy/contradictory evidence**, not about simply adding a verifier or abstention threshold.

## Required novelty stress tests
- Compare against a strong selective-VQA baseline, not only ordinary KB-VQA baselines.
- Include ReCoVERR-style evidence rescue/selection conceptually or experimentally where feasible.
- Demonstrate benefit specifically under evidence noise/contradiction perturbations.
- Report source attribution/provenance correctness or evidence-support quality, not only answer accuracy.
- Show calibration and risk-coverage improvements survive across at least two KB-VQA datasets if compute permits.

## Status
Literature audit: VERIFIED at title/abstract/official-page level for the above sources. Full-paper extraction and exhaustive systematic review: PARTIAL.

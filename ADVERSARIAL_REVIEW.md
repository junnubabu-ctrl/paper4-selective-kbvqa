# ADVERSARIAL REVIEW — PRE-EXPERIMENT

Scores are provisional and intentionally harsh; no score represents acceptance probability.

| Criterion | Score /10 | P-level | Main weakness before experiments |
|---|---:|---|---|
| Novelty | 6.5 | P0 | ReCoVERR/selective-VQA/ReAG overlap makes naive novelty claim unsafe. |
| Technical depth | 7.0 | P1 | Verifier currently transparent baseline; stronger learned/NLI ablation needed. |
| Mathematical rigor | 7.0 | P1 | Formal notation/pseudocode still to be finalized. |
| Dataset appropriateness | 8.0 | P1 | Good candidates; official split/evaluator integration not yet executed. |
| Baseline strength | 5.5 | P0 | Strong contemporary baselines not yet reproduced. |
| Ablations | 3.0 | P0 | Not executed. |
| Statistical rigor | 4.0 | P0 | Code utilities exist; no real paired predictions. |
| Implementation | 7.5 | P1 | Core reliability modules exist; VLM/data integration incomplete. |
| Reproducibility | 7.5 | P1 | Scaffold strong; clean free-GPU replay unverified. |
| Figures/tables | 2.0 | P2 | Must be generated from real results. |
| Error analysis | 2.0 | P1 | Requires real failures. |
| Scientific integrity | 9.5 | P0 safeguard | Unrun results explicitly marked; leakage guard exists. |

## P0 fixes required before manuscript result claims
1. Lock a novelty delta against ReCoVERR, selective VQA, ReAuSE, Wiki-PRF, ReAG, and 2026 grounding methods.
2. Run B0–B5 under a frozen protocol.
3. Integrate official evaluation and validate no test-set tuning.
4. Run the required ablations and paired statistical analysis.
5. Add a contradiction/noise stress test so “reliable evidence verification” is empirically identifiable rather than a label.

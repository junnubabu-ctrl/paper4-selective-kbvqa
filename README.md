# Paper 4 — Reliable Selective KB-VQA

Implementation-first PhD research repository for:

**EviTrust-VQA: Provenance-Calibrated Evidence Verification and Selective Answering for Knowledge-Based Visual Question Answering**

The experimentally defensible novelty target is: **source-traceable multi-source evidence, relevance filtering, explicit support/contradiction/insufficiency verification, provenance-aware reliability fusion, validation-only confidence calibration, and risk-controlled selective answering under noisy external knowledge**.

## Scientific status

No benchmark result is claimed until produced by an executed experiment. Synthetic smoke-test outputs are engineering fixtures and must never be copied into a manuscript table.

### Implemented and CPU-tested

- leakage-safe dataset manifest and inference view
- A-OKVQA and OK-VQA manifest construction
- modular Wikipedia, Wikidata, ConceptNet, and combined providers
- local caching and checkpoint/resume support
- Qwen2.5-VL 3B lazy GPU adapter with token-likelihood confidence
- question-conditioned visual-entity extraction
- transparent relevance filtering
- separate Qwen evidence critic with SUPPORTED / CONTRADICTED / INSUFFICIENT probabilities
- decomposed provenance: citation validity, source traceability, source diversity
- cross-family DeBERTa NLI verification control
- semantic reranking control
- Platt, isotonic, and temperature calibration utilities
- validation-only target-risk threshold selection
- coverage, selective accuracy/risk, ECE, Brier, and AURC utilities
- paired bootstrap and McNemar utilities
- official-formula A-OKVQA direct-answer evaluation
- **clean matched B0-B3 runner**
- B4/B5 calibration and selective-policy workflow
- deterministic calibration/evaluation data separation
- selective COCO image downloader for bounded calibration runs
- automatic B0-B5 result-summary export
- controlled source/top-k ablation framework
- automatic paper tables/figures from recorded metrics
- final free-GPU execution notebook

## One-click execution path

Use the final notebook:

**[`notebooks/Paper4_Final_Free_GPU_Execution.ipynb`](notebooks/Paper4_Final_Free_GPU_Execution.ipynb)**

In Google Colab or Kaggle:

1. Enable a CUDA GPU.
2. Open the notebook.
3. Keep `EVAL_MAX=None` for the real evaluation run.
4. Run all cells from top to bottom.
5. Preserve the generated `results/` artifacts.
6. Push the real results back to this repository before manuscript population.

The notebook automatically performs:

- environment and CUDA audit
- unit/leakage tests
- official A-OKVQA annotation preparation
- COCO validation-image setup
- fixed A-OKVQA training calibration subset using seed 2026
- selective download of calibration images
- B3 calibration inference
- validation-only Platt calibration and 5% target-risk selection
- matched B0/B1/B2/B3 execution on untouched A-OKVQA validation data
- B4 calibrated evaluation
- B5 selective evaluation
- automatic generation of `results/metrics/aokvqa_B0_B5_summary.json`

## Matched experimental progression

- **B0** — Qwen2.5-VL image + question only
- **B1** — B0 + raw multi-source evidence
- **B2** — B1 + relevance filtering
- **B3** — B2 + separate evidence critic + provenance-aware reliability
- **B4** — B3 + validation-fitted calibration
- **B5** — B4 + validation-frozen target-risk selective answering

The same question IDs and generator contract must be used for matched comparisons.

## Manual command equivalent

```bash
python scripts/run_variants.py --manifest <manifest> --variant B0 --out results/predictions/B0.jsonl --checkpoint results/checkpoints/B0.json
python scripts/run_variants.py --manifest <manifest> --variant B1 --out results/predictions/B1.jsonl --checkpoint results/checkpoints/B1.json
python scripts/run_variants.py --manifest <manifest> --variant B2 --out results/predictions/B2.jsonl --checkpoint results/checkpoints/B2.json
python scripts/run_variants.py --manifest <manifest> --variant B3 --out results/predictions/B3.jsonl --checkpoint results/checkpoints/B3.json
```

Fit calibration and the selective policy only on the fixed calibration subset:

```bash
python scripts/fit_selective_policy.py \
  --manifest <calibration_manifest> \
  --predictions <calibration_B3_predictions> \
  --target-risk 0.05 \
  --out results/policies/aokvqa_policy_5pct.json
```

Apply that policy unchanged to held-out B3 predictions:

```bash
python scripts/apply_selective_policy.py \
  --policy results/policies/aokvqa_policy_5pct.json \
  --predictions <heldout_B3_predictions> \
  --out results/predictions/B5.jsonl
```

## Result artifacts required before manuscript claims

A manuscript-ready benchmark release should contain at least:

- `results/predictions/aokvqa_val_B0.jsonl`
- `results/predictions/aokvqa_val_B1.jsonl`
- `results/predictions/aokvqa_val_B2.jsonl`
- `results/predictions/aokvqa_val_B3.jsonl`
- `results/predictions/aokvqa_val_B5.jsonl`
- `results/policies/aokvqa_policy_5pct.json`
- `results/metrics/aokvqa_val_B0.json`
- `results/metrics/aokvqa_val_B1.json`
- `results/metrics/aokvqa_val_B2.json`
- `results/metrics/aokvqa_val_B3.json`
- `results/metrics/aokvqa_val_B4.json`
- `results/metrics/aokvqa_val_B5.json`
- `results/metrics/aokvqa_B0_B5_summary.json`

After the main result gate, run the prespecified verifier, source, top-k, reranking, corruption, statistics, and efficiency experiments.

## Integrity rules

- Ground-truth answers never enter retrieval queries, generation prompts, evidence ranking, or verification.
- Calibration and operating-threshold selection use separate calibration data only.
- Held-out evaluation data are never used to tune weights, prompts, top-k, or thresholds.
- Synthetic smoke metrics are never reported as benchmark results.
- Do not fabricate accuracy, p-values, confidence intervals, runtime, VRAM, baselines, or ablations.
- Do not commit datasets, model weights, secrets, or large caches.
- Label every partial/subset experiment with exact sample count, split, model revision, seed, and hardware.

## Resource policy

The default path uses PyTorch, Hugging Face Transformers, public knowledge resources, and open models. No paid OpenAI, Anthropic, Gemini, commercial vector database, or paid inference endpoint is required.

See `REPRODUCIBILITY.md`, `EXPERIMENT_PLAN.md`, `docs/novelty_audit.md`, and `IMPLEMENTATION_STATUS.md` before promoting any number into the manuscript.

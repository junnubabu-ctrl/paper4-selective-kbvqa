# Paper 4 — Reliable Selective KB-VQA

Implementation-first PhD research repository for the working direction:

**Evidence Verification and Selective Answering for Knowledge-Based Visual Question Answering with Large Language Models**

The experimentally defensible novelty target is narrower than the title: **source-traceable multi-source evidence, relevance filtering, contradiction/provenance-aware verification, validation-only confidence calibration, and risk-controlled selective answering under noisy external knowledge**.

## Scientific status

No benchmark result is claimed until produced by an executed experiment. Synthetic smoke-test outputs are engineering fixtures and must never be copied into a manuscript table.

### Implemented and CPU-tested

- leakage-safe dataset manifest and inference view
- A-OKVQA and OK-VQA manifest construction
- modular Wikipedia, Wikidata, ConceptNet, and combined providers
- local caching and checkpoint/resume support
- relevance-aware evidence filtering
- Qwen2.5-VL 3B lazy GPU adapter with token-likelihood confidence
- visual-entity extraction path
- evidence/provenance/contradiction verifier baseline
- Platt, isotonic, and temperature calibration utilities
- validation-only target-risk threshold selection
- coverage, selective accuracy/risk, ECE, Brier, AURC utilities
- paired bootstrap and McNemar count utilities
- official-formula A-OKVQA direct-answer evaluation
- B0/B1 baseline runners and full B2/B3 inference runner
- B4/B5 validation-policy fit/apply workflow
- controlled source/top-k ablation matrix generator
- automatic CSV/Markdown/LaTeX table generation from recorded metrics
- automatic risk–coverage PDF + 600-dpi PNG generation from recorded predictions
- free-GPU end-to-end notebook

Current CPU gate: **17/17 tests passed** at the packaging checkpoint.

Full OK-VQA/A-OKVQA VLM benchmark experiments are **NOT EXECUTED** in this ChatGPT container because CUDA is unavailable.

## Repository workflow

### 1. CPU integrity gate

```bash
python scripts/inspect_environment.py
PYTHONPATH=src pytest -q
PYTHONPATH=src python scripts/dev_smoke.py
```

### 2. Free-GPU development run

Open:

```text
notebooks/Paper4_End_to_End_Free_GPU.ipynb
```

Use Kaggle GPU or Colab Free. `DEV_MODE=True` is the default; do not promote development-subset metrics to paper results.

### 3. Build an official-data manifest

A-OKVQA example:

```bash
python scripts/build_manifest.py aokvqa \
  --aokvqa-dir datasets/aokvqa \
  --coco-dir datasets/coco \
  --split val \
  --out datasets/manifests/aokvqa_val.jsonl
```

### 4. Run baseline progression

```bash
python scripts/run_baseline.py --manifest datasets/manifests/aokvqa_val.jsonl \
  --baseline B0 --out results/predictions/b0_val.jsonl \
  --checkpoint checkpoints/b0_val.json

python scripts/run_baseline.py --manifest datasets/manifests/aokvqa_val.jsonl \
  --baseline B1 --out results/predictions/b1_val.jsonl \
  --checkpoint checkpoints/b1_val.json

python scripts/run_proposed.py --manifest datasets/manifests/aokvqa_val.jsonl \
  --out results/predictions/b3_val.jsonl \
  --checkpoint checkpoints/b3_val.json
```

### 5. Fit calibration/selective policy on validation only

```bash
python scripts/fit_selective_policy.py \
  --manifest datasets/manifests/aokvqa_val.jsonl \
  --predictions results/predictions/b3_val.jsonl \
  --target-risk 0.05 \
  --out results/metrics/selective_policy.json
```

Freeze the resulting policy. Do not refit on the held-out final test split.

### 6. Apply the frozen policy and evaluate

```bash
python scripts/apply_selective_policy.py \
  --policy results/metrics/selective_policy.json \
  --predictions results/predictions/b3_test.jsonl \
  --out results/predictions/b5_test.jsonl

python scripts/evaluate.py \
  --manifest datasets/manifests/aokvqa_test_w_ans.jsonl \
  --predictions results/predictions/b5_test.jsonl \
  --dataset aokvqa --confidence-field calibrated_confidence \
  --out results/metrics/b5_test.json
```

For OK-VQA, this repository labels its built-in raw-string consensus score as **auxiliary**. Publication reporting should use the official VQA evaluation toolchain.

### 7. Controlled ablations

Generate a dry-run matrix first:

```bash
python scripts/run_ablations.py \
  --manifest datasets/manifests/aokvqa_val.jsonl \
  --out-dir results/ablations
```

Execute only after the development gate passes:

```bash
python scripts/run_ablations.py ... --execute
```

### 8. Generate manuscript assets from actual recorded results

```bash
python scripts/generate_paper_assets.py \
  --metrics results/metrics/b0_test.json results/metrics/b5_test.json
```

No table values are manually hard-coded by the asset generator.

## Free/open-resource policy

The default path uses PyTorch, Hugging Face Transformers, FAISS/Sentence Transformers where applicable, free public knowledge resources, and a locally downloaded open model. No paid OpenAI, Anthropic, Gemini, commercial inference, paid vector database, or paid GPU endpoint is required.

## Integrity rules

- Ground-truth answers never enter retrieval queries, generation prompts, evidence ranking, or verification.
- Calibration and operating-threshold selection use held-out validation data only.
- Final test data are evaluation-only.
- Do not fabricate accuracy, p-values, confidence intervals, runtime, VRAM, baselines, or ablations.
- Do not commit datasets, model weights, secrets, or large caches.
- Label every partial/subset experiment with its exact sample count and split.

See `REPRODUCIBILITY.md`, `EXPERIMENT_PLAN.md`, `docs/novelty_audit.md`, and `IMPLEMENTATION_STATUS.md` before any manuscript claim is written.

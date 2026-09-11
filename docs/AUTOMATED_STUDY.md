# Resumable A-OKVQA study

The updated `notebooks/Paper4_Final_Free_GPU_Execution.ipynb` is the entry point. Select a CUDA GPU, run the cells, and authorize Drive mounting. It freezes the checked-out Git commit and model revision IDs, writes outputs directly under `MyDrive/Paper4Runs/<RUN_NAME>`, and exports a results ZIP excluding COCO images, weights and retrieval caches.

## Scope and status

The code has CPU regression and postprocessing tests. GPU/model execution, live dataset downloads and live knowledge APIs have not been validated in the authoring environment. No benchmark numbers were generated while implementing this change.

The extended matrix has 38 inference configurations: B0–B3, three single-source controls, three additional top-k settings, NLI and semantic reranking controls, two fusion-component controls, and twelve evidence perturbations applied to B1 and B3. Calibration runs and the 50-question development gate are additional executions. B4/B5 reuse B3 answers and do not independently regenerate answers.

The default seed is 2026. Official OK-VQA evaluation, protocol-matched external baselines, additional corruption-seed repetitions, independent evidence-support assessment, evidence-ID-removal controls and paired selective-risk/AURC confidence intervals remain outstanding. Completing this script is not equivalent to completing every experiment in `EXPERIMENT_PLAN.md` or producing a submission-ready paper.

## Commands

```bash
python -m pip install -e '.[vlm,retrieval,dev,analysis]'
python -m pytest -q
python scripts/run_full_study.py --plan
python scripts/run_full_study.py --out results/full_study --scope extended
```

Use `--scope main` for B0–B5, calibration and target-risk sensitivity. A separate development run can use `--cal-n 100 --eval-max 100 --out results/development`. Do not change configuration and reuse an existing directory. Resuming with the same command checks stage outputs, input hashes and run identity. Interrupted inference resumes from checksum-verified prediction records, not a potentially stale checkpoint list. A partial final JSONL write is discarded; a corrupt complete record raises an error. Hardware/runtime errors stop the process immediately.

Run directories contain source URL records, manifests, pinned model revisions, resolved dependencies, predictions, checksums, policies, per-question metrics, stage logs, session hardware/VRAM metadata, tables and 600-dpi risk-coverage figures. Some runtime limits may require more than one Colab session. Persistent Drive storage requires sufficient space for images and can add I/O overhead.

## Scoring and interpretation

The A-OKVQA direct-answer scorer uses exact string agreement and excludes `difficult_direct_answer=True` records, as in the [official evaluator](https://github.com/allenai/aokvqa/blob/main/evaluation/eval_predictions.py). Inference still records predictions for the full selected manifest; the evaluated sample count can consequently be smaller. Training-split calibration and validation-split evaluation are distinct at both question and image level.

The correctness event for calibration/selective risk is full-credit agreement. The 1%, 5%, 10% and 20% operating points maximize empirical calibration-set coverage subject to the target risk. They are not finite-sample risk guarantees. Actual held-out selective risk must be reported, including when it exceeds the target. An all-abstain policy uses a threshold above 1 and reports undefined selective accuracy/risk as null in evaluator output.

Ablations use the matched runner. Each clean ablation fits its own calibration policy on the same calibration IDs. Corruption experiments freeze the corresponding clean B1 or B3 policy. Contradictions are synthetic linguistic perturbations, not independent gold factual contradictions. Cross-question distractors come only from calibration retrievals; they are not guaranteed irrelevant. Ranking perturbations can occasionally leave short evidence lists unchanged. These limitations must accompany robustness claims.

Entity-indexed Wikidata and ConceptNet APIs receive visible entity names (or question-only keyword fallback) instead of an entire question as a concept ID. Wikipedia receives a question/entity query. All retrieved evidence is recorded, and source yield/empty-retrieval diagnostics are exported. These are search snippets/entity descriptions and ConceptNet edges, not a claim of multi-hop KG traversal. Citation diagnostics are not gold grounding accuracy. GPU time and memory fields describe observed execution, including cold-start effects and resumptions; they are not controlled steady-state throughput benchmarks.

The report computes paired bootstrap confidence intervals for four planned soft-accuracy contrasts and exact McNemar full-credit tests with Holm adjustment across those four tests. It does not infer significance for selective-risk differences or claim that within-family baselines establish superiority to external methods.

## Repairs included

- Fixed the annotation downloader's misplaced future import and calibration JSONL serialization path.
- Preserved the official difficult-answer evaluation flag.
- Prevented parse failures from inventing citations to every evidence item.
- Fixed summary selective-risk lookup and the confidence=1 all-abstain edge case.
- Added prediction-driven crash recovery, configuration/model identity, strict completeness checks and calibration/evaluation overlap checks.
- Added atomic, validated COCO downloads with bounded retries.
- Added NLI label-mapping failure detection instead of reporting zero probabilities as valid verification.

Review the realised outputs before transferring any number into a manuscript. `completion.json` deliberately retains `manuscript_ready=false` and lists further scientific requirements.

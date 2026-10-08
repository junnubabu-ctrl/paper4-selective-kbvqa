# Paper 4: bounded fixed-candidate restart

Restart date: 9 October 2026 (Asia/Kolkata).

## Why this restart is needed

The October manuscript asks whether critic and citation/provenance diagnostics improve selective release beyond an independently calibrated likelihood-only policy for the **same frozen answer and ordered evidence**. The available September runner generates independent B0–B3 answers. It cannot, by itself, establish that fixed-candidate comparison. The branch named `automation/scirep-review-resolution-20261007` still resolved to source commit `45190e6d2bee19e9805751fd1bed9aa1430ec43d` at the audit. This bounded restart is based on that source; it does not recover every unarchived manuscript implementation.

## Scope and implemented contract

- TRAIN-derived development only, bounded to 1–50 questions. No calibration fitting, held-out evaluation, accuracy claim or benchmark completion.
- Generate one filtered-evidence B2 answer; archive it before critique; derive a B3 critic score from exactly that candidate and the same ordered context. Candidate and context hashes, input hashes, model revisions and runtime identities protect resume.
- Retain exact raw generator text and parser diagnostics. Invalid answer JSON/schema blocks completion. Invalid grounding JSON falls back to question keywords with its diagnostic retained.
- Preserve repeated and unknown string citations in the raw record. Send the valid subset to the critic. Citation validity uses distinct generated IDs; traceability uses distinct valid cited items. Source coverage is not evidence truth or source independence.
- Globally cap retrieved evidence after combining queries and sources. Archive the selected live context before answer generation. This is a saved retrieval context, not an immutable upstream knowledge release.
- Requested NF4 must be confirmed on loaded models. Dynamic Qwen image processing uses a maximum pixel budget of 1,003,520, not a universal 224 × 224 crop. Record actual image and processor dimensions.
- Keep labels in a separate offline reference manifest. Do not pass human rationales, multiple-choice answers or reference answers to retrieval, generator or critic.

## Verified local starting evidence

The official A-OKVQA annotation archive was downloaded (3,661,719 bytes; SHA256 `3992b488babc0c1147f0def18c7a55274aeeb37ab668cf80226b8f62ee35a8e1`). The train annotation contains **17,056 questions referring to 16,540 distinct image IDs**. These counts are annotation identities, not downloaded or executed images.

The restart freezes 50 questions from 50 distinct TRAIN image groups with seed 2026. The image IDs of all 50 groups must be excluded from subsequent calibration pools. This is a new development partition; do not reuse an earlier calibration file without proving disjointness against these groups.

Local result at restart: 50 inference records, 50 separate offline reference records, **zero verified JPEGs and zero genuine model predictions**. COCO image-host requests timed out / name resolution failed here. No CUDA GPU or model-execution stack is available in the local runtime. All **159 CPU software tests passed**, including the explicit transport extension; `src` and `scripts` compiled successfully. CPU tests use software stand-ins and are not model or benchmark results.

## Execute the first gate

Use `notebooks/Paper4_Fixed_Candidate_Restart.ipynb` in a CUDA Colab runtime. The notebook pins this corrected source to a commit, defaults to temporary runtime storage and includes archive-download cells after every gate. Download those archives for retention, including after a failed gate. Optional Drive mounting requires enabling it and granting access. Run cells in order; failure must stop the sequence.

The live Colab recovery authenticated successfully and allocated a Tesla T4 with Torch 2.11.0+cu130 / CUDA 13.0. The custom COCO HTTPS hostname failed certificate hostname validation. A bounded probe confirmed its DNS alias points to the `images.cocodataset.org` S3 bucket in US East and the same selected object returned HTTP 200 via `https://s3.amazonaws.com/images.cocodataset.org/train2017/000000287900.jpg`, with default certificate validation enabled. Use the declared `coco-s3-path` image transport rather than disabling TLS validation. The actual selected download URL, canonical COCO identity and verified image hashes remain in the inventory. This connectivity probe is not a decoded-image or model result.

Local command equivalents after installing a compatible CUDA stack:

```bash
python scripts/prepare_restart_pilot.py --out-dir DATA/pilot50 --download-annotations --download-images --image-transport coco-s3-path --timeout 30
python scripts/restart_preflight.py --pilot-dir DATA/pilot50 --out RUNS/preflight.json
python scripts/run_fixed_candidate_pilot.py --pilot-dir DATA/pilot50 --out-dir RUNS/one_image --max-samples 1
python scripts/run_fixed_candidate_pilot.py --pilot-dir DATA/pilot50 --out-dir RUNS/pilot50 --max-samples 50
```

The one-image inference gate still requires the prepared 50-image inventory to validate. Data preparation downloads only those 50 images, not COCO in full. It stops after three consecutive download failures and records the remaining identities as unattempted. Repeat the same preparation command to resume; it rejects changed frozen identities or image hashes.

If provider access fails, diagnose the source URL and retain the failure; do not silently relabel a question-only or reduced-source run as the intended retrieval experiment. An optional label-free frozen evidence file is a separately declared source mode, not an automatic fallback.

## Pass criteria and next scientific work

1. Actual CUDA, pinned model revisions, decoded original JPEGs and requested loaded quantization are recorded.
2. Every requested candidate is valid; the exact candidate and ordered context are retained for both score variants; the critic returns finite auditable continuation scores.
3. Raw text, citations, retrieval context, processor geometry, prompts, identities and measured execution resource logs survive a runtime disconnect and a validated resume.
4. The one-image gate succeeds before the 50-question development run. Neither result establishes accuracy or superiority.

After development succeeds, freeze the evaluation protocol. Build image-disjoint calibration-fit and threshold-selection partitions from TRAIN, excluding all 50 development image groups. Fit separate policies for likelihood and critic/provenance scores; lock each policy before touching A-OKVQA validation. Then evaluate paired coverage/risk and uncertainty, add component controls, run independent OK-VQA replication, and complete blinded human evidence checks for grounding claims.

Do not extend to a new 15,000-image dataset, Paper 5, or training a foundation LLM until this Paper 4 evidence path is working. The scientific contribution is the tested decision mechanism and its bounded evidence, not ownership of the adopted pretrained Qwen weights or a chat interface.

## Primary sources used to define the boundary

- A-OKVQA data and official scoring: https://github.com/allenai/aokvqa and https://github.com/allenai/aokvqa/blob/main/evaluation/eval_predictions.py
- Qwen2.5-VL processor and model API: https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_vl
- Generation likelihood API: https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation
- ReCoVERR prior confidence/verification work: https://aclanthology.org/2024.findings-acl.767/
- AWS path-style and CNAME bucket routing: https://docs.aws.amazon.com/AmazonS3/latest/userguide/VirtualHosting.html

ReCoVERR already studies confidence, verification and evidence acquisition. Paper 4's proposed incremental advantage must therefore be tested against a strong likelihood control under its specified fixed-candidate design; it cannot be inferred from the existence of a critic.

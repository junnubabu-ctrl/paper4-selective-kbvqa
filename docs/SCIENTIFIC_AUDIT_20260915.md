# Scientific integrity audit 15 September 2026

Base: ec63986f30aa48c8ad7322328497447695b43da7 on fix/calibration-jsonl-20260911. This isolated audit branch updates Paper 4 only.

## Corrections and reasons

1. Development samples previously came from the calibration subset. Development image groups are now excluded from calibration, separately downloaded and validated.
2. Platt fitting and operating-threshold selection previously reused the same calibration labels. They now use deterministic disjoint image groups (seed 2026, approximately half the groups each). The fit partition must include both outcome classes. Policy files identify both partitions. The 5% full-credit error target remains an empirical operating point, without a finite-sample guarantee. Selecting a threshold on its own partition still incurs selection optimism; held-out evaluation is required.
3. Malformed generator output previously received fallback citations that the model did not supply. Fallback text now receives no citations.
4. Real Qwen generation records now include a SHA256 prediction identity over image bytes, question, exact ordered evidence payload, rendered prompt, model configuration and requested/resolved revision, processor configuration, decoding settings, quantization, image budget and torch initial seed. Resume additionally rejects changed questions, image bytes or supplied visual entities. This does not justify reusing a prediction after changing evidence. Outer run identities and frozen artifacts remain required.
5. Rejecting every question now reports undefined selective risk as null, not zero.

## Verification

62 CPU software tests pass on Python 3.12.14, including image-group separation, empty-coverage semantics, malformed-citation handling, changed-input resume rejection, and both datasets' synthetic postprocessing. These fixtures are software checks, not benchmark findings. CI has not been independently revalidated for this revision. Real model inference was not executed in this environment (no CUDA GPU and no installed torch).

## Scientific gates still open

- Run real frozen-model inference and retain complete manifests, prediction identities, logs, metrics and resource measurements. The existing 87-configuration plan is not an executed experiment.
- Replace live-source dependence with licensed immutable knowledge snapshots; archive source/version hashes and verify dataset permissions and model licences at exact revisions.
- Independently annotate evidence support, contradiction and sufficiency. The Qwen critic shares a model family with the generator and is not independent ground truth.
- Complete capacity/budget-matched external verification/selective baselines. Heuristic provenance weights are not learned truth estimates.
- Realistically validate corruption semantics. Synthetic negation alone does not establish realistic misinformation robustness.
- Complete image-cluster inference, multiplicity control, transfer analysis and cost reporting from real outputs.
- Finish final-version reference reconciliation, close-work novelty review, institutional conditions, author declarations and approvals.

The manuscript is a research draft. No benchmark number or submission-ready status is asserted by this audit.

## Checkpoint freeze

`configs/models_20260915.json` records exact upstream checkpoint revisions queried on 15 September 2026. The automatic study uses these pins and rejects an incompatible existing model lock. Qwen2.5-VL-3B carries the Qwen Research License Agreement, not Apache 2.0; the 1.5B critic and MiniLM reranker use Apache 2.0, and the DeBERTa model card states MIT. Model use and any redistribution remain subject to those upstream terms. No weights were downloaded or redistributed in this audit.

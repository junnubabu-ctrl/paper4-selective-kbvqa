# Constrained JSON failure replay notebook

Open `Paper4_Constrained_JSON_Failure_Replay_20261009.ipynb` in Colab, select a CUDA GPU, and upload `JB_Paper4_JSON_Failure_Replay_Inputs_20261009.zip` through the Files pane into `/content/`.

Run cells in order, not Run All. Each asynchronous launch returns immediately; repeat its monitor cell until `FINISHED`, then run the next validation cell. Existing jobs are displayed rather than duplicated. Existing run/code folders require a deliberately fresh `RUN_TAG`.

The notebook pins source `8377ce77b2f1b1f8851ff6e72f36ccbd2902b7bd`, Transformers 4.57.1 and LM Format Enforcer 0.11.3; verifies the exact 1,544,334-byte ZIP and all 53 member hashes; runs the full 215-test suite without skips; validates the inference boundary and original captured contexts; requires one valid GPU case before the nine-case diagnostic; and downloads a complete archive including failures and raw outputs.

These are nine prior strict-format failures selected from TRAIN development. `benchmark_result=False`; no model training, reference-answer accuracy, calibration fitting, threshold selection or held-out evaluation occurs. Unknown citations and invalid outputs remain diagnostics. Frozen question-indexed facts are reused without knowledge-provider HTTP requests. Public model downloads remain separate. No Drive mount, login or compute purchase is requested.

If any validation fails, preserve the logs and use the final archive cell after children finish. A lost Colab runtime loses temporary files. This blank notebook contains no executed results and makes no answer-correctness claim.

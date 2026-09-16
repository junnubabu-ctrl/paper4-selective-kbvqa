# Paper 4 Colab automation — 16 September 2026

## Use
Open notebooks/Paper4_Run_All_Connected.ipynb in Colab. Choose a GPU, select Runtime > Run all, and authorize Drive when prompted. The notebook uses the frozen scientific code at 2b9788e09e965ad894a5c29ce81e84cccfa52a7d.

The run folder name is derived from the code and configuration. Data, progress and predictions are written inside MyDrive/Paper4Runs. Reconnecting and running all with the same settings resumes this folder. A core dependency mismatch stops execution rather than mixing environments.

## Automation boundary
Sign-in, GPU allocation, Drive consent and reconnection are controlled by Google and may require user interaction. A Drive connection permits file access; it does not start or control a GPU session. This notebook does not buy services, publish predictions, submit manuscripts, use anti-idle tricks, or bypass runtime limits. Google policy: https://research.google.com/colaboratory/faq.html

After an ordinary study exception or normal completion, a bundle is attempted in the run folder. It includes prediction/evidence records, metrics, policies, logs, status and checksums, excluding datasets, weights and retrieval caches. An abrupt runtime loss can prevent export; previously flushed Drive files remain. The source caches are not licensed immutable knowledge snapshots.

## Verification status
The underlying scientific checkpoint had 71 passing CPU software checks. This new notebook was inspected and JSON-structure checked but was not executed in Python or on a GPU. The workspace became unavailable, and installing the local test runner was blocked by an automatic approval-review usage-limit error. No passing test result for this launcher is claimed. The more extensive colab_run_all.py helper and its new tests remain unpublished work in progress and are not used by this notebook.

The notebook runs the scientific checkpoint's tests before model inference. Their success is software evidence only. GPU compatibility, memory needs, runtime and benchmark findings remain unverified.

## Scientific completion
COMPLETED_REQUESTED_PROTOCOL means the requested stages produced their expected completion/report files. It does not establish evidence-label validity, state-of-the-art gains, complete external comparisons, PhD eligibility or manuscript readiness. Full independent evidence annotation, corruption validity, snapshot licensing, final literature reconciliation and author review remain required.

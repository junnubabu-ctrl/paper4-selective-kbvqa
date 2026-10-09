# Structured JSON development recovery

This is a separate development protocol, not a benchmark result. The original
strict parser and all earlier invalid records remain unchanged. The opt-in CLI
flag `--constrained-json` uses `lm-format-enforcer==0.11.3` to restrict generated
tokens to the declared answer or entity object schema. No Markdown stripping,
JSON repair, reference-answer assistance, or citation-ID substitution is used.
Citation strings are not restricted to an enum of supplied IDs: unknown IDs
remain visible to the existing provenance diagnostics. Schema constraints do
not establish factual support or answer correctness. Budget exhaustion and
whitespace-only answers can still fail the strict parser.

## Score definition

The default generator remains the legacy path using processed generation
scores. In the constrained path, whole-response log probabilities come from
Transformers `output_logits=True`, before grammar masking and other generation
processors. This scores the selected response under the original model given
its generated prefix, including JSON, citation and EOS tokens. The geometric
mean is uncalibrated and is not a correctness probability. The new score source
is explicitly versioned; it is not numerically equivalent to legacy processed
scores, which included the checkpoint's repetition penalty. New calibration is
required before any selective-answering policy can use these outputs.

Prompts, models, 192-token answer budget, image processing and critic remain
unchanged. Constraint schema, package version, effective parser configuration,
tokenizer identity/cleanup and score source are retained in generation identity.
The constrained entity decoder is available, but the nine-case answer replay
uses the original captured entity lists to isolate the answer decoder.

## Bounded nine-case replay

The input bundle selects exactly the nine strict answer-JSON failures from the
archived 50-input TRAIN-development attempt. All nine original images, questions,
retrieved facts and effective visual entities are preserved. Selection is based
on prior failure and cannot support an unbiased accuracy estimate. The replay
uses frozen question-indexed evidence and `--no-auto-entities`, so it makes no
live Wikipedia, Wikidata or ConceptNet requests. Model downloads are separate
from knowledge-provider access. The archive is a local capture, not verified
immutable upstream knowledge or a blanket license declaration.

After verifying the bundle manifest, attestation, image inventory, source record
hashes and model lock, run a one-case gate in a new directory, then the nine-case
diagnostic in another new directory. Keep `--continue-on-invalid` for the full
diagnostic; invalid records must remain failures with a nonzero CLI status.

```bash
python scripts/run_fixed_candidate_pilot.py \
  --manifest /content/JB_Paper4/json_replay/inputs/inference_manifest.jsonl \
  --train-development-attestation /content/JB_Paper4/json_replay/inputs/train_development_attestation.json \
  --evidence-jsonl /content/JB_Paper4/json_replay/inputs/frozen_question_evidence.jsonl \
  --source-ledger /content/JB_Paper4/json_replay/inputs/source_ledger.json \
  --sources wikipedia,wikidata --no-auto-entities --constrained-json \
  --max-samples 9 --continue-on-invalid \
  --out-dir /content/JB_Paper4/json_replay/constrained_diagnostic9
```

Adjust only the declared input/output paths to the verified extracted bundle.
Execution is complete only when all expected checksummed records and a final
summary are present. Reconcile old/new context content, actual parser status,
unknown/repeated IDs, critic execution, raw logits likelihood and resources.
Retain failures, commands, environments and code identity. Do not combine replay
counts with new unique benchmark cases or claim all 50 original inputs completed.

## Subsequent research gates

Audit evidence relevance and freeze appropriate source versions and licenses.
Then run a broader development check, prepare image-disjoint calibration-fit and
threshold subsets, lock the policies, and evaluate held-out predictions with
matched comparisons, ablations, uncertainty and failure analysis. No model
training, manual upload/chat interface, manuscript readiness or PhD completion
follows from JSON recovery alone.

## Primary implementation sources

- [Transformers 4.57.1 generation outputs](https://huggingface.co/docs/transformers/v4.57.1/en/internal/generation_utils): unprocessed logits versus processed scores.
- [Transformers 4.57.1 generation API](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation): prefix token constraints and generation configuration.
- [LM Format Enforcer official repository](https://github.com/noamgat/lm-format-enforcer): direct Transformers prefix callback.
- [Pinned LM Format Enforcer release](https://pypi.org/project/lm-format-enforcer/0.11.3/): package version 0.11.3. Published wheel SHA256 `cf586350875def1ae7a8fba84fcbbfc8371424b6c9d05c1fcba70aa233fbf06f`.

Only the applicable documentation and implementation paths were reviewed; this
is not a full literature review or a novelty claim.

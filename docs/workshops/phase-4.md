# Phase 4 — OCR-free pipeline workshop plan

**Status: Engineering delivery complete, 2026-10-15.** The user
authorized complete Phase 4 delivery. A–D now retain 47 executed code cells,
resource failures, actual tiny/main training, validation selection, matched
comparison, and source/GPU/CPU verification. See the
[delivery record](../phases/phase-4.md) and linked experiment reports. Test remains
sealed. This plan preserves the original learning sequence below; it is not a
claim that the learner personally completed every reflection. [PROJECT.md](../../PROJECT.md)
owns phase status; the [experimental contract](../../EXPERIMENT_DESIGN.md) owns scoring.

## Purpose and inherited boundary

Build a second extraction path that reads the whole canonical `alb_id` card and
returns the same seven fields as the adopted OCR pipeline:

```text
1084 × 680 RGB canonical card
    -> Donut image processor
    -> image encoder + autoregressive decoder
    -> generated structured sequence
    -> parser
    -> shared normalization and semantic validation
    -> seven-field result, raw evidence, and timings
```

Donut must learn both which text belongs to each field and how to serialize the
answer. OCR already receives known field rectangles; Donut receives the image
without those rectangles. This compares the two declared pipelines, rather
than isolating architecture alone. Neither greater accuracy nor higher latency
is assumed for Donut.

Reuse the 70/15/15 identity split, seed 2026, canonical images, field schema,
normalization, validators, and scorer. Train only on training identities;
use validation for selection. Keep test sealed through Phase 4 development.
Model-specific resizing/padding is internal preprocessing, is recorded, and is
timed; the shared canonical input is unchanged. Full degradation sweeps belong
to Phase 5, routing to Phase 6, and serving/load to Phase 7.

## Concepts and learning approach

The initial [OCR-free learning guide](../learning/ocr-free.md) explains the
architecture and the decisions this phase must investigate. Extend learning
material alongside executable work, linking concepts to actual cells and runs.

The learning sequence covers:

1. Whole-document extraction versus crop recognition; Swin image encoding,
   hierarchical features, decoder cross-attention, and autoregressive output.
2. Task prompts, field delimiters, token IDs, vocabulary extension, fixed field
   order, sequence length, end tokens, and conversion back to a mapping.
3. Training examples, paired views of an identity, teacher forcing, shifted
   labels, ignored padding/prompt tokens, loss, and free-running generation.
4. Weights, gradients, activations, optimizer state, precision, gradient
   accumulation/checkpointing, and the limits of a small GPU.
5. Overfitting, checkpoint selection, plausible but wrong values, malformed
   output, and accuracy/cost comparison under the same evaluation contract.

For each workshop: explain the question, predict the outcome, run a small
example, inspect actual tensors/sequences/errors, and record the interpretation.
Reflection prompts do not become passed assessments merely because code ran.
This plan does not prescribe a learner/agent division of notebook authorship.

## Proposed notebooks and checkpoints

All four notebooks exist with explanations and retained outputs. C/D explicitly
review real separately executed expensive runs, then perform their own replay
and live checks; they do not fabricate training or timing evidence.

| Notebook | Work | Evidence needed to move on |
|---|---|---|
| [05a — Donut foundations](../../notebooks/05a_donut_foundations.ipynb) | Executed: canonical paired inputs, native/landscape preprocessing, encoder and token trace, isolated vocabulary probe, real GPU generation | Delivered model revision/hashes, tensor shapes, parameter count, visual checks, raw output, memory, and a historical failure record; training feasibility remains unmeasured |
| [05b — dataset](../../notebooks/05b_donut_dataset.ipynb) | Executed data/target/parser/label alignment checks and audit of actual fresh-process probes | 170 round trips; 37–43 tokens; 70/15 disjoint permitted identities; strict failures; measured Adafactor follow-up |
| [05c — fine-tuning](../../notebooks/05c_donut_finetuning.ipynb) | Executed review/replay of real tiny/main training and validation selection | Tiny 14/14; 16 main epochs; epoch 10 selected; 3360 field records replayed; complete curves/errors/source snapshots |
| [05d — evaluation](../../notebooks/05d_donut_evaluation.ipynb) | Executed paired scoring/gallery/timing review and live GPU/CPU requests | 30/30 sequence and 210/210 raw-field parity; 180/180 exact repeats; measured latency; verified reusable source and freeze |

### Workshop 0 — Orientation and candidate investigation

Start `05a` by tracing what Phase 3 already provides and what Donut must learn.
Investigate `naver-clova-ix/donut-base` as the initial pretrained candidate, pin
its exact revision before training, and verify compatibility with the installed
Transformers version. A receipt/DocVQA checkpoint is not automatically an
identity-field extractor. A base-checkpoint generation is an input/output smoke
check, not evidence that the seven-field task works without adaptation.

Inspect image preprocessing visually, including orientation, aspect ratio,
padding, and text size. Record actual GPU/RAM availability; the Phase 3 machine
had a GTX 1650 with 3.64 GiB visible memory. A's inference alone did not demonstrate
training feasibility; B's [resource evidence](../experiments/donut-feasibility.md)
now records the full-step investigation and failures.

Checkpoint: explain the image-to-sequence path and identify every input the
inference function may receive. References and identity metadata are evaluation
data, not model prompts.

### Workshop 1 — Targets, data, and structural correctness

In `05b`, serialize exactly the names in `schema.REQUIRED_FIELDS`, in a fixed
order. A proposed format uses a task token and paired field delimiters:

```text
<s_alb_id><s_document_number>EXAMPLE</s_document_number>...
<s_sex>F</s_sex></s>
```

The ellipsis illustrates omitted fields; real targets contain all seven fields.
Freeze actual tokens, value representation, escaping, field order, and parser
behavior in versioned configuration before training. Derive targets from
annotations under the existing normalization rules and retain original values.
Register schema tokens explicitly; do not learn field vocabularies from test.

Proposed initial training data: one canonical template and one rectified photo
for each of 70 training identities (140 observations). Validation similarly has
30 observations from 15 identities. Verify actual coverage before asserting
these counts. Both views remain together and are not independent identities.
Start without a synthetic augmentation search.

Measure target token lengths on allowed splits and reject truncation. Test
serialization round trips and parsing of missing/extra/duplicate keys, unmatched
delimiters, empty values, invalid dates, missing end tokens, and length cutoffs.
Choose and freeze the duplicate-key policy before evaluating candidates.
Unparseable output must reach the shared scorer as a structural failure; do not
silently repair it or drop the observation. Preserve sequence/token evidence.

Checkpoint: manually trace one image/target pair and explain how padding and
the task prompt affect the training labels and loss.

### Workshop 2 — Training feasibility and a tiny overfit check

Before a full run, measure a complete forward/backward/optimizer step at batch
size one, with all intended trainable parameters and optimizer state present.
Record precision, processor resolution, target length, peak allocated/reserved
GPU memory, CPU memory, and elapsed time. Loading or inference alone is not a
training-feasibility result.

Try a small, declared sequence of adjustments only as needed: bounded processor
resolution with readable text, gradient checkpointing, supported mixed precision,
or freezing part of the model. Gradient accumulation changes effective batch
size but cannot make a single oversized example fit. Report failures and every
changed setting. Freezing parameters changes the adaptation experiment and must
be identified explicitly. Do not replace the seven fields or identity split to
fit memory; changing frozen scope requires a contract change.

If a useful configuration cannot fit, record the blocker and a concrete bounded
alternative before committing to a long run. No cloud spending or distributed
training is implied by this plan.

In `05c`, deliberately overfit a few training examples to check the labels,
gradients, vocabulary, parser, and free generation. Falling teacher-forced loss
alone is insufficient. Reproducing those examples diagnoses implementation; it
does not demonstrate generalization. Reset to the pinned starting checkpoint
before the main run.

Checkpoint: explain why inference fitting in memory does not imply training
will fit, and why tiny-set memorization is useful only as a sanity check.

### Workshop 3 — Bounded adaptation and selection

Before the main run, freeze the seed, data manifest, trainable parameters,
precision, optimizer, learning rate, batch/accumulation, epoch/step budget,
validation cadence, early-stopping rule, and generation policy.

Proposed selection rule: maximize the mean of template/photo validation macro
Field EM, break ties by pooled CER, then retain the earlier candidate. This
matches the Phase 3 objective and must be recorded before candidate selection.
Evaluate free-running sequences through the actual parser and shared scorer;
validation loss is diagnostic. Keep an explicit experiment budget rather than
an open-ended hyperparameter search.

Save processor/tokenizer, extended embeddings, model, configs, histories, and
sample-level predictions together. Verify an offline reload reproduces results.
Inspect constant nationality separately as a diagnostic; it remains in the
primary score. Record name/number regressions and valid-looking wrong outputs.

Checkpoint: distinguish training loss, validation selection, and final test
evaluation, and explain why a well-formed sequence can still be incorrect.

### Workshop 4 — Comparison, source extraction, and freeze

In `05d`, score all 15 validation identities in both views: 30 observations and
210 required fields, retaining failed generations in denominators. Report macro
and per-field EM, corpus/per-field CER, Document EM, schema failures, and raw
counts separately for templates and photos. Inspect paired OCR/Donut failures
as development diagnostics; the full complementarity/oracle benchmark remains
Phase 5 work.

Rerun adopted OCR and selected Donut under matched current timing conditions;
do not treat an old OCR session as a controlled speed comparator. Batch size is
one; keep models resident individually if necessary. Freeze and record warm-ups,
repetitions, request order, hardware, threads, precision, and synchronization.
Measure generation separately and the complete path from in-memory canonical
pixels through preprocessing, decoding, parsing, normalization, and validation.
Report mean/median and memory. Loading, file I/O, reference scoring, and artifact
writing remain outside request latency. Concurrency percentiles come later.

Once behavior is stable, extract simple Donut/serialization code into
`src/kyc_router/`. Reuse schema, normalization, validation, and evaluation;
avoid a new framework. Test structural failure handling and inference boundaries,
then compare source predictions/tokens and scores with retained notebook output.
Inference receives pixels and fixed policy, never identity or reference text.

Checkpoint: explain the observed accuracy/cost tradeoffs without assuming Donut
must beat OCR. A verified negative result is useful research evidence; an
untrained or unverified path is not a completed Phase 4 pipeline.

## Documentation and artifact plan

The artifacts below now exist with their distinct responsibilities. Historical
configurations remain labeled in the reports; v2 is the selected training policy.

| Location | Responsibility |
|---|---|
| `docs/learning/ocr-free.md` | Initial architecture, serialization, training, and failure concepts; extend or split into focused learning guides when needed |
| `docs/experiments/donut-feasibility.md` | Candidate/API checks, memory/step-time evidence, failed configurations, and chosen feasible policy |
| `docs/experiments/donut-finetuning.md` | Data, training/selection protocol, named runs, learning curves, validation predictions, reload, limitations |
| `docs/experiments/ocr-donut-comparison.md` | Paired validation metrics, matched latency measurements, failure gallery, and comparison limits |
| `docs/guides/donut-usage.md` | Environment/artifact recovery, inference/scoring API, tests, and reproduction |
| `docs/phases/phase-4.md` | Final delivered checklist, adopted decision, verification, limitations, and Phase 5 handoff |
| `configs/donut_finetuning_v2.json` | Actual selected Adafactor training policy, written before main training; v1 retains the failed AdamW diagnostic |
| `configs/donut_pipeline_v1.json` | Adopted artifact revisions/hashes, input/processor/tokenizer/parser/decoding policy, shared rules, timing protocol |
| `checkpoints/phase4_*/`, `outputs/phase4_*/` | Ignored local weights, raw runs, predictions/tokens, environment, counts, timings, and audits |

Every executed investigation gets a report, including unsuccessful experiments.
Keep historical runs intact. Update documentation navigation, architecture,
usage, and project status with actual delivery. Do not create empty reports or
invent a frozen configuration before its evidence exists.

## Deliverables checklist

- [x] Notebook A: guided foundations with all cells executed, saved outputs,
      artifact provenance, learning documentation, and a complete experiment report.
- [x] Executed foundations, data, training, and evaluation notebooks.
- [x] Documented feasible training configuration with measured resource evidence.
- [x] Leakage-safe seven-field dataset/serialization and explicit parser policy.
- [x] Fine-tuned candidate selected on validation, saved and verified after reload.
- [x] Complete paired validation accuracy, failure, latency, and memory evidence.
- [x] Versioned training and adopted inference manifests with artifact provenance.
- [x] Reusable whole-image inference path sharing field rules and evaluation.
- [x] Focused tests and notebook-to-source prediction/scoring parity.
- [x] Learning guides, experiment reports, usage guide, delivery record, and
      updated entry points.
- [x] A frozen comparison-ready path for Phase 5, with test identities still sealed.

The final audit passed. The handoff is Phase 5's controlled
benchmark with both frozen paths. This workshop does not start that benchmark
or authorize retuning from test outcomes.

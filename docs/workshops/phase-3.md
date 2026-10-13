# Phase 3 OCR Pipeline — Workshop Plan

**Status:** Engineering delivery complete (2026-10-12), including Workshop 10.
The learner delegated the remainder of Workshop 4, Workshops 5–8, Workshop 9,
and finally authorized Phase 3 closure, documentation organization, and reusable
source extraction. The [delivery record](../phases/phase-3.md) records adoption
and verified parity. This does not certify learner mastery: reflection and
understanding checks remain available learning work, not completed assessments.
See the [Workshop 5–8 evidence report](../experiments/ocr-baseline.md) and
[Workshop 9 results](../experiments/ocr-finetuning.md).

## 1. Purpose

Phase 3 will produce a crop-based OCR baseline for the seven required `alb_id`
fields. Its equally important purpose is to teach how an OCR pipeline works,
how its parts are evaluated, and how experimental decisions are kept honest.

By the end, the learner should be able to explain and implement:

- the difference between text detection, text recognition, and structured
  extraction;
- how an image crop becomes model input, tokens, and decoded text;
- why raw recognition, normalization, semantic validation, and scoring are
  separate stages;
- how Exact Match, Character Error Rate, Document Exact Match, and latency are
  calculated and interpreted;
- how to diagnose OCR failures without leaking information from the test set;
- when fine-tuning is justified and when it is premature.

The practical output remains the Phase 3 contract:

```text
canonical RGB document
    -> seven known field crops
    -> selected text recognizer (pretrained baseline or justified fine-tune)
    -> raw predictions
    -> shared normalization
    -> simple semantic validation
    -> canonical seven-field result
    -> accuracy and latency measurements
```

## 2. How the workshop will work

The default teaching mode is that the learner writes and runs every notebook
cell. For the explicitly delegated Workshops 5–8, the guide authors and executes
the notebook, retaining explanations, worked examples, and reflection prompts;
this does not imply that the learner has completed the mastery checks. The
original delegation stopped after Workshop 8 without training or closing Phase 3.
The learner subsequently authorized the guide to own and execute Workshop 9 in
an extra notebook, with comprehensive learning documentation. That authorization
does not close Phase 3 or establish that the learner has completed its mastery
checks. The subsequent explicit closure authorization supersedes that earlier
boundary; the original teaching plan below remains useful as learning history.

In the default teaching mode, the guide provides the next
small objective, explains the relevant concepts, reviews the learner's code and
observations, and helps debug problems. The guide does not silently build the
notebooks or hand over a completed pipeline.

Each workshop follows the same learning loop:

1. State the question the next experiment should answer.
2. Build a mental model of the relevant concept.
3. Predict what the code or output should do.
4. Write and run a small notebook cell.
5. Inspect actual values, shapes, images, timings, or errors.
6. Explain the result in the learner's own words.
7. Turn the experiment into a small reusable function only after it is
   understood.
8. Finish with a short checkpoint and written reflection.

Notebook markdown is part of the work, not decoration. It should record the
question, reasoning, result, interpretation, and limitation of each experiment.

### Proposed notebook organization

The phase is split into focused notebooks instead of one very large notebook:

```text
notebooks/04a_ocr_foundations.ipynb
notebooks/04b_ocr_pipeline.ipynb
notebooks/04c_ocr_evaluation.ipynb
notebooks/04d_ocr_finetuning.ipynb       # optional; created only if justified
```

The exact filenames can be changed during plan review. Stable code will first
emerge inside these notebooks. We will consider moving it into `src/` only at
the end of the phase, when its interface and behavior are understood.

## 3. Experimental guardrails

These rules apply throughout the workshop:

- Use only `alb_id` and the seven frozen fields.
- Use the canonical 1084 x 680 RGB input and Phase 2 crop geometry.
- Use training identities for learning, debugging, and any fitting.
- Use validation identities for checkpoint/decoding selection and the Phase 3
  baseline report.
- Do not inspect or score test identities during Phase 3 development.
- Start with one pretrained printed-text recognizer; do not compare a large
  collection of OCR libraries.
- Preserve raw predictions. Normalization must never erase evidence of model
  errors.
- Keep synthetic-degradation benchmarking for the later benchmark phase. Phase
  3 may use tiny training-split examples only to understand behavior, not to
  produce the final degradation study.
- Do not fine-tune until the pretrained baseline has been measured and its
  failures classified.
- Freeze the exact model checkpoint, revision, decoding settings, software
  versions, and latency protocol before later evaluation.

## 4. Workshop sequence

### Workshop 0 — Orientation and data-flow audit

**Notebook:** `04a_ocr_foundations.ipynb`

We will first reconstruct the complete recognition path on paper and connect it
to the artifacts from Phases 1 and 2. We will distinguish what is already
solved—document corners, rectification, and field locations—from the new OCR
problem.

Topics:

- detection versus recognition versus structured extraction;
- template images versus rectified photographs;
- raw MIDV field names versus canonical output names;
- training, validation, and sealed test responsibilities;
- image arrays, RGB/BGR, crop geometry, and ground truth;
- inspecting the available CPU/GPU and recording the environment.

Checkpoint: draw and explain the complete data flow, identify what information
the recognizer may use, and load one permitted identity with its seven crops and
ground-truth strings.

### Workshop 1 — Understand and select the pretrained recognizer

**Notebook:** `04a_ocr_foundations.ipynb`

We will learn the minimum transformer concepts needed to use a TrOCR-style
recognizer intelligently. Selection will be based on task fit, documentation,
resource requirements, character coverage, and reproducibility—not test-set
performance.

Topics:

- what “pretrained” means and why domain mismatch matters;
- encoder, autoregressive decoder, tokens, and generated sequences;
- image processor versus tokenizer/decoder;
- checkpoint identifiers, revisions, licenses, and model cards;
- CPU/GPU device placement, inference mode, and numerical precision;
- greedy decoding versus other decoding settings;
- recording one exact checkpoint before the baseline experiment.

Checkpoint: explain what happens between a crop and a decoded string, justify
one checkpoint, and record the chosen model configuration.

### Workshop 2 — First single-crop inference

**Notebook:** `04a_ocr_foundations.ipynb`

We will deliberately run the smallest useful experiment: one clean crop from a
training identity through the chosen model.

Topics:

- converting the RGB NumPy crop into the model's expected input;
- inspecting input image size, tensor shape, dtype, range, and device;
- inference without gradient tracking;
- generated token IDs and decoded raw text;
- model loading time versus per-request inference time;
- why a plausible result is not yet evidence of pipeline quality.

We will then repeat the experiment across fields with different characteristics:
a name, a date, the document number, nationality, and the one-character sex
field.

Checkpoint: trace one prediction through every representation and explain at
least two observed successes or failures without changing scoring rules.

### Workshop 3 — Build the seven-field OCR pipeline

**Notebook:** `04b_ocr_pipeline.ipynb`

We will turn the single-crop experiment into a small, inspectable pipeline for
one canonical card. Each function will have one responsibility.

Topics:

- loading annotations and selecting an identity safely by split;
- reusing Phase 2 rectification and crop behavior;
- iterating over all seven fields without losing field identity;
- mapping MIDV names into the canonical schema;
- separating model setup from request-time inference;
- designing explicit inputs and return values;
- assertions, useful failures, and avoiding hidden notebook state;
- retaining raw predictions and per-field inference information.

The target interface is conceptually:

```python
result = ocr_pipeline(canonical_rgb, template_record)
```

Checkpoint: produce all seven raw fields for clean templates and rectified
photographs from a few training identities, then explain each pipeline stage
and its assumptions.

### Workshop 4 — Normalization and semantic validation

**Notebook:** `04b_ocr_pipeline.ipynb`

We will implement the shared normalization contract as a distinct layer. The
same functions must later process OCR predictions, Donut predictions, and
ground truth.

Topics:

- Unicode NFC, whitespace normalization, and case folding;
- preserving diacritics and meaningful punctuation;
- field-specific document-number normalization;
- explicit date parsing and canonical `YYYY-MM-DD` output;
- normalization versus error correction;
- validators for dates, date ordering, nationality, sex, and document-number
  shape;
- why a validation failure is information rather than permission to rewrite a
  prediction.

We will create small hand-written examples and edge cases before applying these
functions to model output.

Checkpoint: predict and verify normalization/validation results for normal,
malformed, missing, Unicode, and misleading values. Explain why every accepted
transformation is harmless for scoring.

### Workshop 5 — Learn and implement OCR metrics

**Notebook:** `04c_ocr_evaluation.ipynb`

Before evaluating the model, we will compute tiny metric examples by hand and
then reproduce them in code.

Topics:

- per-field Exact Match;
- macro Field Exact Match and why fields receive equal weight;
- Levenshtein edit distance and Character Error Rate;
- Document Exact Match;
- missing, malformed, and extra predictions;
- raw versus normalized evaluation;
- numerators, denominators, and the effect of a small validation set;
- unit-style checks for metric functions.

Checkpoint: manually calculate the metrics for a small artificial prediction
table, make the implementation match, and explain what each metric reveals or
hides.

### Workshop 6 — Run the validation baseline

**Notebook:** `04c_ocr_evaluation.ipynb`

We will run the first reproducible OCR benchmark without opening the test set.
Clean templates and real rectified photographs will be reported separately so
that synthetic cleanliness is not confused with capture robustness.

Topics:

- building one auditable row per identity/source observation;
- storing raw and normalized predictions;
- per-field correctness, edit counts, schema validity, and model metadata;
- aggregating without losing sample-level evidence;
- comparing template and real-capture results;
- separating debugging observations from reported validation measurements;
- saving enough metadata to recalculate results later.

Checkpoint: produce a validation table with per-field Exact Match, macro Field
Exact Match, CER, and Document Exact Match, plus raw counts.

### Workshop 7 — Measure latency correctly

**Notebook:** `04c_ocr_evaluation.ipynb`

We will measure cost as carefully as accuracy.

Topics:

- inference latency versus end-to-end pipeline latency;
- excluding model and dataset loading;
- warm-up runs and repeated measurements;
- synchronization for GPU timing;
- batch size one as the primary contract;
- mean and median latency and why both are retained;
- per-field calls versus one document-level request;
- recording hardware, precision, versions, and measurement settings.

Checkpoint: report reproducible inference and end-to-end latency for the
seven-field request and explain exactly what each timer includes.

### Workshop 8 — Failure analysis and the fine-tuning decision

**Notebook:** `04c_ocr_evaluation.ipynb`

We will inspect failures before attempting to improve the model. Errors will be
classified by likely source rather than treated as one accuracy number.

Initial taxonomy:

- crop or rectification problem;
- model preprocessing problem;
- character substitution, insertion, or deletion;
- spacing, punctuation, case, or Unicode issue;
- short-field or single-character failure;
- systematic field/font/domain mismatch;
- source information genuinely erased by capture conditions.

We will compare representative crops, raw predictions, normalized predictions,
and ground truth. The decision gate is:

```text
Is there a repeated recognition failure that cannot be fixed honestly by
geometry, preprocessing, or the already-approved normalization contract?
```

If no, we freeze the pretrained baseline. If yes, we write a specific
fine-tuning hypothesis and proceed to the optional workshop. A lower-than-hoped
score alone is not a sufficient training plan.

Checkpoint: produce a small failure table/gallery, quantify the main error
categories, and make a written fine-tune-or-freeze decision.

### Optional Workshop 9 — Fine-tune without leakage

**Notebook:** `04d_ocr_finetuning.ipynb`

**Execution record:** implemented and run under the subsequent delegation; see
the [experiment report](../experiments/ocr-finetuning.md) and
[concept guide](../learning/ocr-finetuning.md). At the Workshop 9 handoff, selected
weights remained a candidate for joint review. Their subsequent adoption is
recorded in the [Phase 3 delivery record](../phases/phase-3.md).

This notebook will exist only if Workshop 8 justifies it. Before training, we
will freeze the training-data policy, augmentation policy, checkpoint-selection
rule, and compute budget.

Topics:

- constructing crop/transcription pairs from training identities only;
- dataset and data-loader responsibilities;
- token labels, padding, and ignored loss positions;
- teacher forcing and sequence cross-entropy at an intuitive level;
- train/eval modes, gradients, optimization, and learning rate;
- overfitting risks with 70 identities;
- validation-based checkpoint selection;
- comparing the fine-tuned model with the untouched pretrained baseline;
- retaining negative results.

Checkpoint: either freeze a demonstrably justified fine-tuned checkpoint or
retain the pretrained baseline with a documented reason.

### Workshop 10 — Freeze the pipeline and close Phase 3

**Notebook:** final section of `04c_ocr_evaluation.ipynb`, or `04d` if used.

We will audit the work against the experimental contract and make later phases
able to reuse it.

Closure artifacts:

- exact checkpoint and revision;
- fixed preprocessing and decoding settings;
- canonical schema, normalization, and validation behavior;
- callable seven-field pipeline interface;
- validation accuracy and latency tables;
- sample-level predictions for audit and error analysis;
- hardware and software environment record;
- known limitations and fine-tuning decision;
- a Phase 3 delivery record in `docs/`;
- stable code extracted from notebooks only where reuse now justifies it.

Final understanding check: explain the complete OCR pipeline without reading
the notebook, reproduce one prediction manually stage by stage, and describe
which changes would invalidate or require rerunning the benchmark.

## 5. Guidance style during implementation

The guide will normally provide:

- the immediate goal and why it matters;
- the concept needed for that goal;
- a small coding prompt or pseudocode rather than a finished notebook section;
- questions about expected shapes, types, and outputs before execution;
- review of the learner's code and interpretation;
- progressively stronger hints when blocked;
- a concise summary only after the learner has attempted the step.

Direct code may still be given for unfamiliar boilerplate, obscure APIs, or
after a learner attempt, but it will be explained line by line and kept small
enough to understand. Debugging will identify the cause, not merely provide a
replacement cell.

## 6. Progress and mastery checks

Completion is not based only on cells running. At each checkpoint, the learner
should be able to:

- predict important behavior before execution;
- interpret outputs instead of only displaying them;
- explain the purpose and boundary of each function;
- distinguish evidence from assumptions;
- identify leakage and reproducibility risks;
- change a small requirement without rewriting the whole pipeline;
- diagnose a deliberately introduced mistake such as BGR/RGB confusion,
  incorrect normalization, or timing model loading.

If a checkpoint is unclear, we pause and use a smaller experiment rather than
building more code on top of a weak mental model.

## 7. Estimated effort

For a learning-first pace:

| Work | Focused time |
|---|---:|
| Workshops 0–2 | 6–9 hours |
| Workshops 3–4 | 7–10 hours |
| Workshops 5–7 | 8–12 hours |
| Workshops 8 and 10 | 4–6 hours |
| Optional fine-tuning | 12–25 additional hours |

The expected baseline phase is approximately **25–37 focused hours**, usually
four to seven weeks at a relaxed part-time pace. Understanding takes priority
over the calendar; repeated experiments and debugging are part of the workshop,
not schedule failures.

## 8. Approval questions

The following were the original pre-Workshop-0 review questions, retained as
planning history. They are not a new approval gate for the delegated work:

1. The proposed split into three main notebooks plus one optional notebook.
2. The depth and teaching rhythm described above.
3. Whether stable code should be extracted into `src/` at Phase 3 closure or
   remain notebook-only until Phase 5.
4. The proposed scope boundary: validation-only Phase 3 reporting, with the test
   split kept sealed for the later frozen comparison.

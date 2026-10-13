# Phase 3 — Workshops 5–8 evidence and handoff

**Historical handoff: Phase 3 remained open after Workshops 5–8.** At that
handoff, Workshop 9 was pending the user's decision; no training or test-set
evaluation had been performed. Final freezing and Workshop 10 had not happened.

**Subsequent work:** the user authorized Workshop 9 training and documentation.
Its [fine-tuning report](ocr-finetuning.md) records the executed experiment.
This document retains the original pretrained-baseline evidence. At that
handoff, Phase 3 was still open and Workshop 10 remained pending. For current
status, see the [Phase 3 delivery record](../phases/phase-3.md).

Date: 2026-10-12. Main artifact:
[04c_ocr_evaluation.ipynb](../../notebooks/04c_ocr_evaluation.ipynb).
Scope follows the [workshop plan](../workshops/phase-3.md),
[roadmap](../../PROJECT.md), and [experimental contract](../../EXPERIMENT_DESIGN.md).

## What is implemented

- Workshop 5: a hand-worked example; per-field, macro/micro Field EM, Document
  EM, Levenshtein edits, and corpus CER; missing/malformed/extra-output checks;
  duplicate/completeness assertions; and normalized versus raw comparisons.
- Workshop 6: all 15 validation identities, separately reported as templates
  and rectified photographs; raw/normalized predictions, tokens, crop geometry,
  validators, checkpoint/settings, environment, and input hashes retained.
- Workshop 7: synchronized GPU inference and canonical-input end-to-end timers;
  warm-up exclusion; repeated shuffled requests; raw and aggregated timings.
- Workshop 8: every failed crop reviewed across six gallery pages; actual
  processor inputs audited; observed symptoms distinguished from likely causes;
  a small training-only preprocessing control; a fine-tuning hypothesis.

The notebook contains explanatory Markdown and review prompts. Delegated
execution is not recorded as learner mastery. Longer material lives in
[OCR evaluation](../learning/ocr-evaluation.md) and
[TrOCR generation and timing](../learning/ocr-generation-and-timing.md), including
attention, causal/cross-attention, token generation, caching, dynamic programming,
aggregation, and asynchronous GPU timing.

## Configuration measured

| Setting | Value |
|---|---|
| Checkpoint | `microsoft/trocr-small-printed` |
| Exact revision | `04e994ab854b0089d4929f48c2b4dbe2ce78a340` |
| Input | 1084×680 RGB card; seven Phase 2 crops |
| Crop padding | x 10%, at least 12 px; top 10%; bottom 40% |
| Model input | One float32 `[1, 3, 384, 384]` tensor per crop |
| Decoding | Greedy; one beam; max total length 20; no sampling |
| Decoder KV caching | Disabled, preserving the `04b` checkpoint configuration |
| Normalization / validation | Existing `v1` / `alb_id_v1` rules |
| Sources | 15 templates + 15 rectified photographs, paired by identity |
| Split | Validation only for reported baseline; test remains sealed |

Model and processor loaded from the pinned local cache. No decoding search,
post-hoc nationality replacement, separator repair, or identity exclusion was
used. Known boxes/corners isolate recognition from localization. Test records
inside the shared dataset annotation containers were not selected, scored, or
displayed; test images were not opened.

## Correctness

There are 105 required field observations per source. Macro and micro Field EM
coincide because every field contributes exactly 15 observations.

| Measurement | Templates | Rectified photos |
|---|---:|---:|
| Raw field matches | 43/105 (40.95%) | 55/105 (52.38%) |
| Normalized field matches / macro Field EM | 63/105 (60.00%) | 76/105 (72.38%) |
| Corpus CER | 158/911 (17.34%) | 46/911 (5.05%) |
| Document EM | 0/15 (0%) | 0/15 (0%) |
| Schema failures | 0 | 0 |
| Generation length-limit hits | 0 | 0 |

| Field | Template EM | Photo EM | Template CER | Photo CER |
|---|---:|---:|---:|---:|
| document_number | 12/15 (80.00%) | 15/15 (100%) | 2.22% | 0% |
| surname | 9/15 (60.00%) | 11/15 (73.33%) | 9.78% | 5.43% |
| given_names | 11/15 (73.33%) | 10/15 (66.67%) | 5.05% | 7.07% |
| date_of_birth | 7/15 (46.67%) | 14/15 (93.33%) | 40.00% | 4.00% |
| expiry_date | 11/15 (73.33%) | 13/15 (86.67%) | 2.67% | 1.33% |
| nationality | 0/15 (0%) | 0/15 (0%) | 27.78% | 8.89% |
| sex | 13/15 (86.67%) | 13/15 (86.67%) | 13.33% | 13.33% |

The approved normalization resolves **41** raw-only mismatches: 20 template and
21 photo fields. **71** field errors remain. Nationality fails in all 30
documents, which by itself forces complete-document accuracy to zero.

Photos outperform templates here, particularly on birth dates and nationality
character errors. This is observed behavior of this checkpoint and these pixels,
not evidence that capture generally helps recognition. Font/background detail,
resampling, and model preprocessing interact; that cause was not isolated.

Malformed dates also explain part of the CER gap: a valid reference becomes
year-first, while an unsupported prediction such as `23:08-1993` remains
day-first. CER counts both the separator error and the representation-order
difference. The raw text shows that the template digits themselves were retained.
The scoring contract was preserved rather than weakened to repair those outputs.

With 15 identities per source, one changed field outcome moves its field EM by
6.67 percentage points. A perfect 15/15 is a result on this workload, not a
general reliability guarantee. The two sources are paired observations, not 30
independent identities.

## Warm latency on the GTX 1650

One request is **all seven fields**, processed sequentially at batch size one.
Three complete warm-up requests per source were discarded. All 30 observations
were then measured three times in seeded shuffled order: **45 requests per
source**, 90 total. All six warm-ups and all 90 measured predictions matched the
accuracy baseline exactly.

| Timing (ms per seven-field request) | Templates | Rectified photos |
|---|---:|---:|
| Inference mean | 260.80 | 269.14 |
| Inference median | 260.55 | 269.58 |
| End-to-end mean | 268.61 | 278.23 |
| End-to-end median | 268.39 | 277.40 |
| End-to-end standard deviation | 4.10 | 11.00 |
| End-to-end observed range | 261.08–277.41 | 267.00–346.53 |

Inference is the sum of synchronized `model.generate` intervals, including
encoder and decoder. End-to-end starts with a preloaded canonical RGB image and
includes crops, processor work, device transfer, generation, decoding,
normalization, and prediction validation. It excludes loading, rectification to
the shared canonical input, reference scoring, and file writing. It is therefore
not raw-camera-to-result service latency. Nested synchronization overhead is
included; no concurrency or tail-latency conclusions are drawn.

Nationality is the slowest field: mean inference **47.37 ms** for templates and
**54.44 ms** for photos, with mean generated sequence suffix lengths of 8 and 10
tokens respectively. Sex averages roughly **29.3 ms** even for one character,
because it still incurs an encoder pass and generation overhead. Decoder caching
remains a possible separate speed experiment, not an implemented improvement.

Hardware: Intel i5-12400, 12 logical CPUs, 31.08 GiB system RAM; GTX 1650,
3.64 GiB visible GPU memory; driver 615.71.09; CUDA runtime 13.0. Float32;
PyTorch 2.14.1+cu130, TorchVision 0.29.1+cu130, Transformers 5.19.0, OpenCV 5.0.0,
Python 3.14.7. PyTorch CPU threads 4, interop threads 6, OpenCV threads 1.
Peak PyTorch allocated/reserved GPU memory during the timing workload was about
0.250/0.266 GiB. Other desktop processes and existing Python kernels occupied GPU
memory; this was a local desktop run, not an exclusive-device benchmark.

## Failures and their likely sources

All **71** failures were visually reviewed, along with representative processor
inputs for every failing field/source combination. The processor receives RGB
float32 tensors with the expected shape and finite values. The usual square
resize visibly changes aspect ratio; it is inherited model preprocessing, not
by itself proof of a pipeline bug.

| Diagnostic group | Template failures | Photo failures | Interpretation |
|---|---:|---:|---|
| Nationality | 15 | 15 | Complete, readable text; repeated wrong character sequences on both sources. Strong recognition/domain-mismatch candidate. |
| Birth dates | 8 | 1 | Template first-separator substitutions; photo year character error. Tight upper crop alignment in photo `alb_id/10` may contribute. |
| Expiry dates | 4 | 2 | Five `202x` → `201x` year errors and one day substitution. Every erroneous output remains a valid date. |
| Document numbers | 3 | 0 | Two substitutions and one deletion in readable template crops. |
| Names and surnames | 10 | 9 | Character substitutions/insertions/deletions, including a diacritic. Repeated cross-source failures; blur and nearby label fragments may contribute in photos. |
| Sex | 2 | 2 | Visible `F` transcribed as `5` or `E`, including clean templates. |

For sex, `M` is correct in 8/8 observations per source, while `F` is correct in
5/7 per source. The aggregate 13/15 hides that asymmetry. These samples are too
few to support a broad demographic-performance claim, but the character-specific
failure is actionable evidence.

Examples:

- `alb_id/14`, template expiry: `30-08-2028` → `30-08-2018`. It parses and remains
  after birth date, so semantic validation cannot detect the wrong year.
- `alb_id/32`, template number: `618170573` → `61817573`, a deletion caught by
  the number-length rule. Other number substitutions keep nine digits and pass.
- `alb_id/36`, both sources given names: `Arbërie` → `ARDERIE`. Case folding helps
  representation but must preserve the missing diacritic/character error.
- Nationality on most templates: `Shqiptare/Albanian` → `SNAPTARE/ABANGAN`.
  Most photos instead produce a closer but still wrong `SHGIPTARE/ALBANIAN`.

**27 wrong fields pass semantic validation**: 16 templates and 11 photos.
Another 44 wrong fields fail it. All 139 correct fields pass. Validation detects
some errors; it does not certify transcription correctness.

No confirmed irrecoverable information loss was needed to explain these
validation failures. Some photos are blurred and crop alignment remains a
possible contributor. Phase 2's known glare-erased fields were not added as
extra evaluation examples; they must not be repaired by guessing.

### Training-only preprocessing control

To test one simple alternative before recommending adaptation, the same
recognizer processed native crops centered on white squares, preserving aspect
ratio before the fixed processor resize. It was compared with the original
stretch on `alb_id/02`, `/03`, `/04`, and `/05`, all training identities. The
selection was the first three training identities plus the first training `F`.
Four fields × two sources × four identities give 32 field observations per
variant. This is diagnosis, not a validation or degradation benchmark.

The original processor scored **18/32**; white-square padding scored **8/32**.
Nationality remained **0/8** for both. Padding frequently yielded only hyphens,
and dates/names worsened. That particular aspect-ratio intervention did not solve
the failures. Other preprocessing changes remain possible; the control does not
prove that fine-tuning is the sole remedy. No baseline settings were changed.

## Recommendation for the joint Workshop 9 decision

There is enough repeated failure on readable text to justify **considering a
bounded fine-tuning experiment**. The hypothesis is specific: adaptation on
training identity-field crop/transcription pairs may reduce printed-font,
separator, short-field, and character errors that persist across sources.

If approved, use only the 70 training identities, retain exact labels and
diacritics, select checkpoints using validation macro Field EM, and compare
every field with this untouched baseline under the same evaluation contract.
Choose an explicit compute budget before training. Because nationality is
constant, fixing it alone may reflect memorization; improved names, dates, and
numbers would provide stronger evidence of useful adaptation. Crop/preprocessing
limitations must remain visible, and capture-erased text cannot be reliably
recovered by training alone.

At this handoff, this was a recommendation, not a final fine-tune-or-freeze decision. No `04d`
notebook, training job, selected fine-tuned checkpoint, source extraction, Phase 4
implementation, synthetic robustness study, or Phase 3 closure was performed.

## Reproduction and artifacts

Executed run: `20261012T220250890693Z`. Local artifacts are under
[`outputs/phase3/20261012T220250890693Z/`](../../outputs/phase3/20261012T220250890693Z/)
(ignored by Git; the executed notebook retains the displayed tables/galleries).

Useful files: `run_metadata.json`, `field_predictions.jsonl`,
`document_predictions.jsonl`, `per_field_accuracy.csv`, `accuracy_summary.csv`,
`latency_measurements.csv`, `field_latency_measurements.csv`, `latency_summary.csv`,
`failures.csv`, `paired_outcomes.csv`, `training_preprocessing_control.csv`, and
the PNG galleries. Metadata includes exact generation/model/processor settings,
software/hardware, split and config hashes, validation image/canonical-pixel
hashes, timing policy, and training-control scope.

Run from the repository root with the validated project environment and GPU
access:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  notebooks/04c_ocr_evaluation.ipynb --ExecutePreprocessor.timeout=900
```

A rerun creates a fresh artifact directory. The report describes the recorded
run; timing values will vary on later runs. The notebook explicitly includes the
reused `04b` function definitions and their source hashes, avoiding hidden state.

Verification: **42 code cells executed from a clean kernel, zero error outputs**;
metric and edge-case assertions passed; all serialized accuracy aggregates
recalculated identically; all repeated GPU predictions matched; notebook format
validated. An initial run before adding the training-only control yielded the
same 210 validation predictions. Subsequent notebook changes add interpretation
Markdown only; they do not alter the executed code or measured results.
An independent recursive oracle also verified 225 short-string edit-distance
cases; a deliberately removed scored row was correctly rejected. All 25 reused
code cells match `04b` exactly. `audit_manifest.json` records hashes of the final
notebook, report, and local run artifacts.

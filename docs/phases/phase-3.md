# Phase 3 — OCR pipeline delivery

**Complete: 2026-10-12 (local date).** The user authorized Workshop 10 closure,
documentation organization, and the first reusable source implementation.
Engineering delivery is complete; learner reflection prompts are not marked as
passed merely because experiments and tests ran.

## Adopted decision

Adopt **`joint_epoch_05`**, the selected fine-tuned TrOCR checkpoint, as **`ocr_v1`**.
The [frozen manifest](../../configs/ocr_pipeline_v1.json) identifies its parent
revision, local checkpoint, file hashes, preprocessing, generation, shared field
rules, split provenance, and latency protocol. The historical
[candidate manifest](../../configs/ocr_finetuning_candidate_v1.json) and training
configuration remain unchanged snapshots of the earlier experiment.

This follows the already declared objective: maximize the mean of template/photo
macro Field EM, break ties by pooled CER, then retain the earlier candidate.
The untouched pretrained model remains the comparator; no new training or
selection search was performed for closure.

The gain is substantial, but not universal. **Photo document-number accuracy
regressed from 15/15 to 13/15.** Adoption explicitly accepts this tradeoff under
the existing equal-field objective; it is not a claim that every field improved
or that this model is operationally safe for identity verification. No lookup
correction, per-field ensemble, fallback, or scoring-rule change hides the errors.

## Accuracy retained and reproduced

Validation contains 15 identities, each with template and rectified-photo views:
30 observations and 210 required fields. The two views are not 30 independent
identities. All validation fields remain included.

| Model | Source | Correct fields / macro EM | Corpus CER | Complete documents |
|---|---|---:|---:|---:|
| Pretrained | template | 63/105 — 60.00% | 158/911 — 17.34% | 0/15 |
| Pretrained | photo | 76/105 — 72.38% | 46/911 — 5.05% | 0/15 |
| Adopted | template | 98/105 — 93.33% | 11/911 — 1.21% | 9/15 |
| Adopted | photo | 96/105 — 91.43% | 17/911 — 1.87% | 8/15 |

Overall normalized field correctness is **194/210 (92.38%)**, versus
139/210 (66.19%). There are 60 fixes, five regressions, and 16 remaining errors.
Excluding the constant nationality field, correctness is **164/180 (91.11%)**,
versus 139/180 (77.22%). This diagnostic shows gains beyond nationality
memorization; it does not establish cross-country recognition.

| Field | Adopted template matches | Adopted photo matches |
|---|---:|---:|
| document_number | 14/15 | 13/15 |
| surname | 11/15 | 11/15 |
| given_names | 13/15 | 13/15 |
| date_of_birth | 15/15 | 14/15 |
| expiry_date | 15/15 | 15/15 |
| nationality | 15/15 | 15/15 |
| sex | 15/15 | 15/15 |

The [fine-tuning report](../experiments/ocr-finetuning.md) retains every remaining
error, regressions, field CER, training history, and galleries. Twelve errors are
name fields, three are document numbers, and one is a date. **14/16 wrong fields
still pass semantic validation.** Well-formed output is not confidence or proof
of correctness. No calibrated confidence score is invented for this delivery.

## Frozen behavior and input boundary

- One `alb_id` canonical **1084 × 680 RGB uint8** image, plus seven known
  canonical field rectangles. The caller supplies canonical pixels; document
  detection, image loading, and rectification are outside recognition.
- Native template geometry is scaled by separately rounding x/y/width/height.
  Crops use horizontal 10% padding with a 12-pixel minimum, top 10%, bottom 40%,
  clipped to the image, preserving the notebook's copied crop bytes.
- Parent model: `microsoft/trocr-small-printed`, revision
  `04e994ab854b0089d4929f48c2b4dbe2ce78a340`. Adopted local weights:
  `checkpoints/phase3_finetuning/20261012T230738117335Z/joint/best`.
- Float32; seven sequential batch-one field calls; greedy decoding,
  `max_length=20`, `max_new_tokens=None`, `use_cache=False`; special token IDs
  frozen in the manifest. Decode skips special tokens and disables cleanup.
- Shared v1 normalization preserves diacritics and meaningful punctuation,
  uses NFC/whitespace/casefold, explicit date formats, number whitespace removal,
  and sex uppercase. Malformed dates remain visible and invalid, never guessed.
- Separate simple `alb_id` validators check shape, allowed nationality/sex,
  calendar dates, and expiry after birth. They do not correct predictions.

Inference accepts no identity or reference text. The small VIA geometry adapter
does not read `value` transcriptions. References and split metadata remain in
experiments/evaluation. The equal-input, split, metric, and normalization rules
in the [experimental contract](../../EXPERIMENT_DESIGN.md) are unchanged.

## Reusable source and verification

The installable [`src/kyc_router`](../../src/kyc_router/) package contains schema,
geometry, normalization, validation, OCR, and evaluation modules. Functions handle
stateless steps; one concrete `OcrPipeline` owns the loaded model. No abstract
framework or future-phase placeholder is introduced. See the
[architecture decision](../architecture.md) and [usage guide](../guides/ocr-usage.md).

Final verification run **`20261013T004540259201Z`** (UTC; October 12 locally):

- **35 unit tests passed** without MIDV or a GPU: field/date/Unicode/missing-value
  behavior, geometry/copies, scoring denominators, sequential mocked inference,
  bad inputs, and verified offline loading failures.
- **210/210 crop geometry and byte comparisons** matched the notebook reference.
- **210/210 raw strings and generated token sequences** matched the selected
  checkpoint reload records; normalized scores and all document validators
  matched too. Source summaries reproduced the table above.
- **90/90 timed repeats** retained exact predictions and token sequences.
- Independent scoring replay also reproduced **2,520 saved validation field
  records** across the 12 retained pretrained/epoch/reload files.
- A real CPU smoke request also matched the GPU strings, tokens, and normalized
  fields for one validation template; this is not a full CPU benchmark.
- Editable packaging and a regular wheel were built without replacing existing
  libraries; the wheel imported independently from outside the checkout. Package
  imports do not execute notebooks, load datasets/models, select a GPU, or change
  thread pools. The migration check alone executes explicitly listed historical
  notebook definitions on its reference side; it does not run training cells.

The reproducible check is
[`experiments/verify_ocr_source.py`](../../experiments/verify_ocr_source.py).
Local verification, predictions, CSVs, environment, source/notebook code hashes,
and artifact hashes are in
[`outputs/phase3_closure/20261013T004540259201Z/`](../../outputs/phase3_closure/20261013T004540259201Z/).
Its [final audit manifest](../../outputs/phase3_closure/20261013T004540259201Z/audit_manifest.json)
records unit tests, link/notebook checks, wheel/CPU checks, and current file hashes.
The Workshop 10 appendix in [04d](../../notebooks/04d_ocr_finetuning.ipynb)
explains the extraction and displays the recorded checks without retraining.

## Latency and environment

Source timing uses the same warm-up/repetition policy as the fine-tuning
comparison: three warm-up documents per source, three shuffled repetitions of
all 30 observations, 45 measured requests per source. CUDA generation/request
boundaries synchronize. Timings begin with canonical pixels in memory; loading,
rectification, reference scoring, and output writing are excluded.

| Source | Generation mean / median | End-to-end mean / median |
|---|---:|---:|
| template | 279.37 / 279.37 ms | 287.16 / 287.21 ms |
| photo | 279.02 / 278.32 ms | 286.86 / 286.05 ms |

These are warm local measurements, not service latency or evidence of a speedup
over the previous ~297 ms session. The runs were not counterbalanced replications.
The official policy did not adopt the resource lab's KV cache, result/prepared
caches, or field batching. Its [report](../experiments/ocr-resource-profile.md)
remains useful explanatory evidence, not a replacement load benchmark.

Recorded machine: Intel i5-12400, 12 logical CPUs, 31.08 GiB RAM, GTX 1650
(3.64 GiB visible), CUDA 13.0; Python 3.14.7, torch 2.14.1+cu130,
torchvision 0.29.1+cu130, Transformers 5.19.0. The verification used four PyTorch
threads, six interop threads, and one OpenCV thread; the package does not impose
these globally. Exact measured direct dependencies are in
[`requirements/phase3-recorded.txt`](../../requirements/phase3-recorded.txt),
explicitly an environment record rather than a portable full lockfile.

## Documentation and evidence inventory

The [documentation map](../README.md) indexes every maintained document:

| Location | Purpose and delivered documents |
|---|---|
| Root | `PROJECT.md`: scope/roadmap/status; `EXPERIMENT_DESIGN.md`: authoritative rules; `README.md`: entry point |
| `docs/phases/` | `phase-2.md`: input-preparation delivery; `phase-3.md`: this adoption/closure record |
| `docs/experiments/` | `ocr-baseline.md`: pretrained results; `ocr-finetuning.md`: adaptation/results/failures; `ocr-resource-profile.md`: resource side experiment |
| `docs/learning/` | `OCR.md`: foundations; `ocr-evaluation.md`: metrics; `ocr-generation-and-timing.md`: transformers/generation/timing; `ocr-finetuning.md`: training operations; `ocr-resource-profiling.md`: transfers/caches/quotas/queues |
| `docs/workshops/` | `phase-3.md`: approved sequence, delegation history, learning checks |
| `docs/guides/` | `ocr-usage.md`: setup, artifact recovery, API, tests and reproduction |
| `docs/architecture.md` | Package boundaries and the simple design choices, with primary references |
| Notebooks `04a`–`04e` | Foundations; pipeline; evaluation; fine-tuning + closure; resource lab |
| `configs/` | Historical training/candidate/profiling policies; new `ocr_pipeline_v1.json` is the current freeze |

Earlier reports moved from the docs root into these folders; links were rebased.
Notebook research code/outputs were preserved, with Markdown navigation/history
notes and the appended Workshop 10 section. Historical audit manifests remain
untouched snapshots: their old report paths/document hashes do not describe the
reorganized tree. New closure evidence records current source hashes instead of
retroactively rewriting past audits.

Weights, full raw runs, and MIDV data remain local and Git-ignored. A fresh clone
can run unit tests after installing dependencies, but cannot infer with the
adopted checkpoint without restoring its exact hashed artifacts. No upload,
publication, staging, or commit was performed as part of this delivery.

## Limits and next boundary

This is development evidence on one mock layout with only 15 paired validation
identities, repeated selection, one training seed, constant nationality, known
corners, and known field geometry. It does not establish unseen-layout, real-KYC,
or synthetic-degradation robustness. Document-number regressions remain visible.

**Test is still sealed. Phase 4 is not started.** Next work is the OCR-free
workshop and model path, reusing these field rules and metrics. No Donut, router,
quality features, synthetic benchmark sweep, predicted localizer, HTTP endpoint,
distributed worker, queue, or deployment layer was implemented here.

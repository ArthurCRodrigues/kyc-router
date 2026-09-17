# Adaptive KYC Pipeline — Project Overview & Roadmap

## Current project status

As of 2026-09-16:

- **Phase 0 — Define the Experiment: complete.** The authoritative v1
  decisions are recorded in [`EXPERIMENT_DESIGN.md`](EXPERIMENT_DESIGN.md).
- **Phase 1 — Explore MIDV-2020: complete for the initial experiment.** The
  executable investigation and its outputs are in
  [`notebooks/01_explore_midv.ipynb`](notebooks/01_explore_midv.ipynb).
- **Phase 2 — Image Processing and Rectification: complete.** See the
  [Phase 2 delivery record](docs/phase-2.md) for the outputs, checks, and
  known capture limitations.
- **Next phase: Phase 3 — Build the OCR Pipeline.**

This file remains the broad roadmap. When an older provisional statement here
differs from the frozen v1 experiment, `EXPERIMENT_DESIGN.md` is authoritative.

## 1. Project Summary

This project studies a simple **identity-document processing pipeline** with **adaptive model routing**.

The system will receive an image of an identity document and return structured fields such as:

```json
{
  "document_number": "...",
  "surname": "...",
  "given_names": "...",
  "date_of_birth": "...",
  "expiry_date": "...",
  "nationality": "...",
  "sex": "..."
}
```

The interesting part is not only extracting the fields. We want to compare two different inference paths:

1. a **modular OCR-based pipeline**, and
2. an **OCR-free document model**, initially Donut.

Later, we will build a small decision model that learns when each path should be used.

The final experiments will study how this routing decision changes under:

- image degradation;
- model confidence;
- throughput;
- concurrent requests;
- p50 / p95 / p99 latency;
- limited compute resources.

The project is intentionally designed for **consumer hardware** such as a gaming PC or MacBook. It is a learning and research project, not a production KYC platform.

---

# 2. Main Research Question

A useful working research question is:

> Can an identity-document processing system selectively route documents between a lightweight OCR pipeline and a more expensive OCR-free model in order to preserve field-extraction accuracy while reducing expensive inference and maintaining reasonable latency under load?

We are **not** assuming that Donut will be more accurate or more robust than OCR.

Discovering when each approach succeeds or fails is part of the experiment.

---

# 3. Scope

## Included

The project focuses on **identity-document processing**:

- locating the document in an image;
- rectifying its perspective;
- extracting important identity fields;
- validating extracted values;
- evaluating robustness to degraded images;
- choosing between inference pipelines;
- measuring inference latency and throughput.

## Not included initially

To keep the project realistic, the first version will **not** attempt:

- face matching;
- liveness detection;
- biometric verification;
- NFC reading;
- hologram verification;
- forgery detection;
- AML checks;
- sanctions screening;
- Kubernetes;
- distributed inference;
- large-scale production infrastructure.

These can be considered future work, but they are not required for the main research question.

---

# 4. Core System Idea

A simplified final architecture may look like this:

```text
                    Identity-document image
                              |
                              v
                    Image-quality features
                              |
                              v
                           Router
                         /        \
                        /          \
                       v            v
              OCR pipeline      OCR-free model
                       \            /
                        \          /
                         v        v
                       Normalized fields
                              |
                              v
                     Semantic validation
                              |
                              v
                            JSON
```

Later, system load may also become an input to the router:

```text
image features ----\
model confidence ---+--> router --> OCR / Donut
queue depth --------/
```

The router should not be told the true synthetic degradation label during real inference.

For example, the benchmark may know that an image is `blur severity 3`, but the real router must work with measurable signals such as:

- blur score;
- brightness;
- contrast;
- fraction of very dark pixels;
- fraction of nearly saturated bright pixels;
- OCR confidence;
- validation failures;
- later, queue depth or system utilization.

---

# 5. Dataset

## Primary dataset: MIDV-2020

MIDV-2020 will be the main dataset.

It provides identity-document images and annotations useful for:

- document localization;
- text fields;
- field values;
- document geometry;
- scans and photographs;
- multiple capture conditions.

For the beginning of the project, we should work with:

- templates;
- still photographs;
- optionally upright/rotated scans.

We should **ignore videos initially**.

They add complexity that does not help answer the first research questions.

## Initial document scope

The first working prototype uses **`alb_id`**, a planar Albanian identity-card
layout from MIDV-2020. Phase 1 confirmed that it has 100 identities, complete
annotations for the seven selected fields, no apparent vertical text among
those fields, and a relatively large, straightforward layout.

Once the entire experiment works for one document type, we can expand to additional identity-card types.

This keeps the early project manageable.

---

# 6. Initial Experimental Contract

Phase 0 is complete. The detailed and authoritative v1 contract is
[`EXPERIMENT_DESIGN.md`](EXPERIMENT_DESIGN.md). The summary below reflects its
main decisions; changes to a frozen decision must be documented in that file.

## Input

A still image containing one identity document.

## Initial rectification

During the first OCR/OCR-free experiments, use MIDV's **ground-truth document corners**.

This isolates recognition from localization.

```text
MIDV photograph
      |
ground-truth corners
      |
perspective rectification
      |
canonical document
     / \
    /   \
 OCR   Donut
```

Later we replace the ground-truth corners with our own localizer and measure the complete end-to-end pipeline.

## Required v1 fields

Both pipelines return exactly these seven required fields:

- document number;
- surname;
- given names;
- date of birth;
- expiry date;
- nationality;
- sex.

The `alb_id` raw-field mapping is:

```text
number       -> document_number
surname      -> surname
name         -> given_names
birth_date   -> date_of_birth
expiry_date  -> expiry_date
nationality  -> nationality
gender       -> sex
```

There are no optional output fields in v1. `photo`, `face`, and `signature` are
retained as non-textual annotations but are not extraction targets.
`id_number`, `issue_date`, `birth_place`, and `authority` are excluded from the
minimal v1 output.

A field that is absent from a document type must be distinguished from a field that exists but was extracted incorrectly.

## Data split

Use this exact identity-level split:

```text
70 identities training
15 identities validation
15 identities test
```

The split seed is `2026`, and the final assignment will be stored in a
version-controlled manifest. The important rule is **split by underlying
document identity**.

All versions derived from one identity must remain in the same split.

For example:

```text
document_042 clean
document_042 blur_1
document_042 blur_2
document_042 low_resolution
```

must never be distributed across training and test sets.

## Controlled degradations

The controlled v1 benchmark uses four degradation families:

1. blur;
2. downsampling / low resolution;
3. JPEG compression;
4. exposure / brightness, with underexposure and overexposure reported
   separately.

Use severity:

```text
0 = original
1 = very mild
2 = mild
3 = medium
4 = strong
5 = severe
```

The exact parameters for each level will be calibrated and frozen in Phase 2
using only training and validation identities. Only one synthetic degradation
is applied at a time.

Perspective distortion is deferred because it mixes recognition robustness
with rectification/localization error. It will later be studied separately by
perturbing document corners.

These degradations are generated programmatically, so their type and severity are known during benchmarking.

## Real capture conditions

Clean templates are the base for controlled synthetic degradations. Rectified
MIDV photographs separately test whether conclusions transfer to real capture
conditions.

Controlled corruptions and real captures should be treated as different experiments.

## Accuracy metrics

Primary:

**Macro Field Exact Match**

A field is correct only when its normalized prediction equals the normalized ground truth.

Normalization may include harmless changes such as:

- trimming whitespace;
- normalizing case;
- standardizing date formats.

Secondary:

**Character Error Rate (CER)**

CER helps distinguish a one-character mistake from a completely wrong prediction.

Also report **Document Exact Match**:

> Percentage of documents for which every required field is correct.

## Performance metrics

Initially record both:

- model inference latency per request;
- end-to-end pipeline latency per request.

Later add:

- p50 latency;
- p95 latency;
- p99 latency;
- throughput;
- queue latency;
- CPU/GPU utilization;
- memory usage.

## Main optimization objective

Avoid inventing a complicated weighted score.

The first routing goal is:

> Remain no more than one absolute percentage point below the stronger fixed
> pipeline's Macro Field Exact Match while minimizing how often Donut is used.

Later we can add latency constraints.

---

# 7. Roadmap

---

## Phase 0 — Define the Experiment

**Status: Complete (2026-09-15).**

The resulting authoritative artifact is
[`EXPERIMENT_DESIGN.md`](EXPERIMENT_DESIGN.md).

### Goal

Agree on what is being measured before implementation begins.

### Tasks

- [x] Define project scope.
- [x] Choose MIDV-2020.
- [x] Define the canonical output schema.
- [x] Define initial accuracy metrics.
- [x] Decide how train/validation/test separation works.
- [x] Choose initial controlled degradations.
- [x] Define OCR-based vs OCR-free comparison.
- [x] Decide that ground-truth corners will initially isolate recognition from localization.
- [x] Define the main routing objective.

### What we should have in hand

The broad roadmap in this document and the frozen v1 decisions in
[`EXPERIMENT_DESIGN.md`](EXPERIMENT_DESIGN.md).

We should also have a clear statement such as:

> Both pipelines receive the same rectified document image and produce the same normalized identity-field schema.

### Expected result

We know exactly what a successful experiment looks like.

No model needs to exist yet.

### Technologies

Almost none.

Useful tools:

- Markdown;
- Git;
- GitHub.

### Done when

We can explain:

- what goes into the system;
- what comes out;
- what models are being compared;
- how success is measured.

---

# Phase 1 — Explore MIDV-2020

**Status: Complete for v1 (2026-09-15).**

The completed investigation is
[`notebooks/01_explore_midv.ipynb`](notebooks/01_explore_midv.ipynb). It loads
and visualizes VIA annotations, measures field coverage and missing-value
states, inspects image and document geometry, decodes capture conditions,
defines the identity key, checks common annotation problems, and compares the
five planar ID-card candidates.

Completed outcomes:

- [x] Confirmed the `templates` and `photo` directory and annotation structure.
- [x] Parsed and visualized VIA rectangles and document quadrilaterals.
- [x] Measured field availability and distinguished absent, missing, empty, and
  non-empty annotation states.
- [x] Inspected image dimensions, modes, formats, and relative document area.
- [x] Documented capture-condition encoding and identity/condition confounding.
- [x] Defined the leakage-safe `document_type/sample_id` identity key.
- [x] Compared the five planar card types and selected `alb_id`.
- [x] Froze the seven required v1 fields.
- [x] Measured common annotation and geometry sanity conditions.

## Goal

Understand the dataset before building abstractions around it.

This phase should be exploratory.

A Jupyter notebook is preferable to immediately creating complicated dataset classes.

## Tasks

### 1. Download a small useful subset

Start with:

- `templates`;
- `photo`.

Optionally later:

- upright scans;
- rotated scans.

Do not download/use videos yet unless convenient.

### 2. Understand directory structure

Explore:

- document types;
- image naming;
- annotation files;
- relationship between templates and captured images.

### 3. Understand annotations

Learn how MIDV represents:

- document quadrilaterals;
- text field bounding boxes;
- field names;
- ground-truth values.

### 4. Visualize annotations

Write notebook code that can:

- display an image;
- draw the document boundary;
- draw field rectangles;
- print the value associated with a field.

### 5. Analyze available fields

Count fields across:

- documents;
- document types.

This lets us choose the canonical project schema based on the dataset rather than intuition.

### 6. Select first document type

Selected: `alb_id`.

### 7. Decide the final v1 field list

Selected from actual annotation frequency and card semantics: document number,
surname, given names, date of birth, expiry date, nationality, and sex.

## What we should have in hand

A notebook such as:

```text
notebooks/01_explore_midv.ipynb
```

that demonstrates:

```python
sample.image
sample.document_corners
sample.fields
```

even if those are not yet wrapped in a clean class.

The notebook includes field-count and coverage tables equivalent to:

| Field | Number of documents | Number of document types |
|---|---:|---:|
| document_number | ... | ... |
| birth_date | ... | ... |
| surname | ... | ... |

## Expected result

MIDV is no longer a mysterious directory of files.

We understand how a sample maps to:

```text
image
annotations
document geometry
field geometry
field values
```

We selected `alb_id` and the seven required v1 fields recorded in the
experimental contract.

## Technologies

- Python;
- Jupyter;
- NumPy;
- Pandas;
- matplotlib;
- Pillow;
- basic OpenCV.

No PyTorch training is required.

## What to learn

Only the basics needed to manipulate:

- files;
- JSON;
- NumPy arrays;
- images;
- coordinates;
- simple plots.

## Done when

Given a MIDV image, we can:

1. load it;
2. load its annotation;
3. draw its document boundary;
4. display its text fields;
5. retrieve ground-truth field values.

---

# Phase 2 — Image Processing and Rectification

**Status: Complete (2026-09-16).** The delivered artifacts and validation are
summarized in the [Phase 2 delivery record](docs/phase-2.md). The frozen
calibration values live in the [versioned configuration](configs/controlled_degradations_v1.json).

## Goal

Learn the computer-vision operations that turn an arbitrary document photograph into a normalized document image.

## Main pipeline

```text
photograph
    |
document corners
    |
perspective transform
    |
rectified document
    |
field crops
```

Initially, document corners come directly from MIDV ground truth.

## Tasks

### 1. Learn essential OpenCV concepts

Work with:

- image loading;
- BGR vs RGB;
- resizing;
- grayscale;
- drawing;
- cropping;
- geometric coordinates.

### 2. Rectify documents

Given four known document corners:

- define destination rectangle;
- compute a homography/perspective transform;
- warp the photograph into a canonical card.

Important functions:

```python
cv2.getPerspectiveTransform(...)
cv2.warpPerspective(...)
```

### 3. Normalize document dimensions

Choose one standard output resolution per template/document type.

For example:

```text
rectified card -> fixed width × height
```

The selected dimensions are recorded in the versioned configuration linked
above and apply to both extraction paths.

### 4. Crop known fields

Once the card is rectified, use template annotations to extract field images such as:

```text
surname crop
date-of-birth crop
document-number crop
```

### 5. Implement controlled degradation functions

Create simple functions:

```python
apply_blur(image, severity)
apply_downsampling(image, severity)
apply_jpeg(image, severity)
apply_exposure(image, severity)
```

Keep these deterministic/reproducible where possible.

Apply controlled degradations independently to clean canonical template images.
Underexposure and overexposure are recorded separately. Perspective distortion
is deferred to a later corner-perturbation experiment because it belongs to the
localization/rectification boundary rather than the initial recognition study.

### 6. Build a very simple localizer baseline later

After the recognition experiment works, try an OpenCV-based document localizer:

```text
grayscale
 -> edges
 -> contours
 -> large quadrilateral
 -> four corners
```

This does not need to be sophisticated.

A learned detector can be future work if the simple baseline is insufficient.

## What we should have in hand

Reusable functions similar to:

```python
rectify_document(image, corners)
crop_field(rectified_image, bbox)
apply_degradation(image, kind, severity)
```

The investigations are in:

```text
notebooks/02_rectification.ipynb
notebooks/03_controlled_degradations.ipynb
```

showing several before/after examples.

## Expected result

We can take a MIDV photograph and produce:

```text
original photograph
        ->
clean rectified card
        ->
individual field crops
```

This becomes the shared input preparation for later model experiments.

## Technologies

Primary:

- OpenCV;
- NumPy;
- Pillow;
- matplotlib.

Optional for augmentation:

- Albumentations.

Avoid complex computer-vision frameworks for now.

## Done when

For the selected document type:

- rectification visually works;
- field crops contain the expected text;
- synthetic degradations can be generated at multiple severities.

---

# Phase 3 — Build the OCR Pipeline

## Goal

Create the first complete system that turns a rectified identity document into structured fields.

This is the **modular** path.

## Simplified architecture

```text
rectified document
       |
known field positions
       |
field crops
       |
OCR recognizer
       |
text normalization
       |
structured fields
       |
semantic validation
```

## Important simplification

Do **not** begin by training OCR.

First test a pretrained OCR/text-recognition model.

Training should happen only if experiments show that it is needed.

## Tasks

### 1. Select a pretrained OCR recognizer

A reasonable first experiment is TrOCR or another simple pretrained recognizer.

We do not need to compare ten OCR libraries.

Pick one, make it work, and establish a baseline.

### 2. Run OCR on individual field crops

Example:

```text
[ image containing "12/04/1999" ]
                  |
                 OCR
                  |
             "12/04/1999"
```

### 3. Normalize predictions

Implement simple functions for:

- whitespace;
- case;
- dates;
- common formatting differences.

### 4. Map predictions to the canonical schema

Return something like:

```python
{
    "surname": "...",
    "given_names": "...",
    "date_of_birth": "...",
    "document_number": "..."
}
```

### 5. Add simple semantic validators

Start with easy deterministic rules:

```text
date parses correctly
expiry > date of birth
document number matches expected format
known nationality code
```

Do not build a complicated rules engine.

### 6. Measure accuracy

Calculate:

- exact-match accuracy per field;
- overall field accuracy;
- CER.

### 7. Measure latency

Record how long OCR inference takes.

At this stage, simple Python timing is enough.

### 8. Fine-tune only if necessary

If pretrained OCR fails systematically, then consider fine-tuning on MIDV/synthetic field crops.

Fine-tuning is an experiment, not a requirement for the first baseline.

## What we should have in hand

An interface similar to:

```python
prediction = ocr_pipeline(rectified_document)
```

returning:

```python
{
    "fields": {...},
    "confidence": {...},
    "latency_ms": ...
}
```

Confidence can be basic initially.

## Expected result

The first end-to-end field-extraction baseline.

We should be able to produce a table such as:

| Field | Exact Match | CER |
|---|---:|---:|
| surname | ... | ... |
| birth_date | ... | ... |
| document_number | ... | ... |

This is our first real model benchmark.

## Technologies

Likely:

- Python;
- PyTorch;
- Hugging Face Transformers;
- TrOCR or a similar pretrained recognizer;
- OpenCV;
- NumPy/Pandas.

Avoid:

- custom CUDA;
- distributed training;
- complicated training infrastructure.

## Done when

For a rectified MIDV document, the OCR pipeline can return structured identity fields and we can measure whether they are correct.

---

# Phase 4 — Build the OCR-Free Pipeline

## Goal

Build a second pipeline that solves approximately the same field-extraction problem using an end-to-end document model.

Initial candidate: **Donut**.

## Conceptual difference

OCR path:

```text
document
 -> crops
 -> OCR
 -> parsed fields
```

OCR-free path:

```text
document image
 -> Donut
 -> structured sequence
 -> fields
```

"OCR-free" does not mean the model does not read text.

It means there is no separately executed OCR engine in the architecture.

## Tasks

### 1. Understand Donut input/output

Learn how to:

- preprocess an image;
- tokenize target output;
- run inference;
- decode generated text.

### 2. Define a target sequence

For example:

```text
<surname>SMITH</surname>
<given_names>JOHN</given_names>
<birth_date>1999-04-12</birth_date>
```

Keep the schema small.

### 3. Build training examples

Convert MIDV annotations into:

```text
input = rectified document image
target = serialized fields
```

### 4. Start from a pretrained model

Do **not** train Donut from scratch.

Fine-tune a pretrained checkpoint.

### 5. Keep training small

Use:

- one/few document types initially;
- modest image resolution;
- small batches;
- early stopping/checkpoints;
- consumer GPU if available.

If hardware is too limited, reduce scope rather than making training infrastructure complicated.

### 6. Evaluate using exactly the same fields as OCR

Both systems must output the same normalized schema.

### 7. Record inference latency

The OCR-free pipeline must be evaluated not just for accuracy but also cost/latency.

## What we should have in hand

An interface similar to:

```python
prediction = donut_pipeline(rectified_document)
```

that returns the same field schema as the OCR pipeline.

## Expected result

Two comparable inference paths:

```text
same input
  |
  +--> OCR pipeline ----> fields
  |
  +--> Donut -----------> fields
```

We should now know their basic:

- accuracy;
- latency;
- failure behavior.

## Technologies

- PyTorch;
- Hugging Face Transformers;
- Donut;
- Pillow/OpenCV;
- Pandas;
- optionally TensorBoard or simple CSV logging.

We do **not** need:

- distributed training;
- MLflow;
- Kubernetes;
- multi-node GPUs.

A CSV file and a few plots are enough.

## Done when

The OCR-free path can process the same test samples as OCR and produce directly comparable outputs.

---

# Phase 5 — Build the Benchmark Dataset

## Goal

Run both pipelines against the same controlled samples and record what happens.

This dataset becomes the foundation for routing.

## Tasks

### 1. Generate evaluation samples

For each test document, evaluate:

```text
clean
blur 1..5
downsampling 1..5
JPEG 1..5
underexposure 1..5
overexposure 1..5
```

Each sample receives only one degradation at a time. Exact parameters are
calibrated on training and validation identities in Phase 2 and frozen before
test evaluation.

### 2. Run both pipelines

Every sample must be evaluated by both pipelines offline.

### 3. Record input-quality features

Examples:

- blur score;
- mean brightness;
- contrast;
- fraction of very dark pixels;
- fraction of nearly saturated bright pixels.

These are what a future router can actually observe.

Do not give the router the true synthetic degradation label unless we are explicitly running an oracle/diagnostic experiment.

### 4. Record pipeline outcomes

Example row:

```text
sample_id
document_type
degradation
severity

blur_score
mean_brightness
contrast
dark_pixel_fraction
bright_pixel_fraction

ocr_correct
ocr_field_accuracy
ocr_confidence
ocr_latency_ms

donut_correct
donut_field_accuracy
donut_latency_ms
```

### 5. Build the oracle

For every sample, determine which pipeline would have been preferable if we magically knew the outcomes beforehand.

For example:

```text
OCR correct and cheaper -> OCR
OCR wrong, Donut correct -> Donut
both correct -> cheapest
both wrong -> neither succeeds
```

The oracle is **not deployable**.

It tells us the theoretical opportunity available to a router.

### 6. Analyze complementarity

Important questions:

- On how many samples does OCR succeed and Donut fail?
- On how many does Donut succeed and OCR fail?
- How often do both succeed?
- How often do both fail?
- Does one pipeline dominate the other?

If one pipeline is always both faster and more accurate, routing is not useful.

That itself would be an important experimental result.

## What we should have in hand

A CSV/Parquet file such as:

```text
results/benchmark.csv
```

plus plots showing:

```text
accuracy vs degradation severity
latency vs model
OCR vs Donut success matrix
```

## Expected result

We now have empirical evidence about where each path is useful.

This phase determines whether the adaptive-routing hypothesis is promising.

## Technologies

- Python;
- Pandas;
- NumPy;
- matplotlib;
- scikit-learn metrics where useful.

No serving system is needed yet.

## Done when

Every test sample has comparable OCR and Donut outcomes and we can compute the oracle-routing result.

---

# Phase 6 — Build the Router

## Goal

Learn when to choose each pipeline.

Do not begin with neural routing.

Start with the simplest possible baselines.

## Experiments

### Router 0 — Always OCR

```text
every request -> OCR
```

### Router 1 — Always Donut

```text
every request -> Donut
```

### Router 2 — Simple threshold

Example:

```python
if blur_score > threshold:
    use_donut()
else:
    use_ocr()
```

### Router 3 — Simple learned model

Features might include:

```text
blur score
brightness
contrast
dark-pixel fraction
bright-pixel fraction
```

Candidate models:

- a depth-limited Decision Tree;
- optionally, Logistic Regression as an additional simple baseline.

The depth-limited Decision Tree is the required learned baseline because it is
small and easy to interpret. Neural routers and large ensembles are unnecessary
for v1.

### Router 4 — Confidence-aware cascade

Later:

```text
run OCR
   |
OCR confidence / validation
   |
high confidence -> accept
low confidence  -> Donut
```

This may outperform a router that only looks at image quality, but Donut fallbacks now pay both OCR and Donut cost.

That tradeoff is part of the experiment.

## Target variable

A simple formulation is:

```text
choose OCR
or
choose Donut
```

Training labels can be created from benchmark outcomes.

However, the objective should reflect cost as well as correctness.

For the first implementation, keep it simple:

> Prefer OCR when it is correct; choose Donut when OCR is wrong and Donut is correct.

Then evaluate how often this decision agrees with the oracle.

## Metrics

Compare routers on:

- field accuracy;
- percentage routed to Donut;
- average latency;
- oracle agreement;
- failure rate.

## What we should have in hand

A function such as:

```python
decision = router.predict(features)
```

returning:

```text
OCR
```

or:

```text
DONUT
```

## Expected result

A plot/table resembling:

| Strategy | Field Accuracy | Donut Usage | Mean Latency |
|---|---:|---:|---:|
| Always OCR | ... | 0% | ... |
| Always Donut | ... | 100% | ... |
| Threshold | ... | ... | ... |
| Learned Router | ... | ... | ... |
| Oracle | ... | ... | ... |

The important result is the **tradeoff**, not simply "router accuracy."

## Technologies

- scikit-learn;
- Pandas;
- NumPy;
- matplotlib.

A neural network is unnecessary unless simple models clearly fail.

## Done when

We have a routing policy that can be compared against:

- always OCR;
- always Donut;
- simple threshold;
- oracle.

---

# Phase 7 — Introduce System Load and Latency Constraints

## Goal

Move from offline model selection to a small serving experiment.

This is where the project gains its systems component.

## Important principle

Keep the serving architecture simple.

We do not need Kubernetes or a distributed cluster.

One application running on one consumer machine is enough to study queueing and latency.

## Initial architecture

```text
HTTP request
      |
      v
   FastAPI
      |
      v
    Router
    /    \
   OCR   Donut
    \     /
     v   v
   response
```

## Tasks

### 1. Expose one inference endpoint

For example:

```text
POST /extract
```

Request:

```text
image
```

Response:

```json
{
  "fields": {},
  "route": "ocr",
  "latency_ms": 0
}
```

### 2. Generate concurrent traffic

Use a simple load-testing tool.

Candidate:

- Locust;
- k6;
- or a custom async Python script.

Start simple.

### 3. Measure latency distribution

Record:

- p50;
- p95;
- p99.

Also separate:

```text
queue/wait time
model inference time
end-to-end latency
```

when practical.

### 4. Measure throughput

Requests per second completed successfully.

### 5. Increase concurrency

Example experiment:

```text
1 concurrent request
2
4
8
16
...
```

Stop when the machine clearly saturates.

### 6. Observe queueing

The expensive path may cause requests to wait even if its individual model inference time does not change.

This is why tail latency matters.

### 7. Add load information to routing

A later router may receive:

```text
image difficulty
OCR confidence
queue depth
recent latency
```

Do not begin with a complicated scheduler.

A rule is enough:

```python
if difficult_image and expensive_queue_is_small:
    use_donut()
else:
    use_ocr()
```

Then compare it against the load-unaware router.

## What we should have in hand

A small local service and benchmark scripts.

Example outputs:

```text
results/load_test_1.csv
results/load_test_4.csv
results/load_test_16.csv
```

## Expected result

Final experiments can answer:

- Does adaptive routing reduce expensive inference?
- Does it preserve accuracy?
- What happens as request concurrency rises?
- How do p95/p99 behave?
- Does considering system load improve tail latency?
- What accuracy is sacrificed, if any?

## Technologies

Start with:

- FastAPI;
- Python;
- PyTorch;
- Locust or k6;
- psutil;
- optional GPU monitoring if CUDA is available.

Only consider ONNX/TensorRT/Triton **after the basic experiments work**, and only if they solve a concrete problem.

They are not requirements.

## Done when

We can produce graphs such as:

```text
accuracy vs Donut usage
accuracy vs throughput
accuracy vs p99 latency
p99 latency vs concurrency
routing distribution vs concurrency
```

---

# Phase 8 — Final Experiments and Write-Up

## Goal

Turn the implementation into a coherent research result.

## Main comparisons

At minimum:

```text
Always OCR
Always Donut
Quality-threshold router
Learned router
Confidence-aware router (if implemented)
Load-aware router
Oracle
```

Not every advanced router is mandatory.

If time is limited, prioritize:

```text
Always OCR
Always Donut
Simple Router
Learned Router
Oracle
```

and one load-aware extension.

## Ablations

Possible simple questions:

- What happens without blur information?
- What happens without OCR confidence?
- Does routing help under clean images?
- Does routing help more under moderate degradation?
- At which degradation severity do both pipelines fail?
- How much does routing change p99 under load?

## Failure analysis

Inspect examples where:

- OCR fails but Donut succeeds;
- Donut fails but OCR succeeds;
- both fail;
- router selects the wrong path.

A handful of visual examples can be more informative than another complicated model.

## Final expected result

A final report should be able to make a statement in the form:

> Under the tested MIDV-2020 workload, adaptive routing retained approximately X of the stronger pipeline's field accuracy while reducing expensive-path usage by Y and changing p95/p99 latency by Z under the tested load.

The exact outcome may show routing is useful, only useful in specific degradation regimes, or not useful with the chosen models.

All are valid research findings.

---

# 8. Suggested Repository Structure

Keep this small initially:

```text
adaptive-kyc/
├── README.md
├── PROJECT_ROADMAP.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── midv2020/
│
├── notebooks/
│   ├── 01_explore_midv.ipynb
│   └── 02_rectification.ipynb
│
├── src/
│   ├── data/
│   ├── image/
│   ├── ocr/
│   ├── donut/
│   ├── routing/
│   └── evaluation/
│
├── experiments/
│
└── results/
```

Do not create all modules on day one.

Let the structure grow as code becomes reusable.

---

# 9. Technology Summary

| Area | Initial technology |
|---|---|
| Language | Python |
| Exploration | Jupyter |
| Images | OpenCV, Pillow |
| Arrays/data | NumPy, Pandas |
| Plots | matplotlib |
| OCR model | pretrained TrOCR or similarly simple OCR |
| OCR-free model | Donut |
| Deep learning | PyTorch |
| Model library | Hugging Face Transformers |
| Router | scikit-learn |
| HTTP serving | FastAPI |
| Load testing | Locust / k6 |
| Version control | Git + GitHub |

The guiding rule is:

> **Add technology only when the experiment requires it.**

---

# 10. What We Intentionally Avoid

This project can easily become too ambitious.

Avoid prematurely adding:

- Docker orchestration;
- Kubernetes;
- Redis queues;
- Kafka;
- Triton;
- Ray;
- custom CUDA kernels;
- distributed training;
- reinforcement learning;
- large neural routing models;
- complicated MLOps stacks.

These technologies may be interesting, but they are not necessary for answering the research question.

If the simple system produces meaningful results, it is already a successful project.

---

# 11. Six-Month Rough Schedule

This is a guide, not a deadline for each individual implementation.

| Period | Focus | Status |
|---|---|---|
| Weeks 1–2 | Phase 0–1: contract + MIDV exploration | Complete |
| Weeks 3–5 | Phase 2: image processing and rectification | Current |
| Weeks 6–9 | Phase 3: OCR pipeline | Not started |
| Weeks 10–14 | Phase 4: OCR-free pipeline | Not started |
| Weeks 15–17 | Phase 5: degradation benchmark | Not started |
| Weeks 18–20 | Phase 6: router | Not started |
| Weeks 21–23 | Phase 7: load/latency experiments | Not started |
| Week 24 | Phase 8: final analysis and write-up | Not started |

There should be flexibility.

The OCR-free model may consume more time than expected, while the OCR pipeline may be easier.

If Donut fine-tuning becomes disproportionately difficult on available hardware, reduce model/dataset scope rather than sacrificing the entire routing and systems experiment.

---

# 12. Immediate Next Step

We are currently at:

```text
Phase 3 — Build the OCR Pipeline
```

The next concrete objective is:

> Run a pretrained printed-text recognizer on the seven crops produced from
> the frozen canonical RGB input, then measure field accuracy and latency.

Use the [Phase 2 delivery record](docs/phase-2.md) for the completed input
preparation and known real-capture limitations. Phase 3's detailed scope and
evaluation rules are in [`EXPERIMENT_DESIGN.md`](EXPERIMENT_DESIGN.md).

The next milestone is:

```text
canonical rectified card
    |
    v
seven field crops
    |
    v
OCR predictions in the canonical seven-field schema
```

---

# 13. Useful Starting References

- MIDV-2020 — official dataset page, University of La Rochelle
- MIDV-2020: A Comprehensive Benchmark Dataset for Identity Document Analysis
- OpenCV documentation for perspective transforms
- Hugging Face TrOCR documentation
- Donut: OCR-free Document Understanding Transformer
- Hugging Face Donut documentation
- scikit-learn documentation for simple classification models
- Dean & Barroso, *The Tail at Scale*

These are references to consult as their corresponding phases begin. There is no need to read all of them before starting.

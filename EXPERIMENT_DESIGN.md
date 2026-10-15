# Adaptive KYC Router — Experimental Contract

## 1. Status and purpose

This document defines version 1 of the experiment for the adaptive KYC router
research project. It translates the broader roadmap in `PROJECT.md` into rules
that can be implemented and evaluated consistently.

Phase 2 calibration is complete. Its delivered artifacts and validation are
recorded in [`docs/phases/phase-2.md`](docs/phases/phase-2.md); the frozen image size and
degradation values are in
[`configs/controlled_degradations_v1.json`](configs/controlled_degradations_v1.json).

The contract has three kinds of decisions:

- **Frozen v1 decisions** define what the first experiment means. Changing one
  requires documenting the change and deciding whether prior results must be
  rerun.
- **Calibration parameters** are implementation values, such as blur sigma or
  canonical image resolution. Phase 2 will measure and freeze them before the
  test set is evaluated.
- **Deferred features** are intentionally outside v1. They are not forgotten;
  they are excluded to keep the first study small and interpretable.

The contract was written from the roadmap and the evidence produced in
`notebooks/01_explore_midv.ipynb`.

Where this document narrows or changes a provisional choice in `PROJECT.md`,
this contract is authoritative for the v1 experiment. The roadmap remains the
broader project plan.

---

## 2. Research question

The main question is:

> Can a document-processing system route identity documents between a modular
> OCR pipeline and an OCR-free document model while preserving most of the
> stronger fixed pipeline's field-extraction accuracy and reducing use of the
> Donut path?

This wording deliberately does not assume that Donut is more accurate, more
robust, or slower. Those are empirical questions.

The first experiment tests three working hypotheses:

1. The OCR and Donut pipelines have at least some complementary failures.
2. Cheap image-quality measurements contain enough information to predict some
   of those failures.
3. A simple router can stay within one percentage point of the stronger fixed
   pipeline's field accuracy while sending fewer than 100% of samples to Donut.

If one pipeline dominates the other, or the router cannot meet the constraint,
that is a valid negative result rather than a failed project.

---

## 3. Scope

### 3.1 Included in v1

The first version includes:

- one planar identity-card type;
- extraction of seven basic identity fields;
- document rectification using ground-truth corners;
- one crop-based OCR pipeline;
- one OCR-free Donut pipeline;
- clean, synthetically degraded, and real-capture evaluation;
- field accuracy and character-error metrics;
- image-quality features;
- fixed-route, threshold-route, learned-route, and oracle comparisons;
- per-request inference and end-to-end latency.

### 3.2 Excluded or deferred

The following are not part of v1:

- document localization with predicted corners;
- perspective-error experiments;
- combined synthetic degradations;
- face matching or face recognition;
- signature recognition;
- liveness, forgery, hologram, or NFC checks;
- AML or sanctions screening;
- multiple document layouts in one model;
- confidence-aware fallback routing;
- load-aware routing and concurrent serving;
- neural routing models;
- distributed training or serving infrastructure.

These features would confound the first recognition and routing result or add
substantial implementation work without being necessary to answer the initial
question.

---

## 4. Dataset decision

### 4.1 Initial dataset and document type

The initial dataset is MIDV-2020. Version 1 uses only:

```text
alb_id
```

The dataset investigation found 100 `alb_id` identities with complete template
and photograph pairs. Every selected text field is annotated and non-empty for
all 100 identities.

`alb_id` was selected because:

- all seven desired basic fields are present in 100/100 identities;
- the layout has no apparent vertical fields among those targets;
- its template and field regions are comparatively large;
- its labels make the meanings of card number and personal number explicit;
- the layout is simple enough for the first crop-based OCR system;
- names and places still provide some Unicode/diacritic variation.

This is a practical starting choice, not a claim that `alb_id` represents all
identity cards.

### 4.2 Roles of templates and photographs

MIDV templates and photographs serve different purposes:

- **Template annotations** provide field rectangles, raw field names, and
  ground-truth values.
- **Photo annotations** provide the document quadrilateral and face region.
- **Templates** provide the clean canonical starting point for controlled
  synthetic-degradation experiments.
- **Rectified photographs** provide the real-capture evaluation workload.

Controlled synthetic and real-capture results will be reported separately.
This prevents a constructed degradation such as Gaussian blur from being
treated as equivalent to an uncontrolled low-light photograph.

### 4.3 Dataset replacement and VIA portability

The project may later replace or supplement MIDV with another dataset using the
VGG Image Annotator format. VIA standardizes the JSON structure, but not field
semantics, file layout, or identity grouping. Therefore the data layer has
three conceptual boundaries:

```text
generic VIA parser
        |
dataset-specific adapter
        |
canonical document sample
        |
dataset-independent pipelines and evaluation
```

The generic VIA parser is responsible only for structural operations:

- reading `_via_img_metadata`;
- reading `shape_attributes` and `region_attributes`;
- representing rectangles and polygons;
- retaining unknown attributes rather than silently discarding them.

The MIDV adapter is responsible for:

- locating images and annotation files;
- matching template and photo filenames;
- constructing the MIDV identity key;
- interpreting `field_name` and `value`;
- mapping MIDV raw names into the canonical schema;
- decoding MIDV-specific capture metadata.

A future dataset gets its own adapter. It should not require changes to metric
implementations or model output schemas. This is a design boundary, not a
requirement to build a large abstraction framework during exploration.

---

## 5. Evaluation unit and split

### 5.1 Identity key

The independent unit is the underlying mock document identity:

```text
{document_type}/{two-digit sample ID}
```

For example:

```text
alb_id/42
```

The following are different observations of one identity, not independent
examples:

```text
template/alb_id/42
photo/alb_id/42
synthetic_blur/alb_id/42
synthetic_jpeg/alb_id/42
```

### 5.2 Split policy

The 100 `alb_id` identities will be split as follows:

| Split | Identities | Purpose |
|---|---:|---|
| Training | 70 | Model and router fitting |
| Validation | 15 | Model selection, thresholds, and calibration |
| Test | 15 | One final unbiased evaluation |

The split uses seed `2026`. It is generated at the identity level. Every source
view and every synthetic derivative inherits its identity's split.

The final assignment will be saved as a version-controlled manifest rather
than regenerated implicitly. At minimum, the manifest will contain:

```text
identity_key, document_type, sample_id, split, split_seed
```

No decision may be tuned from test results. If a serious annotation defect is
found in the test set, the correction or exclusion must be documented rather
than quietly replacing the sample.

### 5.3 Why this policy exists

Splitting template and photo views separately would expose the same name,
portrait, numbers, and other identity-specific values during training and
testing. Synthetic copies make the leakage even more direct. Grouping all
observations by identity prevents that error.

---

## 6. Input contract and rectification

### 6.1 Logical input

The logical input to each extraction pipeline is one canonical RGB document
image with a fixed width and height.

For the real-photo experiment, that image is produced as follows:

```text
MIDV photograph
    -> ground-truth doc_quad
    -> perspective transform
    -> fixed-size canonical document image
```

For the controlled-degradation experiment, the clean template is resized to
the same canonical dimensions before degradation.

The canonical dimensions selected in Phase 2 are frozen in the versioned
configuration linked above. They preserve the `alb_id` template aspect ratio
and are used by both pipelines before final evaluation.

### 6.2 Ground-truth corners

Version 1 uses the annotated document quadrilateral. This intentionally removes
localization accuracy from the comparison. The experiment therefore measures
recognition after idealized localization, not a complete production KYC
pipeline.

Predicted corners and localization failure become a separate later experiment.

### 6.3 Equal-input rule

For any evaluated sample, both pipelines receive the same canonical pixel
array. The OCR path may then make field crops internally, while Donut consumes
the whole canonical image. Neither path may receive a cleaner or differently
resized version.

---

## 7. Canonical output schema

### 7.1 Required fields

Version 1 has exactly seven required textual fields:

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

The MIDV mapping is:

| Canonical field | MIDV raw field | Reason |
|---|---|---|
| `document_number` | `number` | The card labels it as the card number |
| `surname` | `surname` | Direct semantic match |
| `given_names` | `name` | The card labels it as given name |
| `date_of_birth` | `birth_date` | Direct semantic match |
| `expiry_date` | `expiry_date` | Direct semantic match |
| `nationality` | `nationality` | Direct semantic match |
| `sex` | `gender` | Same concept under the canonical key |

All seven occur with non-empty values in 100/100 `alb_id` template annotations.
Consequently, an empty prediction for any of them is an error in v1.

### 7.2 Optional fields

There are no optional output fields in v1. This avoids ambiguous metric
denominators and keeps Donut's target sequence small.

Possible future additions include:

- issue date;
- birthplace;
- issuing authority.

Adding one requires updating the contract and applying the same field to both
pipelines.

### 7.3 Non-textual regions

The following MIDV annotations are retained as dataset metadata but are not
extraction targets:

- `photo`;
- `face`;
- `signature`.

Their empty text values are expected and must not be reported as missing text
annotations.

### 7.4 Excluded textual fields

The MIDV `id_number` field is excluded from v1. On `alb_id`, it is explicitly
the personal number, while `number` is the card number. Extracting both is not
needed for the minimal experiment.

`issue_date`, `birth_place`, and `authority` are also excluded from v1 output.
They remain available in the raw annotations and can be reconsidered later.

---

## 8. Shared normalization contract

Exact Match is meaningful only if harmless formatting differences are handled
consistently. One shared, versioned normalization implementation will process
both predictions and ground truth. Models must not have separate scoring rules.

Raw predictions will also be retained for error analysis.

### 8.1 Common normalization

Every textual value undergoes:

1. conversion to a string, unless it is missing;
2. Unicode NFC normalization;
3. removal of leading and trailing whitespace;
4. collapse of consecutive internal whitespace to one space;
5. case-insensitive comparison using Unicode-aware case folding.

Diacritics are preserved. For example, a prediction without an expected accent
is not considered an exact match. Meaningful punctuation is also preserved
unless a field-specific rule below says otherwise.

No fuzzy string matching is used for Exact Match.

### 8.2 Names and nationality

`surname`, `given_names`, and `nationality` use only common normalization.
They are not transliterated, reordered, or translated. The bilingual
`Shqiptare/Albanian` ground truth remains a single value.

### 8.3 Document number

`document_number` is normalized by:

- applying common normalization;
- removing all remaining whitespace;
- converting letters to uppercase.

Punctuation is preserved because punctuation may be part of identifiers in a
future dataset. No confusion map such as `O -> 0` or `I -> 1` is applied; those
are recognition mistakes, not harmless formatting.

### 8.4 Dates

Ground truth and prediction dates are parsed only through an explicit list of
supported formats. Successful values are serialized as:

```text
YYYY-MM-DD
```

The `alb_id` source format is initially `DD-MM-YYYY`. The implementation may
accept separators such as `/`, `-`, or `.` when the day/month/year ordering is
unambiguous. It must not use unconstrained natural-language or locale guessing.

An impossible or unparseable date is invalid and cannot receive Exact Match
credit merely because some characters resemble the target.

### 8.5 Sex

`sex` uses common normalization followed by uppercase conversion. Version 1
expects the dataset values `M` or `F`. It does not automatically convert words
such as `male` into `M`, because doing so would hide a schema-generation error.

---

## 9. Compared extraction pipelines

### 9.1 OCR pipeline

The modular path is:

```text
canonical image
    -> known field rectangles
    -> seven field crops
    -> pretrained TrOCR-style recognizer
    -> field-specific normalization
    -> canonical JSON
```

The first implementation starts with a pretrained printed-text recognizer. It
will not train an OCR system from scratch. Fine-tuning is allowed only if the
pretrained baseline shows systematic failures and the reason is recorded.

The field rectangles come from template geometry mapped into canonical image
coordinates. This means v1 evaluates recognition, not text detection or layout
discovery.

### 9.2 OCR-free pipeline

The OCR-free path is:

```text
canonical image
    -> fine-tuned Donut model
    -> serialized structured sequence
    -> parser
    -> field-specific normalization
    -> canonical JSON
```

Donut consumes the full image and generates only the seven required fields. It
starts from pretrained weights and is fine-tuned; it is not trained from
scratch.

### 9.3 Fair-comparison rules

Both paths must use:

- the same identity splits;
- the same canonical image for each sample;
- the same seven required fields;
- the same normalization implementation;
- the same ground truth;
- batch size one for the primary latency comparison;
- the same hardware and recorded software environment.

Model-specific preprocessing is allowed when required by the model, but it must
be included in end-to-end latency.

The exact pretrained checkpoints are implementation parameters to record before
training begins. Test results must not determine which checkpoint is reported.

---

## 10. Controlled image degradation

### 10.1 Purpose

Controlled degradation creates known changes to a clean input so accuracy can
be plotted against severity. The known degradation name and severity are
benchmark metadata, not features available to the deployed router.

### 10.2 Placement

Synthetic degradations are applied to the canonical document image, after its
size and color representation are standardized. For the controlled benchmark,
the base image is the clean template.

This choice isolates recognition robustness. It does not claim to reproduce a
complete camera or localization process.

### 10.3 Degradation families

Version 1 uses four independent families:

1. **Gaussian blur** — reduces edge and character sharpness.
2. **Downsampling** — reduces effective text resolution by shrinking and then
   resizing back to the canonical dimensions.
3. **JPEG compression** — introduces blocking, ringing, and detail loss through
   an encode/decode round trip.
4. **Exposure change** — creates underexposed and overexposed variants while
   retaining the canonical image dimensions.

Perspective distortion is deferred. Because v1 assumes ground-truth
rectification, adding perspective would mix recognition robustness with
rectification error. That deserves a separate experiment based on controlled
corner perturbations.

### 10.4 Severity scale

Each degradation uses:

```text
0 = clean canonical image
1 = very mild
2 = mild
3 = medium
4 = strong
5 = severe
```

Only one degradation is applied to a sample at a time. Version 1 does not test
combinations such as blur plus JPEG because combinations would multiply the
benchmark size and make causal interpretation harder.

Exposure direction is recorded explicitly as `underexposure` or
`overexposure`; the two directions are never pooled into one unlabeled result.

### 10.5 Parameter calibration

The exact parameters—blur sigma, downsampling scale, JPEG quality, and exposure
factor—were selected in Phase 2 using training and validation identities.
The values must satisfy three conditions:

- severity is monotonic by construction;
- adjacent levels are visually distinguishable without being redundant;
- the range extends from nearly harmless to clearly damaging.

Parameters must not be chosen from test accuracy. The frozen parameter table
is in the versioned configuration linked above; the calibration and validation
record is in the Phase 2 delivery document.

### 10.6 Reproducibility

Each generated observation will retain at least:

```text
identity_key
base_source
degradation_family
exposure_direction, if applicable
severity
parameter_values
random_seed, if randomness is used
transform_version
```

Transforms should be deterministic whenever possible. Both extraction
pipelines are evaluated on the exact same generated pixels.

### 10.7 Real-capture evaluation

Rectified MIDV photographs form a separate real-capture benchmark. Results may
be grouped by the published conditions:

- projective distortion;
- text background;
- keyboard background;
- outdoors;
- table background;
- highlights/reflections;
- low light;
- cloth background.

Condition and identity are not independently sampled: each identity has one
still photograph. Therefore comparisons between condition groups are
descriptive and not controlled within-identity causal comparisons.

---

## 11. Defining and estimating difficulty

### 11.1 Difficulty is outcome-relative

An image is not intrinsically “easy” or “hard.” It is easy or hard relative to
a pipeline and an output requirement.

For each sample, offline evaluation creates one of four useful descriptions:

| Outcome | Interpretation |
|---|---|
| Both pipelines correct | Easy for both; prefer the cheaper route |
| OCR correct, Donut wrong | OCR-favorable |
| OCR wrong, Donut correct | Donut-favorable |
| Both wrong | Unsolved by the current system |

For partial document results, the number of exactly correct required fields is
used rather than pretending that all failures are equally severe.

### 11.2 Router-observable image features

The first router uses only cheap measurements computed from the canonical input:

- variance of the grayscale Laplacian as a sharpness feature;
- mean grayscale brightness;
- grayscale standard deviation as a contrast feature;
- fraction of very dark pixels;
- fraction of nearly saturated bright pixels.

Thresholds used for dark and bright pixels are calibration parameters and must
be frozen from training/validation data.

Version 1 deliberately excludes complicated learned quality embeddings and
specialized JPEG-blocking estimators. A feature can be added later only if a
failure analysis shows a concrete need.

### 11.3 Forbidden router inputs

The deployed router must not receive:

- synthetic degradation family;
- synthetic severity;
- ground-truth field values;
- pipeline correctness;
- test-set labels.

Those values are available only to the offline benchmark and oracle.

### 11.4 Confidence-aware routing

OCR confidence and semantic-validation results may be strong difficulty
signals, but using them requires running OCR before the routing decision. That
creates a cascade with different latency economics. It is deferred until the
image-quality router has been evaluated.

---

## 12. Accuracy metrics

### 12.1 Primary metric: macro Field Exact Match

For field `f` and document `i`:

```text
EM(i, f) = 1 if normalize(prediction(i, f)) == normalize(truth(i, f))
           0 otherwise
```

Accuracy is first computed separately for each of the seven fields. The primary
score is the unweighted mean of those seven field accuracies:

```text
Macro Field EM = mean(field accuracy for each required field)
```

Macro averaging ensures that a field with more observations in a future dataset
cannot dominate the result. In the initial `alb_id` test, every field has the
same number of observations, so macro and micro Field EM should coincide.

### 12.2 Per-field Exact Match

Every report includes the seven individual field accuracies. This prevents one
strong aggregate number from hiding a consistently weak field such as document
number or nationality.

### 12.3 Character Error Rate

CER is secondary and diagnostic:

```text
CER = (substitutions + deletions + insertions) / ground-truth characters
```

It is calculated on normalized strings and reported overall and per field.
Corpus-level CER sums edit operations and ground-truth characters before
division. CER distinguishes a one-character error from a wholly incorrect
value, but it does not replace Exact Match for identity data.

### 12.4 Document Exact Match

A document receives Document Exact Match credit only if all seven required
fields match:

```text
Document EM(i) = 1 if every required field has EM(i, f) == 1
```

This is a strict secondary metric. It is informative for complete-record
quality but can be unstable on a 15-identity test set, so it is not the primary
optimization target.

### 12.5 Missing, malformed, and extra output

- A missing required key is scored as an empty prediction and is incorrect.
- An explicit empty value for a required field is incorrect.
- Invalid JSON or an unparseable generated sequence produces missing required
  fields and is recorded as a structural-output failure.
- Unexpected extra keys are ignored for the seven-field score but recorded as
  schema violations.
- An invalid date is incorrect even if some characters match.

The selected `alb_id` ground truth has no absent required fields. If a future
dataset has structurally unavailable fields, it must supply an explicit
applicability mask; structural absence must never be encoded as an ordinary
empty ground-truth string.

---

## 13. Routing task and optimization objective

### 13.1 Fixed baselines

Every router is compared with:

- **Always OCR** — 0% Donut usage;
- **Always Donut** — 100% Donut usage.

These baselines establish whether routing is useful at all.

### 13.2 Simple routing baselines

Version 1 evaluates:

- a one-feature threshold router, initially based on sharpness;
- a small, depth-limited decision tree using the five quality features.

The decision tree is preferred over a neural router because it is interpretable
and appropriate for the small dataset. Logistic regression may be reported as
an additional simple baseline, but it is not required.

### 13.3 Offline routing label

Both pipelines are run offline on every benchmark sample. Let their per-sample
score be the number of required fields with Exact Match.

The preferred label is:

```text
DONUT if Donut has more correct required fields than OCR
OCR otherwise
```

Ties go to OCR because the initial objective is to minimize Donut usage. This
is a training/diagnostic label, not an input available at inference.

### 13.4 Oracle

The oracle applies the same outcome rule with perfect knowledge of both
pipelines' results. It reports the upper bound available to a two-route selector
under the tie-to-OCR rule.

The oracle is not deployable. Its purpose is to reveal whether enough pipeline
complementarity exists to justify routing research.

### 13.5 Primary routing objective

Let the stronger fixed pipeline be whichever of Always OCR and Always Donut has
the higher test Macro Field EM. The v1 target is:

> Achieve Macro Field EM no more than one absolute percentage point below the
> stronger fixed pipeline while minimizing the percentage of samples routed to
> Donut.

For example, if the stronger fixed pipeline achieves 92.0%, the target floor is
91.0%, not 91.08% and not a one-percent relative reduction.

The one-point constraint is simple, interpretable, and avoids an arbitrary
weighted combination of accuracy and cost. The complete tradeoff table is still
reported; failure to meet the target is not hidden.

### 13.6 Router evaluation metrics

Each strategy reports:

- Macro Field EM;
- per-field Exact Match;
- CER;
- Document Exact Match;
- percentage routed to Donut;
- average end-to-end latency;
- agreement with the oracle;
- rate at which the chosen route produces zero correct required fields.

“Router classification accuracy” is not the main result. A route mistake that
does not change extracted fields is less important than one that causes a
document failure.

---

## 14. Latency contract

### 14.1 Measurements

Two times are recorded for each request:

1. **Model inference time** — the model forward/generation operation.
2. **End-to-end pipeline time** — model-specific preprocessing, crop creation
   where applicable, inference, decoding, parsing, normalization, and validation.

Model loading, dataset loading, synthetic-degradation generation, and result
file writing are excluded from per-request latency.

End-to-end latency is the main comparison because it represents the actual cost
of returning structured fields. Inference time helps explain where that cost is
spent.

### 14.2 Measurement conditions

Latency experiments must record:

- CPU, GPU, and memory configuration;
- operating system and major library versions;
- model checkpoint and precision;
- canonical image resolution;
- batch size, fixed to one for the primary result;
- warm-up count and measured repetition count;
- whether execution uses CPU or GPU.

GPU timings must synchronize the device before stopping the timer. Warm-up
requests are excluded from reported values.

### 14.3 Initial and later reporting

During the extraction phases, report at least median and mean per-request
latency. During the later serving/load phase, add:

- p50, p95, and p99 end-to-end latency;
- queue latency;
- throughput;
- CPU/GPU utilization;
- memory usage;
- concurrency level.

Concurrent serving and load-aware routing do not alter the initial offline
accuracy contract.

---

## 15. Experimental procedure

The experiment is executed in this order:

1. Generate and save the identity split manifest.
2. Implement canonical rectification and select the fixed image resolution.
3. Freeze the four degradation functions and their severity parameters using
   only training and validation identities.
4. Build the pretrained OCR baseline on training identities.
5. Fine-tune Donut on training identities.
6. Select checkpoints and any decoding parameters using validation identities.
7. Freeze both extraction pipelines.
8. Run both frozen pipelines on clean, synthetic, and real-capture observations
   for each split.
9. Build router-training rows from training-identity outcomes.
10. Select router thresholds and tree hyperparameters using validation
    identities.
11. Freeze the routing policies.
12. Evaluate all fixed strategies once on test identities.
13. Produce aggregate tables, degradation curves, capture-condition summaries,
    and representative failure examples.

The extraction pipelines and router may never be retuned from final test
results. If the experiment is changed after viewing test performance, the
change must be logged and the prior test result retained as exploratory.

---

## 16. Benchmark record format

One row represents one evaluated observation under one pipeline. A practical
schema includes:

```text
identity_key
split
base_source
capture_device
capture_condition
degradation_family
degradation_direction
severity
degradation_parameters

sharpness
mean_brightness
contrast
dark_pixel_fraction
bright_pixel_fraction

pipeline
raw_prediction
normalized_prediction
per_field_exact_match
correct_field_count
character_edits
document_exact_match
schema_valid

inference_latency_ms
end_to_end_latency_ms
model_version
experiment_version
```

Capture fields are null for template-only controlled observations. Degradation
fields are `none`/zero for clean or real-capture-only observations.

Storing sample-level predictions allows metrics to be recalculated and audited
without rerunning expensive inference.

---

## 17. Required result tables and plots

At minimum, the final analysis reports:

1. Per-field Exact Match and CER for Always OCR and Always Donut.
2. Macro Field EM and Document EM for both fixed pipelines.
3. Accuracy versus degradation severity for each degradation family and model.
4. Underexposure and overexposure as separate curves.
5. Descriptive results by real capture condition.
6. The OCR/Donut outcome matrix:
   - both correct;
   - OCR only correct;
   - Donut only correct;
   - both incorrect.
7. The routing tradeoff table:

   | Strategy | Macro Field EM | Donut usage | End-to-end latency |
   |---|---:|---:|---:|
   | Always OCR | ... | 0% | ... |
   | Always Donut | ... | 100% | ... |
   | Sharpness threshold | ... | ... | ... |
   | Decision tree | ... | ... | ... |
   | Oracle | ... | ... | ... |

8. A small qualitative failure gallery showing:
   - OCR succeeds and Donut fails;
   - Donut succeeds and OCR fails;
   - both fail;
   - the router chooses the worse route.

The analysis must report confidence intervals or raw numerator/denominator
counts where practical because the final test contains only 15 identities.

---

## 18. Reproducibility and change control

Every reported run must preserve:

- Git commit or source version;
- split-manifest version;
- Python and dependency versions;
- model checkpoint identifiers;
- training configuration and random seeds;
- canonical resolution;
- degradation configuration;
- normalization version;
- hardware description;
- sample-level prediction records.

Changes to any of the following create a new experimental-contract version:

- required fields;
- dataset or document type;
- identity grouping or split manifest;
- normalization rules;
- primary metric;
- degradation families or their placement;
- routing objective;
- test-set exclusions.

Changing a calibration value before test evaluation does not require a new
research question, but the final value must be versioned. Changing it after
inspecting test performance makes the new result a separate experiment.

---

## 19. Interpretation boundaries

The final claims must be limited to the tested workload. Important limitations
include:

- one MIDV identity-card layout;
- only 100 mock identities;
- generated identity data, faces, and signatures;
- printed and laminated mock documents rather than live KYC traffic;
- two phone models in deliberately constructed capture conditions;
- capture condition confounded with identity in still photographs;
- ground-truth rather than predicted document corners;
- field crops derived from known template geometry for OCR;
- four isolated synthetic degradation families;
- no forgery, biometric, security-feature, or fraud assessment;
- consumer-hardware measurements that may not transfer to other systems.

The appropriate form of a conclusion is:

> On the tested `alb_id` subset and specified degradations, the routing policy
> achieved X Macro Field EM, used Donut for Y% of observations, and had Z
> end-to-end latency under the recorded hardware conditions.

The study must not conclude that the system is generally suitable for real KYC
deployment.

---

## 20. Remaining calibration work

The research-design decisions are closed. Phase 2 froze the canonical image
dimensions and the blur, downsampling, JPEG, and two exposure-direction
severity ladders in the versioned configuration linked above. Phase 3 froze the
OCR checkpoint, crop/decoding policy, and OCR latency protocol in
[`configs/ocr_pipeline_v1.json`](configs/ocr_pipeline_v1.json); its
[delivery record](docs/phases/phase-3.md) documents the decision and verification.
Phase 4 froze the selected Donut checkpoint, tokenizer/serialization,
preprocessing, and decoding in
[`configs/donut_pipeline_v1.json`](configs/donut_pipeline_v1.json), and matched
warm-up/repetition conditions for both paths in
[`configs/ocr_donut_comparison_v1.json`](configs/ocr_donut_comparison_v1.json).
The [Phase 4 record](docs/phases/phase-4.md) links training, verification, and
comparison evidence. These are implementation/calibration records, not changes
to v1 scoring, scope, splitting, or the sealed-test policy.

Later benchmark work must still freeze the dark- and bright-pixel thresholds
using only training/validation observations.

These are not open questions about the purpose of the experiment. They are
calibration parameters whose selection procedure is already defined above.

---

## 21. Phase 0 completion criteria

Phase 0 is complete when this contract is accepted and versioned. At that point
we can explain, without inventing new rules:

- what the system receives;
- what both pipelines return;
- which dataset and identities are used;
- how leakage is prevented;
- which degradations are tested;
- how correctness and latency are measured;
- what the router may observe;
- what qualifies as routing success;
- which claims the evidence can and cannot support.

Historical Phase 0 next step was image rectification and degradation calibration.
For current phase status and next work, consult [PROJECT.md](PROJECT.md).

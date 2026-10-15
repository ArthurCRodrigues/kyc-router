# Phase 4 — OCR-free pipeline delivery

**Complete: 2026-10-15.** The user authorized complete Phase 4
delivery, including executed learning notebooks, experimental evidence, and
source extraction. Engineering checks and learning reflection are separate:
running the workshops does not certify learner mastery.

## Adopted decision and comparison

Adopt validation-selected **epoch 10** from Donut training run
`20261015T122657718426Z` as **`donut_v1`**. The
[frozen manifest](../../configs/donut_pipeline_v1.json) records exact trained
artifact hashes, pretrained revision, vocabulary/schema, PIL preprocessing,
generation, shared rules, selection, and comparison protocol. Existing
`ocr_v1` is unchanged.

This is **frozen-encoder, all-decoder adaptation with Adafactor**, selected because
the recorded full-model/AdamW alternatives exceeded available GPU memory. It is
not described as full-model fine-tuning. The image encoder remains in inference;
training-only feature caches never enter the public API or timing study.

| Pipeline | Correct fields / 210 | Macro Field EM | Corpus CER | Complete documents / 30 |
|---|---:|---:|---:|---:|
| Adopted OCR | 194 | 92.38% | 28/1822 — 1.54% | 17 |
| Adopted Donut | 193 | 91.90% | 17/1822 — 0.93% | 16 |

These 30 views belong to 15 validation identities. Donut has zero structural
failures after selection, but 17 wrong fields remain: 14 names and three document
numbers. Fifteen errors still pass semantic validation. Relative to OCR, surnames
and birth date improve while given names regress. Donut is a useful comparator,
not a universal improvement or a certified identity-verification system.

Current matched warm mean request times are about **302 ms OCR / 512 ms Donut**.
An earlier retained run measured about **286 / 484 ms**, with the same accuracy.
The [comparison report](../experiments/ocr-donut-comparison.md) retains both named
runs, per-source medians/counts, memory, hardware, boundaries, session variation,
and development-only complementarity. No service/load or degraded-image claim
is inferred from those local measurements.

## Delivered workshops and learning material

| Artifact | Delivered learning/execution |
|---|---|
| [05a — foundations](../../notebooks/05a_donut_foundations.ipynb) | Pretrained processor/encoder/token path, actual GPU generation, orientation/detail plots, memory and stopping behavior |
| [05b — dataset](../../notebooks/05b_donut_dataset.ipynb) | Identity-safe paired inputs, all target/token round trips, escaping/parser failures, label shift/padding demonstration, real fresh-process resource records |
| [05c — fine-tuning](../../notebooks/05c_donut_finetuning.ipynb) | Actual tiny/main training evidence, optimizer/update explanations, full curves, epoch selection, raw failures, 3360-record scoring replay |
| [05d — evaluation](../../notebooks/05d_donut_evaluation.ipynb) | Paired accuracy/error gallery, actual matched timing study, research/source/reload checks, live GPU and CPU API requests |
| [OCR-free concepts](../learning/ocr-free.md) | Swin hierarchy, cross-attention, prompts/tokens, parsing, generation/caches, timing/memory interpretation |
| [Donut training](../learning/donut-training.md) | Target grammar, vocabulary/embeddings, teacher forcing, optimizer state, Adafactor, frozen-feature validity, accumulation, selection |

All four notebooks retain executed outputs. B/C/D together add 26 executed code
cells to A's 21: **47 code cells** total. They include setup, prediction prompts,
worked examples, reflection/exercises, and reproduction guidance. C/D explicitly
review genuine saved expensive runs by default rather than secretly retraining
or pretending archived timings were measured inside the notebook kernel.

## Data, configuration, and experiment records

- **70 training / 15 validation / 15 sealed test identities**, unchanged manifest,
  paired template/photo views and canonical pixels.
- **170/170 target round trips**, 37–43 tokens, no truncation. Seven original
  annotation values are preserved beside shared normalized targets.
- Resource evidence includes complete optimizer steps and OOMs. The selected
  Adafactor resource follow-up passed ten steps with about 1453 MiB allocated peak;
  the successful main run peaked at about 1677 MiB including validation work.
- Tiny training-only diagnostic reached 14/14 fields at 20 updates; those weights
  were discarded. Main training reset the parent/seed/optimizer and ran 16 epochs,
  stopping six epochs after the selected epoch 10.
- The [feasibility report](../experiments/donut-feasibility.md) preserves A's old
  evidence and adds B's actual resource procedure, failures, and selected policy.
- The [fine-tuning report](../experiments/donut-finetuning.md) retains every named
  attempt, full successful history, data/training/selection protocol, errors,
  regressions, raw counts, cache provenance, environment, and reproduction.
- The [comparison report](../experiments/ocr-donut-comparison.md) owns final paired
  accuracy, source verification, six-block warm timing, gallery, and limitations.

Current configuration: [dataset/serialization](../../configs/donut_dataset_v1.json),
[selected training v2](../../configs/donut_finetuning_v2.json),
[adopted pipeline](../../configs/donut_pipeline_v1.json), and
[matched comparison](../../configs/ocr_donut_comparison_v1.json). Earlier
foundations, AdamW v1, and resource follow-up configurations remain historical.
The experimental contract's calibration section links the completed freeze;
no v1 field/split/normalization/metric/routing objective was changed.

## Reusable source and verification

The package adds [serialization.py](../../src/kyc_router/serialization.py) and
[donut.py](../../src/kyc_router/donut.py). Stateless serialization/parser functions
and one concrete loaded pipeline are sufficient; no base framework or serving
abstraction is introduced. Shared schema, geometry checks, normalization,
validators, and scorer are reused. Inference accepts one whole canonical image,
with no reference, identity, field rectangle, training feature, correction map,
or OCR fallback. See [architecture](../architecture.md) and
[Donut usage](../guides/donut-usage.md).

Final model/source comparison run **`20261015T124404754152Z`** verified:

- **30/30 token sequences/raw strings**, **210/210 raw fields** matched independent
  research inference and the selected checkpoint's saved generation records.
- **180/180 measured repeated requests** across OCR and Donut retained exact
  tokens, normalized fields, and validators, with all denominators retained.
- **3360 historical field-score records** across all 16 epochs replayed exactly
  for normalized correctness and corpus CER in notebook C.
- **52 unit tests** passed without MIDV/model/GPU requirements, including the new
  parser, structural failures, whole-image boundary, corrupt/offline loading, and
  decode-policy guards; previous OCR tests remain passing.
- A live GPU request and one CPU validation-template request matched retained
  token/field output. This is not a full CPU benchmark.
- A regular wheel built without dependency replacement and imported independently
  outside the checkout, containing both source modules and allocating no CUDA
  context/changing no thread settings at import.
- Six new measured figures were visually reviewed; notebooks preserve descriptive
  alternative text. Reports include tracked copies of key curves/galleries.

The final audit command is:

```bash
.venv/bin/python -m experiments.verify_phase4
```

It checks artifacts, current-source/run hashes, notebook execution, scoring and
selection evidence, documentation links, package verification, and unit tests,
saving a named record under `outputs/phase4_closure/`.

The [execution/artifact audit](../../outputs/phase4_closure/20261015T125752826144Z/audit_manifest.json)
passed with all 47 executed code cells, 52 tests, intact source/artifact hashes,
and no broken local documentation links. B/C/D also rendered successfully to
HTML with figure descriptions. Its directory's `final_delivery_audit.json`
records final status/navigation hashes after the completion notes were updated;
the original measured/audit snapshots remain unchanged.

The provenance audit also invalidated an early unbound-cache generation smoke.
The [training report](../experiments/donut-finetuning.md) records the original
index error and corrected two-view check with exact image/cache binding,
bit-identical encoder features, and matching generated tokens. Main training's
complete metadata checks and the final 30-view selected/source verification are
unaffected; the invalid smoke is not cited as successful parity evidence.

## Artifact availability and limits

The selected weights, dataset, feature cache, and complete raw evidence remain
local/Git-ignored. A fresh clone/wheel contains source, configurations, reports,
and executed notebook outputs, but needs exact artifact restoration for adopted
model inference. Re-running training makes a new research run; it is not a reason
to overwrite old hashes or claim byte-identical restoration.

This remains one mock layout, 70 training identities, 15 repeatedly used paired
validation identities, one successful main seed, constant nationality, ground-
truth corners, known OCR field geometry, and consumer-hardware timings. Frozen
visual features, given-name regressions, remaining document-number mistakes,
valid-looking wrong values, resource/debug failures, and session timing variation
remain explicit limitations. There is no unseen-layout, real KYC safety, or
degradation-robustness claim.

## Handoff to Phase 5

Both paths are ready for the later frozen comparison workload. **Test remains
sealed.** Phase 4 has not generated the full synthetic benchmark, trained a
router, estimated online confidence, built a predicted localizer, or introduced
HTTP endpoints, queues, concurrent workers, distributed infrastructure, or
deployment. Phase 5 starts with the existing frozen degradation definitions and
the two adopted extraction paths; it must not retune them from test outcomes.

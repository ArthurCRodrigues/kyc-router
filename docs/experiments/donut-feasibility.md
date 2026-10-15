# Donut foundations and feasibility investigation

**Scope: notebook A's historical input/output checks (2026-10-13), followed by
notebook B's actual training-feasibility investigation (2026-10-15).** The A
measurements below remain historical; the appended B section records complete
optimizer steps, failures, and the selected resource compromise. Training and
comparison outcomes live in their own reports.

Execution: [05a — Donut foundations](../../notebooks/05a_donut_foundations.ipynb).
Concepts: [OCR-free extraction](../learning/ocr-free.md).
Policy: [`donut_foundations_v1.json`](../../configs/donut_foundations_v1.json),
an investigation policy rather than a frozen training/adopted inference policy.

## Questions and procedure

Can the pretrained candidate load in the installed environment, process an
`alb_id` canonical card, return inspectable encoder features, and generate a
bounded sequence on the local GPU? What does its native processor do to our
landscape input, and how much memory does a smaller landscape inference require?

The procedure was declared before generation:

1. Select the lexicographically first **training** identity from the existing
   manifest, independently of model output.
2. Open its template and photograph only; reconstruct the Phase 2 canonical RGB
   inputs using resizing and ground-truth photo corners respectively.
3. Verify the exact downloaded artifact hashes, inspect the saved configuration,
   and load the concrete installed processor/model APIs offline.
4. Compare native preprocessing with a bounded landscape configuration on CPU,
   display both views and a diagnostic surname detail, and inspect tokenization.
5. Load float32 weights, inspect parameter counts, run the standalone encoder
   and one next-token decoder step, then trace a short generated sequence.
6. Run two complete smoke calls, one per view; retain raw tokens, text, stopping
   flags, timings, and memory. Compare template token output with the standalone
   encoder/decoder trace.

No optimizer was created, no weights were adapted, and no model/resolution search
was performed. The landscape setting is a bounded candidate for later training
investigation, selected for canvas size/orientation rather than validation scores.

## Data boundary

- Dataset/layout: MIDV-2020 `alb_id`, identity **`alb_id/02`**.
- Manifest: [`splits/alb_id_v1.csv`](../../splits/alb_id_v1.csv), 70 train / 15
  validation / 15 test identities; split seed 2026.
- Loaded observations: **2 views of 1 training identity**, not 2 independent
  identities. No validation or test image/reference was loaded or scored.
- Canonical inputs: 1084 × 680 RGB uint8. Only the selected record in each shared
  VIA annotation file is used/displayed. The shared files themselves include
  records for other identities; hashing/reading their containers is not model
  evaluation on those identities.
- Ground-truth values and diagnostic field geometry are displayed for learning;
  neither references, field rectangles, nor identity metadata enter Donut.
- The model receives whole canonical pixels and the fixed pretrained task prefix.

The run summary preserves manifest, original-image, annotation-container,
canonical-pixel, configuration, and model artifact hashes.

## Model, environment, and exact settings

Successful run: **`20261013T130934816247Z`**.

| Item | Recorded setting or result |
|---|---|
| Pretrained model | `naver-clova-ix/donut-base` |
| Revision | `a959cf33c20e09215873e338299c900f57047c61` |
| Local artifact directory | `checkpoints/phase4_foundations/donut-base` |
| Artifact verification | 8/8 declared files matched the acquisition manifest |
| Architecture | Donut Swin encoder; four-layer mBART-family decoder |
| Parameters | Encoder 74,180,728; decoder 127,671,296; total **201,852,024** |
| Unique parameter storage | **770.00 MiB**, float32; not a total memory prediction |
| Processor | Explicit `DonutImageProcessorPil` with checkpoint tokenizer |
| Native processor output | `[1, 3, 2560, 1920]`; long-axis alignment rotates the landscape card |
| Exploratory processor output | `[1, 3, 640, 1024]`; long-axis rotation disabled |
| Attention | Eager implementation |
| Prompt | Existing `<s_synthdog>` token, ID 57524; not a seven-field task prompt |
| Generation | Greedy, one beam, 48 new tokens, KV cache enabled, forced EOS disabled explicitly |
| Seed | 2026 |
| Device | NVIDIA GeForce GTX 1650, compute capability 7.5; PyTorch-visible total 3726.38 MiB |
| CPU/RAM | 12 logical CPUs, 31.08 GiB RAM; current machine audit identifies Intel i5-12400 |
| Software | Python 3.14.7; torch 2.14.1+cu130; torchvision 0.29.1+cu130; Transformers 5.19.0 |
| Other relevant packages | huggingface_hub 2.2.0; Pillow 12.3.0; NumPy 2.5.3; pandas 3.0.5; OpenCV 5.0.0.93; nbclient 0.11.0 |
| Experiment threads | Four PyTorch threads; one OpenCV thread |

`platform.processor()` returned an empty CPU-model string in the original run;
the CPU model above is a separate `/proc/cpuinfo` audit of this checkout's
machine during documentation/verification, not a retroactive change to its
environment JSON. Exact package/platform records remain in the saved run.

The optional `IProgress` widget was unavailable, so tqdm emitted a progress-display
warning. Plain progress and all model calls still completed; no dependency upgrade
was needed for that cosmetic issue. Notebook A includes guidance for this message.

Model-specific resizing is internal to Donut; the shared canonical input is
unchanged. Native and landscape preprocessing were compared, but **neural
inference ran only with the landscape tensor**. No claim that native resolution
fits on the GPU follows from its successful CPU preprocessing.

## Preprocessing, architecture, and vocabulary findings

The native canvas has **7.5 times** the exploratory canvas area. Its processor
rotates the card clockwise and pads it into the portrait canvas. The landscape
view retains card orientation; a diagnostic zoom showed the surname remains
readable to a human on this training template. This is not an all-field,
all-identity, or model-accuracy readability check.

The encoder's derived stage grids were 160 × 256, 80 × 128, 40 × 64, and 20 × 32,
with 128, 256, 512, and 1024 channels. The actual final encoder tensor matched
**`[1, 640, 1024]`**: 640 spatial feature positions, not characters or field values.
The first decoder call returned `[1, 1, 57525]` logits. The notebook displays its
top eight next-token candidates; these are not calibrated extraction confidence.

The original vocabulary has **57,525 entries**. `<s_document_number>` splits
into **8 ordinary pieces** before registration. A separate tokenizer probe adds
15 schema/task tokens, yielding **57,540 entries**, and converts that field tag
to one token. The actual model and inference tokenizer remain unchanged. This
demonstrates the future embedding/serialization work without sending new IDs to
an unresized model.

## Raw generation and negative findings

The successful template sequence was:

```text
<s_synthdog> mblemrijsurname Ahmed emrigiven name Agim shteeeesia/nationality Shqiptare/Albanian vendindia/place of birth Bulcar, ALB date in dialdate of
```

The photo sequence was:

```text
<s_synthdog> Abmi ns. ns. ns. ns. ns. ns. ns. ns. ns. ns. ns. ns. ns. ns. ns. ns.
```

Trailing whitespace, exact token IDs, and original strings are preserved in
`summary.json`; these displayed text blocks are for reading. The template output
contains recognizable labels/values but misreads the surname and is not a
seven-field serialized answer. The photo output repeats a fragment.

**Both sequences used all 48 allowed new tokens; 0/2 emitted EOS and 2/2 reached
the length limit without EOS.** The convenience parser probe on the template
returned a `text_sequence` container, not the seven required keys. It is not
adopted as the project parser. No Field EM/CER/Document EM is claimed for this
pretrained machinery check, and no sample was dropped to conceal the failure.

The standalone template encoder/decoder trace matched the independent complete
template generation exactly. That checks composition, not held-out accuracy.

## Memory and timing boundary

After loading, PyTorch allocated **771.55 MiB** and reserved **786.00 MiB** on
CUDA. Each measured smoke call peaked at **1144.79 MiB allocated** and **1310.00
MiB reserved**. End-of-call process RSS was **2087.31 MiB**; CPU RSS includes
imports, CPU preprocessing buffers, and notebook figures as well as model work.
These GPU peaks are totals for this process since the per-call reset, not
incremental activation sizes. Other processes affect device free memory.

| View | CPU preprocessing | Transfer | Full model generation | String decode | Smoke request |
|---|---:|---:|---:|---:|---:|
| Template | 11.81 ms | 0.81 ms | 497.27 ms | 0.05 ms | **509.94 ms** |
| Photo | 11.08 ms | 0.80 ms | 498.93 ms | 0.04 ms | **510.85 ms** |

The boundary starts with canonical RGB pixels in memory and includes processing,
transfer, encoder plus decoder generation, token-list transfer to CPU, and string
decoding. CUDA synchronizes at stage boundaries. Ground-truth loading,
rectification, model loading, plotting, file writing, and the convenience parser
probe are excluded. This is not a complete future structured-extraction latency:
the strict parser, output normalization, and semantic validation are not yet
implemented in the Donut inference path.

There is **one measured call per source**, after earlier encoder/decoder
inspection. No controlled warm-up or repeated timing protocol was used, so these
are smoke-call observations rather than latency estimates. The one-time local
weight load/device transfer was 254.09 ms; the isolated encoder inspection was
479.02 ms. Neither is an OCR comparison, throughput result, or service latency.

## Initial failure and corrected execution

Historical attempt **`20261013T113722709059Z`** reached the evidence-export cell
after executing model calls, then failed because installed Transformers loading
diagnostics contain Python sets, which JSON cannot serialize. The fix converts
those diagnostics to sorted lists before saving. The failed input/executed
notebooks and traceback remain in their original run directory.

Review also caught a substantive generation issue in that attempt: passing
`forced_eos_token_id=None` only inside a supplied `GenerationConfig` let
Transformers 5.19 fill it from this checkpoint's forced-EOS default. Its apparent
EOS at the 48-token limit therefore cannot be interpreted as voluntary stopping.
The successful rerun passes explicit generation keyword arguments, which override
defaults after filling, and passes `max_length=None` to avoid competing length
settings. The notebook explains this version-specific behavior and retains
the checkpoint defaults beside the explicit arguments. The old attempt is not
used for stopping or timing conclusions above.

## Reproduction and retained artifacts

From the repository root, in the existing environment:

```bash
.venv/bin/python experiments/prepare_donut_foundations.py
.venv/bin/python experiments/run_donut_foundations.py --device cuda:0
```

The [preparation script](../../experiments/prepare_donut_foundations.py) downloads
only the declared pinned public artifacts and hashes them. The
[execution script](../../experiments/run_donut_foundations.py) uses the invoking
Python interpreter for a fresh notebook kernel, runs offline, retains failures,
and saves successful outputs back into 05a. `--device cpu` is available but was
not exercised for this run and would be a separate resource observation.

Successful local evidence:
[`outputs/phase4_foundations/20261013T130934816247Z/`](../../outputs/phase4_foundations/20261013T130934816247Z/).
It contains:

- `summary.json`: settings, environment, provenance, memory, tokens, raw sequences,
  load diagnostics, and source counts;
- `environment.json`, `checkpoint_manifest.json`, `policy.json`;
- `preprocessing.csv`, `first_token_distribution.csv`, `tokenization.json`;
- canonical images, processor comparison, and surname-detail PNGs;
- input/executed notebooks, `execution.json`, and artifact hashes.

Weights and full run evidence remain Git-ignored; 05a retains executed outputs
in the repository. Restoring MIDV and the pinned checkpoint is necessary for a
fresh clone to reproduce the model calls. The code hash in `execution.json`
identifies the executed cells; later documentation/Markdown edits do not rewrite
the historical execution.

Verification checked all 21 sequential execution counts, absence of cell errors,
the current notebook's code hash against the successful run, run-artifact and
policy hashes, exact template trace parity, and all 125 relative links in the
changed Markdown/notebook documents. The three figures were visually inspected
and given descriptive alternative text. HTML rendering succeeded, with its
learning structure inspected in the local browser. The `research` optional
dependencies now declare `nbclient` directly for the execution runner; the
already-installed measured version satisfies the declared range. Current
Markdown/figure-metadata edits preserve the executed code and historical run files.

## Notebook A's historical decision and next boundary

Notebook A's loading, processor, encoder, tokenizer, and real GPU generation
investigation is complete. Continue with the bounded landscape policy as a
**candidate to test**, keeping both its text-detail tradeoff and the native
orientation difference visible. No adopted Donut pipeline, training policy, or
new experimental contract is frozen here.

Notebook B must define complete seven-field targets and strict parsing, audit
paired train/validation examples, and measure a full forward/backward/optimizer
step with intended trainable parameters and optimizer state. Inference fitting
in memory does not establish training feasibility. Test remains sealed; Phase 4
is still in progress.

## Notebook B — Complete optimizer-step investigation (2026-10-15)

[05b](../../notebooks/05b_donut_dataset.ipynb) reconstructs the 140 training and
30 validation views, audits identity separation, tokenizes all seven-field
targets, tests parsing and teacher-forcing alignment, and displays the actual
resource runs below. Test is refused by the adapter before opening an image.
The resource measurements themselves ran in fresh Python processes to isolate
CUDA allocations; they are not estimates generated from parameter counts.

Policies were recorded in
[`donut_dataset_v1.json`](../../configs/donut_dataset_v1.json), then the explicit
[frozen-feature follow-up](../../configs/donut_feasibility_followup_v1.json) and
[Adafactor follow-up](../../configs/donut_adafactor_feasibility_v1.json). Parent
revision, PIL landscape processor, 1084 × 680 canonical input, float32 precision,
all seven fields, and the identity split remain unchanged. Resource fitting uses
training identities only; the probe selects the longest training target, 43
tokens, and a microbatch of one. No validation accuracy drove these adjustments.

The same GTX 1650/i5-12400 environment and recorded library versions were used,
with four PyTorch threads and one OpenCV thread. Each run saves its exact policy,
environment, mode, completed-step count, failure stage/traceback when applicable,
losses, gradient norms, timings, allocated/reserved peaks, and CPU RSS.

| Run | Mode | Actual outcome |
|---|---|---|
| `20261015T121455832353Z` | Initial adapter check | Failed before a model update: photo annotations were incorrectly assumed to contain every text transcription. Fixed to use the identity-matched template references. |
| `20261015T121507255226Z` | Full model, activation checkpointing, AdamW | OOM during first backward; zero completed optimizer steps. |
| `20261015T121606315164Z` | Frozen encoder, AdamW | OOM while initializing/updating optimizer state; zero completed optimizer steps. |
| `20261015T121708609758Z` | Frozen features, encoder off GPU, AdamW | Two complete steps passed; peak allocated **2426.13 MiB**, peak reserved **2692 MiB**. |
| `20261015T122314734067Z` | Frozen features, encoder off GPU, Adafactor | **Ten complete steps passed**; peak allocated **1453.04 MiB**, peak reserved **1502 MiB**. |

The two-step AdamW pass did not establish enough later headroom: diagnostic run
`20261015T121944731579Z` then OOMed during tiny-set backward before its first
generation check. The reported free memory/other-process use differed; this was
not a controlled study isolating a single allocator/desktop cause. The failed
diagnostic remains historical, not a successful main training run.

The chosen follow-up preserves all **127,686,656 decoder parameters** as trainable
and all **74,180,728 encoder parameters** as frozen. It changes optimizer state
storage rather than freezing the new schema embeddings or reducing output scope.
Adafactor has no first moment and uses factored second moments, explicit learning
rate 5e-5 for the resource probe, disabled relative/parameter scaling, and its own
RMS update clipping without external global clipping. Ten probe losses went from
15.39 to 5.30 with finite gradients; steps 2–10 each took about 0.13–0.14 seconds
for the measured forward/backward/update. This is a training resource observation,
not inference latency or convergence evidence.

Deterministic whole-image features can be cached because the encoder/preprocessing
are fixed. Their file hash, exact image hashes/order/splits, processor settings,
and unchanged encoder-state hash bind the cache to this experiment. It contains
no reference text. A newly received document is still encoded in the deployed
path; this cache is prohibited in comparison timing.

For fresh-process reproduction, with no other research model resident:

```bash
.venv/bin/python -m experiments.donut_feasibility --mode full_checkpointed
.venv/bin/python -m experiments.donut_feasibility --mode frozen_encoder
.venv/bin/python -m experiments.donut_feasibility --mode frozen_features
.venv/bin/python -m experiments.donut_feasibility --mode frozen_features_adafactor
```

Raw evidence lives under [`outputs/phase4_feasibility/`](../../outputs/phase4_feasibility/).
The actual dataset/token/parser audit is
[`outputs/phase4_dataset/20261015T122852277251Z/`](../../outputs/phase4_dataset/20261015T122852277251Z/):
**170/170 round trips**, **37–43 target tokens**, 70/15 disjoint train/validation
identities, all 170 paired views, adversarial parser cases, and the resource records.

The [training concepts](../learning/donut-training.md) explain the memory terms,
target grammar, loss alignment, optimizer, and cache limitations. The
[fine-tuning report](donut-finetuning.md) records the successful main run, the
additional device/cache generation fixes, and the actual validation selection.
The selected training policy is `donut_finetuning_v2.json`; v1 remains the failed
AdamW diagnostic policy. No broad training-feasibility claim is inferred from A's
inference-only numbers or from one short successful resource probe.

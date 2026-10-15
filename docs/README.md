# Documentation map

Start with the [project roadmap](../PROJECT.md) for scope and phase status, then
the [experimental contract](../EXPERIMENT_DESIGN.md) for the authoritative
evaluation rules. This directory separates delivered work, measured evidence,
learning material, and workshop plans so later phases can grow without mixing
those purposes.

## Delivered phases

- [Phase 2 — Image processing](phases/phase-2.md): canonical images, known field
  crops, frozen degradations, checks, and capture limitations.
- [Phase 3 — OCR pipeline](phases/phase-3.md): adopted pipeline, reusable source,
  verification, remaining errors, and the handoff to the next phase.
- [Phase 4 — OCR-free pipeline](phases/phase-4.md): selected Donut path, executed
  learning workshops, training/resource evidence, matched comparison, and verified source.

Phase 0 decisions live in the experimental contract. Phase 1's investigation is
the executed [dataset notebook](../notebooks/01_explore_midv.ipynb); no duplicate
phase report is needed.

Phase 4's [delivery record](phases/phase-4.md) records the completed audit.
All [A](../notebooks/05a_donut_foundations.ipynb),
[B](../notebooks/05b_donut_dataset.ipynb),
[C](../notebooks/05c_donut_finetuning.ipynb), and
[D](../notebooks/05d_donut_evaluation.ipynb) workshops have executed, with trained
weights, actual comparison evidence, and reusable whole-image inference. The
[workshop plan](workshops/phase-4.md) maps the delivered learning sequence.

## Use and maintain the code

- [Application architecture](architecture.md): the small package, its input
  boundary, file responsibilities, and intentionally deferred features.
- [OCR usage](guides/ocr-usage.md): environment setup, loading local weights,
  extracting fields, scoring, reproduction, and tests.
- [Donut usage](guides/donut-usage.md): exact trained artifact restoration,
  whole-image API, structural failures, scoring, CPU/GPU, and reproduction.

## Experiment evidence

These reports describe named runs, not whichever model happens to be loaded now.

| Report | Question answered | Executable record |
|---|---|---|
| [Pretrained OCR baseline](experiments/ocr-baseline.md) | How accurate and fast was the original recognizer, and what failed? | [04c](../notebooks/04c_ocr_evaluation.ipynb) |
| [OCR fine-tuning](experiments/ocr-finetuning.md) | What changed after adaptation, including regressions and cost? | [04d](../notebooks/04d_ocr_finetuning.ipynb) |
| [OCR resource lab](experiments/ocr-resource-profile.md) | What do transfers, cache hits, CPU quotas, batching, and queues cost? | [04e](../notebooks/04e_ocr_resource_lab.ipynb) |
| [Donut foundations / feasibility](experiments/donut-feasibility.md) | What does the pretrained path do, which complete training steps fit, and which resource attempts fail? | [05a](../notebooks/05a_donut_foundations.ipynb), [05b](../notebooks/05b_donut_dataset.ipynb) |
| [Donut fine-tuning](experiments/donut-finetuning.md) | How did bounded adaptation learn the task, which checkpoint was selected, and what errors remain? | [05c](../notebooks/05c_donut_finetuning.ipynb) |
| [OCR/Donut comparison](experiments/ocr-donut-comparison.md) | How do fixed-path accuracy, failures, matched latency, memory, and source/reload parity compare? | [05d](../notebooks/05d_donut_evaluation.ipynb) |

The resource lab is a bounded curiosity experiment. Its faster variants are not
automatically adopted, and it does not replace the later model-comparison or
serving/load benchmark.

## Learn the methods

Read these alongside the notebooks rather than treating them as prerequisites
to finish before trying anything.

1. [OCR overview](learning/OCR.md): recognition strategies, processors, and TrOCR.
2. [Generation and timing](learning/ocr-generation-and-timing.md): patches,
   attention, autoregressive decoding, caching, and completed GPU timing.
3. [Evaluation](learning/ocr-evaluation.md): Exact Match, edit distance, CER,
   complete-record accuracy, and honest interpretation.
4. [Fine-tuning](learning/ocr-finetuning.md): labels, teacher forcing, loss,
   gradients, optimization, memory, and checkpoint selection.
5. [Resource profiling](learning/ocr-resource-profiling.md): transfer costs,
   memory accounting, cache validity, CPU budgets, and overload.
6. [OCR-free extraction](learning/ocr-free.md): whole-image generation,
   structured targets, training versus inference, and memory feasibility;
   executed foundations, whole-image hierarchy, decoding/structure, and comparison.
7. [Donut training](learning/donut-training.md): target grammar/vocabulary,
   teacher-forcing alignment, memory/Adafactor, frozen features, accumulation,
   schedules, tiny overfit, and validation selection.

The [Phase 3 workshop plan](workshops/phase-3.md) records the learning sequence,
guardrails, delegation history, and review prompts. Foundations and pipeline
construction remain in [04a](../notebooks/04a_ocr_foundations.ipynb) and
[04b](../notebooks/04b_ocr_pipeline.ipynb).

The [Phase 4 workshop plan](workshops/phase-4.md) tracks all executed Donut
workshops. The notebooks include setup, predict-first prompts, guided exercises,
actual outputs, failure analysis, and explicit replay/retraining instructions.

## Where new documentation belongs

| Location | Purpose |
|---|---|
| `phases/phase-N.md` | One delivery record per completed phase: decisions, outputs, checks, limits, next boundary |
| `experiments/` | Method and measured results for a specific experiment, with run IDs and reproduction steps |
| `learning/` | Explanations of concepts and operations that readers can reuse |
| `workshops/` | Learning plans and step-by-step workshop organization |
| `guides/` | Practical setup, API usage, and maintenance instructions |
| `figures/` | Small, repository-tracked figures used by reports |

Keep future serving, routing, and OCR-free implementation details in their own
phase work; a directory name is not permission to implement them early. Use
plain names and relative links. Add a document only when it has a distinct
purpose, then link it here and from its phase record.

## Evidence and reproduction conventions

- The experimental contract wins over provisional roadmap or workshop examples.
  Versioned [configurations](../configs/) record actual implementation policies.
- Notebooks retain their executed research snapshots. Reusable source is for
  current execution; historical notebook copies and source hashes explain how
  earlier results were obtained.
- Reports identify the run, input/split boundary, model/settings, timing boundary,
  raw counts, limitations, and reproduction command. Development validation is
  not presented as untouched final test evidence. The test split stays sealed
  until the later authorized comparison.
- Large weights and raw runs live locally in ignored `checkpoints/` and
  `outputs/` directories. Links to them work only when those artifacts are
  present; a fresh clone does not include trained weights or the MIDV dataset.
- Historical audit manifests hash the files as they existed at that run. Moving
  reports or adding status notes changes current document hashes, not the
  recorded measurements. Do not rewrite old audits to pretend they describe the
  new documentation layout.
- Keep old experiment findings, including negative results and regressions.
  Link subsequent decisions instead of silently replacing historical evidence.

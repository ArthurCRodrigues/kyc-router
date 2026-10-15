# KYC Router

Document recognition and adaptive routing, built as a hands-on research project
on consumer hardware.

> Can choosing the right model for each document preserve accuracy while
> reducing processing cost?

## The idea

Turn identity-card images into structured names, numbers, and dates. Compare two
extraction paths, then find out when choosing between them is worthwhile.

- **OCR — built:** recognize text in known field crops.
- **OCR-free — built:** read the whole card with the
  [trained Donut path](docs/guides/donut-usage.md), retaining structured failures.
- **Routing — planned:** choose a path using image quality, then study accuracy
  and latency under load.

## Built to learn, not just to run

| Resource | What you'll find |
|---|---|
| [Notebook workshops](notebooks/) | Step-by-step code, experiments, decisions, and observations. |
| [OCR foundations](docs/learning/OCR.md) | Recognition strategies and how TrOCR reads text. |
| [Transformers and generation](docs/learning/ocr-generation-and-timing.md) | Image patches, attention, decoding, and GPU timing. |
| [Evaluation](docs/learning/ocr-evaluation.md) | Exact Match, character errors, and interpreting results. |
| [Fine-tuning](docs/learning/ocr-finetuning.md) | Training labels, loss, gradients, and checkpoint selection. |
| [Resource profiling](docs/learning/ocr-resource-profiling.md) | CPU/GPU transfers, caches, memory, and queues. |
| [OCR-free extraction](docs/learning/ocr-free.md) | Whole-image generation, structured targets, and Phase 4 training concepts. |
| [Donut training](docs/learning/donut-training.md) | Target grammar, teacher forcing, memory/Adafactor, frozen features, and selection. |

## Current milestone

The OCR path is complete, with fine-tuned weights, resource experiments, and a
[tested Python package](src/kyc_router/).

**Validation field accuracy: 66.19% → 92.38% after fine-tuning.**
That's 194/210 fields across 15 identities and one MIDV-2020 layout.
[Results, failures, and tradeoffs →](docs/phases/phase-3.md)

**Phase 4 is complete:** all A–D workshops executed, Donut fine-tuned and frozen,
and a reusable pipeline verified against the research output.

| Fixed path | Validation correct fields | Macro Field EM |
|---|---:|---:|
| OCR | 194/210 | 92.38% |
| Donut | 193/210 | 91.90% |

Donut has lower character error and higher measured latency. The
[comparison](docs/experiments/ocr-donut-comparison.md) retains failures, timings,
resource compromises, and limits. [Phase 4 delivery →](docs/phases/phase-4.md)
Test remains sealed; the degradation benchmark is next.

## Explore

[Roadmap](PROJECT.md) · [Architecture](docs/architecture.md) ·
[Run the OCR pipeline](docs/guides/ocr-usage.md) · [All documentation](docs/README.md)

A research prototype, not a production identity-verification service.

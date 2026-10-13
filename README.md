# KYC Router

Document recognition and adaptive routing, built as a hands-on research project
on consumer hardware.

> Can choosing the right model for each document preserve accuracy while
> reducing processing cost?

## The idea

Turn identity-card images into structured names, numbers, and dates. Compare two
extraction paths, then find out when choosing between them is worthwhile.

- **OCR — built:** recognize text in known field crops.
- **OCR-free — next:** generate structured fields from the whole document.
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

## Current milestone

The OCR path is complete, with fine-tuned weights, resource experiments, and a
[tested Python package](src/kyc_router/).

**Validation field accuracy: 66.19% → 92.38% after fine-tuning.**
That's 194/210 fields across 15 identities and one MIDV-2020 layout.
[Results, failures, and tradeoffs →](docs/phases/phase-3.md)

## Explore

[Roadmap](PROJECT.md) · [Architecture](docs/architecture.md) ·
[Run the OCR pipeline](docs/guides/ocr-usage.md) · [All documentation](docs/README.md)

A research prototype, not a production identity-verification service.

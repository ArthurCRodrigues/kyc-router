# KYC Router

An exploration of document recognition, adaptive model routing, and the
engineering behind useful machine-learning pipelines—built and documented as
a learning journey.

Identity-document images vary in sharpness, lighting, and readability. A single
extraction method may not be the best choice for every image. This project asks:

> Can we choose the right extraction path for each document to preserve
> accuracy while reducing processing cost?

The aim is to investigate that question with understandable code, measured
experiments, and consumer hardware. This is a research and learning project,
not a production identity-verification service.

## What we're building

The project starts with images of identity cards and extracts structured fields
such as names, document numbers, and dates. We are working toward comparing two
approaches:

- **An OCR pipeline:** crop known fields, recognize their text, then normalize
  and validate the results.
- **An OCR-free document model:** read the whole document image and generate
  structured fields directly.

Once both paths are understood and measured, we will explore simple routing
policies that choose between them. The interesting result is not just which
model scores higher, but where each succeeds, what it costs, and whether choosing
between them is worthwhile. Later experiments will examine image degradation,
throughput, and latency under load.

## Built to learn, not just to run

Understanding the work is as important as producing a working pipeline.
The [Jupyter notebooks](notebooks/) are written as workshops: concepts,
step-by-step operations, code, observations, and interpretation belong together.

We document the reasoning behind decisions, how experiments are constructed,
and what the results actually show—including failures and tradeoffs. Separate
[learning notes](docs/README.md#learn-the-methods) explain topics such as
transformers, OCR evaluation, fine-tuning, GPU transfers, and resource profiling.
As notebook code becomes stable, it moves into a small, tested Python package.

The goal is for someone reading the repository to follow how the project was
built and develop the understanding needed to adapt it, not just reproduce its
outputs.

## Progress so far

Dataset exploration, image preparation, and the first OCR path are complete.
The work includes a pretrained baseline, a fine-tuning experiment, a resource
profiling lab, and a reusable implementation in [`src/kyc_router`](src/kyc_router/).

The fine-tuned OCR model correctly extracted **194 of 210 validation fields
(92.38%)**, compared with 139 before adaptation. These results come from one
MIDV-2020 card layout and 15 validation identities; remaining errors and
regressions are documented in the [Phase 3 report](docs/phases/phase-3.md).
The OCR-free comparison path is next.

## Explore the project

- [Project overview and roadmap](PROJECT.md) — the full plan and research scope.
- [Documentation map](docs/README.md) — reports, workshops, and learning material.
- [Application architecture](docs/architecture.md) — how the reusable code fits together.
- [OCR setup and usage](docs/guides/ocr-usage.md) — installation, model artifacts, and running the pipeline.

The notebooks are a good place to follow the learning journey; the documentation
map is a good place to find a particular result or explanation.

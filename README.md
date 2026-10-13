# kyc-router

A learning-first research project comparing crop-based OCR with an OCR-free
identity-document model, then studying simple adaptive routing. One mock MIDV
layout (`alb_id`), seven fields, consumer hardware—not a production KYC system.

**Phases 0–3 complete.** The OCR path is frozen and reusable. Phase 4 is not
started; test identities remain sealed.

- [Roadmap and scope](PROJECT.md) · [Experimental contract](EXPERIMENT_DESIGN.md)
- [Documentation map](docs/README.md) · [Phase 3 delivery](docs/phases/phase-3.md)
- [Architecture](docs/architecture.md) · [OCR setup and usage](docs/guides/ocr-usage.md)
- [Workshop plan](docs/workshops/phase-3.md) · [Executed notebooks](notebooks/)

## Start with the existing environment

```bash
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m unittest discover -s tests -v
```

This installs the `src/kyc_router` package without replacing current libraries.
See the usage guide for a new environment and the verified local model loader.
Weights, MIDV data, and raw experiment outputs are local and Git-ignored; a fresh
clone does not contain the trained checkpoint.

The adopted OCR model scored **194/210 fields (92.38%) on validation**, with
known regressions and 16 remaining errors. This is development evidence, not a
final test result. The delivery record links the complete accuracy, latency,
failure, reproducibility, and learning material.

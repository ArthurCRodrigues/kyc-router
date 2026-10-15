# Use and reproduce the frozen Donut path

The Phase 4 path reads one canonical `alb_id` card and returns the same seven
normalized fields as OCR. Start with the [delivery record](../phases/phase-4.md),
[architecture](../architecture.md), and [comparison evidence](../experiments/ocr-donut-comparison.md)
to understand what is adopted and what remains limited.

## Environment and local artifacts

Reuse the repository `.venv` and research dependencies in `pyproject.toml`.
[OCR setup](ocr-usage.md) documents the established Python/CUDA environment;
[`requirements/phase4-recorded.txt`](../../requirements/phase4-recorded.txt) records
the direct versions actually measured. It is an environment record, not a
portable complete lockfile or a reason to replace working GPU libraries blindly.

If installing the package into a prepared environment:

```bash
.venv/bin/python -m pip install -e '.[research]'
.venv/bin/python -m unittest discover -s tests -v
```

Unit tests require neither MIDV, trained weights, nor a GPU. Whole-image inference
requires restoring the exact trained artifacts named and hashed by
[`configs/donut_pipeline_v1.json`](../../configs/donut_pipeline_v1.json):

```text
checkpoints/phase4_finetuning/20261015T122657718426Z/best/
    config.json
    generation_config.json
    model.safetensors
    processor_config.json
    tokenizer.json
    tokenizer_config.json
    selection_metadata.json
```

These weights and the MIDV data are Git-ignored, not distributed with a fresh
clone or wheel. Copy the exact directory from retained local artifacts/backups
and verify its hashes. The loader fails on missing/corrupt files before invoking
model loaders; it never silently downloads or substitutes another checkpoint.
Downloading the pretrained parent does not recover the fine-tuned model.

## Inference boundary

The caller supplies a **1084 × 680 RGB uint8** NumPy array, already rectified.
Raw photographs, BGR arrays, float arrays, different dimensions, annotations,
identity keys, and reference values are not inference inputs. Donut consumes
the whole array; it requires no field rectangles.

```python
import numpy as np
from PIL import Image
from kyc_router.donut import load_donut_pipeline

canonical_rgb = np.asarray(Image.open("canonical_card.png").convert("RGB"))
pipeline = load_donut_pipeline("configs/donut_pipeline_v1.json", device="cuda:0")
result = pipeline.extract(canonical_rgb)
print(result["fields"])
```

Load once and reuse the pipeline for sequential calls. It owns a concrete
processor/model on the chosen device; there is no server, concurrent scheduler,
training-feature cache, result cache, router, or OCR fallback inside it.
Use `device="cpu"` explicitly for CPU execution. One validation template's CPU
tokens/fields matched the GPU result in notebook D; that is a smoke check, not
a full CPU benchmark or universal cross-device token-equivalence guarantee.

The manifest freezes the checkpoint, tokenizer/schema IDs, PIL preprocessing,
float32/eager behavior, generation arguments, parsing version, shared field rules,
and comparison protocol. The loader validates supported input/decode policy,
artifact hashes, processor settings, schema tokens, and embedding vocabulary size.
Generation arguments explicitly override the checkpoint's forced-EOS default.

## Results, structural failures, and timing

`result` contains:

| Key | Meaning |
|---|---|
| `fields` | Exactly seven normalized keys; unavailable predictions are `None` |
| `raw_fields` | Parsed original values, retaining missing/extra keys; `None` for a whole structural failure |
| `raw_sequence`, `generated_token_ids` | Retained generation evidence, with schema delimiters preserved |
| `parse` | Structural result/errors; no guessed corrections |
| `validation` | Shared field checks plus raw schema completeness; validity is not correctness |
| `eos_emitted`, `length_limit_reached_without_eos` | Stopping diagnostics |
| `inference_latency_ms` | Completed encoder-plus-decoder generation time |
| `latency_ms` | In-memory pixels through preprocessing, transfer, generation, decode, parse, normalization, validation |

Trailing padding IDs alone are removed before parsing; raw token IDs are kept.
Missing EOS, duplicate keys, or malformed structure produce missing predictions
for scoring. Parseable missing fields lose their own credit; extra keys remain
schema violations. Impossible dates are preserved as invalid, never corrected.

Use **`raw_fields`** with the shared scorer so structural/missing/extra-key
failures remain visible. The filled seven-key `fields` mapping is convenient for
callers, but must not erase raw schema evidence during evaluation.

```python
from kyc_router.evaluation import score_document

field_rows, document = score_document(reference_fields, result["raw_fields"])
```

References belong to this evaluation step, never to `extract`. All seven fields
remain in the denominator. Timing excludes dataset/model loading, rectification,
reference scoring, and artifact writes. Do not compare one cold call to the
recorded warm multi-block measurements or describe it as service latency.

## Reproduce the research

From the repository root with MIDV restored:

```bash
.venv/bin/python experiments/prepare_donut_foundations.py
.venv/bin/python -m experiments.execute_phase4_notebook 05b_donut_dataset
.venv/bin/python -m experiments.train_donut --config configs/donut_finetuning_v2.json
.venv/bin/python -m experiments.compare_ocr_donut
.venv/bin/python -m experiments.execute_phase4_notebook 05c_donut_finetuning
.venv/bin/python -m experiments.execute_phase4_notebook 05d_donut_evaluation
```

The training command creates a new run without changing the frozen manifest.
C/D review the retained adopted experiment unless deliberately updated for a
separate experiment. Training feature caches are valid only for exact fixed
encoder weights, pixels, and preprocessing; the comparison never consumes them.
The dataset adapter refuses test access before opening images. Test remains
sealed for the later frozen benchmark/routing evaluation.

To restore artifacts outside the checkout, pass
`artifact_root="/absolute/restored/root"`; the manifest's relative checkpoint path
resolves under that root, independently of the current working directory. The
wheel contains reusable source only, and package imports do not load a model,
select a GPU, execute notebooks, or change thread pools.

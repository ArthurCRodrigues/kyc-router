# Use the Phase 3 OCR pipeline

This guide covers the reusable OCR package. For the adopted model, measured
accuracy, limitations, and verification results, read the
[Phase 3 delivery record](../phases/phase-3.md). For the small design behind it,
read [Application architecture](../architecture.md).

## 1. Install without changing the validated environment

From the repository root, using the existing project environment:

```bash
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m unittest discover -s tests -v
```

The first command installs this package in editable mode without replacing the
already validated libraries. It requires the configured build backend to be
installed. Editable installation makes later source edits visible to imports;
see the [pip documentation](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs).

For a new environment, install an appropriate PyTorch/torchvision build for its
CPU or CUDA setup, then install `.[research]` from the repository. Do not blindly
copy the CUDA build from another machine. The
[recorded Phase 3 environment](../../requirements/phase3-recorded.txt) captures
the measured direct dependencies; it is not a portable, complete dependency
lock. Package requirements are in [pyproject.toml](../../pyproject.toml).

No credentials or network access are needed to load the already present local
checkpoint. Dataset and initial model acquisition are separate setup tasks
documented in the earlier notebooks.

## 2. Check that the adopted weights exist

The [frozen manifest](../../configs/ocr_pipeline_v1.json) records the selected
checkpoint directory, SHA-256 file hashes, crop policy, and generation policy.
`load_pipeline` verifies those files before loading them. It does not silently
download a substitute model. Transformers supports loading a model saved in a
local directory; see its
[pretrained loading reference](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained).

Large trained artifacts are ignored by Git. A fresh clone alone therefore
cannot run the adopted model. Restore the exact matching checkpoint files from
your local experiment artifacts, or reproduce the
[fine-tuning experiment](../experiments/ocr-finetuning.md). A new training run is
a new candidate: do not overwrite the frozen hashes to disguise different weights.
It needs its own comparison and versioned adoption decision.

Relative checkpoint paths resolve from the manifest's repository root, not the
process's working directory. If matching artifacts are stored elsewhere, pass
`artifact_root=Path("/your/artifact-root")`; that directory must contain the
checkpoint path recorded in the manifest.

## 3. Extract one already-canonical document

The example below runs from the repository root. Supply two input files:

- `canonical-card.png`: an already rectified **1084 × 680 RGB** card; this example
  does not locate or rectify an arbitrary photograph.
- `canonical-rectangles.json`: exactly the seven canonical field names, each
  mapped to integer `x`, `y`, `width`, and `height` coordinates in that card.
  Width and height must be positive. Do not include transcriptions or identities.

```python
import json
from pathlib import Path

import numpy as np
from PIL import Image

from kyc_router.ocr import load_pipeline

repo = Path.cwd()
image_path = repo / "canonical-card.png"
rectangles_path = repo / "canonical-rectangles.json"

with Image.open(image_path) as image:
    canonical_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
rectangles = json.loads(rectangles_path.read_text(encoding="utf-8"))

# Load once; reuse this object for subsequent documents.
pipeline = load_pipeline(repo / "configs/ocr_pipeline_v1.json", device="cpu")
result = pipeline.extract(canonical_rgb, rectangles)
print(json.dumps(result, ensure_ascii=False, indent=2))
```

Use `device="cuda:0"` when CUDA is available and intended. There is no automatic
device selection or silent CPU fallback. Invalid input geometry fails before
recognition. RGB versus BGR cannot be inferred from the array's shape: the caller
must supply the correct color order.

For existing MIDV/VIA records, the small adapter
`kyc_router.geometry.rectangles_from_template(record, source_width, source_height)`
scales native template boxes to canonical coordinates. Its `record` is one
image's VIA metadata record, not the complete annotation container. It reads
`field_name` and rectangle geometry only; `value` is not required or consumed.
Source width/height must describe the native template used by those boxes, not
the photograph dimensions.

## 4. Read the result correctly

| Key | Meaning |
|---|---|
| `fields` | Seven normalized textual values; supported dates become `YYYY-MM-DD` |
| `raw_fields` | Decoded model text before normalization; retain this for analysis |
| `field_details` | Crop bounds/shapes and generated token IDs for every field |
| `validation` | Field checks and cross-field date ordering; not proof of correctness |
| `inference_latency_ms` | Sum of completed encoder/decoder generation intervals |
| `latency_ms` | Canonical image to returned result, including crop/processor/transfer/decoding/postprocessing |

Both timings exclude model loading, image-file loading, rectification, reference
scoring, and output writing. CUDA execution is synchronized for completed time.
The established path uses float32 and seven sequential calls, without decoder
KV caching, batching, or a result cache. Resource-lab variants are not defaults.

Names use case folding and preserve diacritics. A malformed date remains visible
and invalid rather than being guessed. A plausible name or nine-digit number can
pass validation and still be wrong.

To score the preceding result, supply a separate `reference-fields.json` with
exactly the seven annotated values. This is offline evaluation, not an inference
input:

```python
from kyc_router.evaluation import score_document

truth = json.loads((repo / "reference-fields.json").read_text(encoding="utf-8"))
field_scores, document_score = score_document(truth, result["raw_fields"])
print(document_score)
```

Keep scoring outside the request timer. Using `raw_fields` preserves both raw
and normalized comparisons; passing the entire result object would violate the
seven-field scoring schema. Multi-document aggregation is provided by
`summarize_scores`; see its docstring and the
[evaluation guide](../learning/ocr-evaluation.md) for denominator rules.

## 5. Verify or reproduce

Unit tests above cover field rules, geometry, scoring, mocked recognition, and
artifact loading failures. They need neither the MIDV dataset nor a GPU.
The local validation replay is a separate, explicitly invoked check:

```bash
.venv/bin/python -m experiments.verify_ocr_source --device cuda:0
```

It uses the existing validation workload and local weights to compare package
behavior with the retained notebook evidence. It does not fit a model or open
test identities. Use `--device cpu` when CUDA is not intended. This check also
needs the local validation artifacts from the recorded fine-tuning run and the
research dependencies. See the delivery record for its results and artifacts.

Historical experiment reproduction commands remain in their reports:
[pretrained baseline](../experiments/ocr-baseline.md),
[fine-tuning](../experiments/ocr-finetuning.md), and
[resource lab](../experiments/ocr-resource-profile.md). Notebook reruns create
fresh runs; rerunning the fine-tuning notebook trains again. Read the report and
policy before choosing Run All.

`OcrPipeline` is a sequential local extraction component, not a thread-safe
serving system. No endpoint, concurrent scheduler, routing policy, synthetic
benchmark sweep, or future-phase implementation is required to use it.

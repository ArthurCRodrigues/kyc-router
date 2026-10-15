# Application architecture

## Decision: a small Python package, two frozen extraction paths

Phase 3 established the installable [`kyc_router`](../src/kyc_router/) package;
Phase 4 adds verified whole-card Donut inference and tagged-sequence parsing.
Both paths reuse the same field rules and scorer. The [roadmap](../PROJECT.md)
still owns what comes next; the [experimental contract](../EXPERIMENT_DESIGN.md)
owns what the experiment means.

```text
Caller: canonical RGB image + seven canonical rectangles
                          |
                    OcrPipeline.extract
                          |
       crops → processor → TrOCR → raw text
                          |
                normalize → validate
                          |
          JSON-ready result, details, and timings
```

The caller prepares the canonical image. Phase 3 recognition does not load MIDV
images, find a document, rectify photographs, read identity labels, or choose a
route. Ground truth belongs to evaluation, never to the recognizer's inputs.

The Donut boundary is similarly explicit:

```text
Caller: canonical RGB image
             |
     DonutPipeline.extract
             |
PIL processor → full image encoder/decoder → raw tokens/sequence
             |
strict parse → shared normalize/validate → fields + raw evidence + timings
```

It receives no field rectangles, identity/reference, or training-feature cache.
The seven-key normalized `fields` mapping is convenient for callers; `raw_fields`
retains schema omissions/extras or `None` for structural failure so the shared
scorer does not lose failure evidence. Validators check shape/calendar rules;
neither extractor uses a lookup table or correction fallback.

## Files and responsibilities

| File | Responsibility |
|---|---|
| [`schema.py`](../src/kyc_router/schema.py) | Seven required names, inference order, and MIDV name mapping |
| [`geometry.py`](../src/kyc_router/geometry.py) | Validate image/rectangles, scale known boxes, and crop with the established padding |
| [`normalization.py`](../src/kyc_router/normalization.py) | Shared Unicode, whitespace, case, number, sex, and explicit date rules |
| [`validation.py`](../src/kyc_router/validation.py) | Simple `alb_id` field and date-order checks; no guessed corrections |
| [`ocr.py`](../src/kyc_router/ocr.py) | Verify/load local artifacts once and run seven sequential recognition calls |
| [`serialization.py`](../src/kyc_router/serialization.py) | Versioned task/field envelope, escaping, strict parsing, and explicit duplicate/truncation failures |
| [`donut.py`](../src/kyc_router/donut.py) | Verify local model/tokenizer/processor, run one whole-image generation, parse and return shared-rule results |
| [`evaluation.py`](../src/kyc_router/evaluation.py) | Exact Match, edit distance, CER, and complete-document scoring outside inference |

`rectangles_from_template` is a small geometry adapter for existing VIA records.
It reads field names and rectangle coordinates, not their `value` transcriptions.
The pipeline itself receives the resulting canonical rectangles, not an annotation
record. This separates dataset conventions from recognition without inventing a
dataset framework.

## Why these choices are enough

- **Functions for stateless operations.** Cropping and field rules have explicit
  arguments and return values. They are easy to read and test without a model.
- **One concrete class per expensive path.** `OcrPipeline` and `DonutPipeline` keep a processor,
  model, device, and fixed policy alive across requests. It does not reload
  weights for every document. There is no base pipeline class, plugin registry,
  dependency container, or abstract factory.
- **Composition, not inheritance.** The extraction method calls the small
  functions in order. That is the whole coordination mechanism.
- **Explicit policy and failures.** Loading verifies versioned local artifacts;
  a missing checkpoint fails rather than silently downloading or substituting
  another model. The caller chooses CPU or CUDA. Imports do not change global
  thread pools or allocate GPU memory.
- **Preserve measured behavior.** OCR retains float32, greedy
  decoding, maximum length 20, disabled KV caching, sequential field calls, and
  the established crop/normalization rules. Existing pandas scalar missing-value
  semantics are retained instead of introducing a new compatibility change.
  Profiling variants remain experiments, not hidden defaults.

Donut uses the saved PIL landscape processor, float32/eager generation, a trained
single task token, fixed greedy/cache/length settings, and strict `alb_id_tags_v1`
parsing. Values and artifact hashes belong in its versioned manifest. Direct
generation arguments override the checkpoint's forced-EOS default. Inference
always runs the whole encoder, even though training cached fixed encoder features.

Dataset adapters, optimizer/resource probes, training, source parity, and timing
remain explicit research scripts under `experiments/`. The historical
`donut_protocol.py` is retained as the extraction reference; the package imports
neither that module nor any notebook. Package imports allocate no GPU memory,
load no dataset/model, and impose no global thread settings.

The `src/` layout keeps importable code separate from notebooks and experiment
scripts. Installing the package makes tests exercise its package boundary rather
than relying on the repository's current directory. This follows the
[Python Packaging Authority's layout guidance](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/).

Inference uses both evaluation mode and `torch.inference_mode()`: they control
different behavior. Completed CUDA timings synchronize the device. These choices
follow [PyTorch's inference guidance](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)
and its [synchronization API](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html).
The recognizer uses the documented
[VisionEncoderDecoder loading and generation interface](https://huggingface.co/docs/transformers/en/model_doc/vision-encoder-decoder).

## What is intentionally absent

The package contains no router, quality-feature pipeline, degradation benchmark
sweep, HTTP server, queue, result/training-feature cache, distributed worker,
scheduler, or deployment layer. Training-time frozen features belong only to
the recorded research experiment, not the public inference API.
The project does not currently need distributed infrastructure.
Later phases can call the same extraction boundary; they do not need speculative
interfaces or empty folders now.

Historical notebooks retain their executed code and provenance. The new package
is checked against that behavior; notebook results are not retroactively
regenerated to make the refactor appear correct. See the
[Phase 3 delivery record](phases/phase-3.md) for the actual verification and
[OCR usage guide](guides/ocr-usage.md) for setup and calls.

The [Phase 4 record](phases/phase-4.md) and [Donut usage guide](guides/donut-usage.md)
cover the second path. The [comparison report](experiments/ocr-donut-comparison.md)
records exact reload/research/source parity, the shared input boundary, repeated
timings, and their limits. This is a comparator for later routing research,
not a production KYC service.

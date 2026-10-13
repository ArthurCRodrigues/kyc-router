# Application architecture

## Decision: a small Python package, one working extraction path

Phase 3 turns the verified notebook operations into the installable
[`kyc_router`](../src/kyc_router/) package. The package contains only the OCR path
and the shared field rules already needed today. The [roadmap](../PROJECT.md)
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

## Files and responsibilities

| File | Responsibility |
|---|---|
| [`schema.py`](../src/kyc_router/schema.py) | Seven required names, inference order, and MIDV name mapping |
| [`geometry.py`](../src/kyc_router/geometry.py) | Validate image/rectangles, scale known boxes, and crop with the established padding |
| [`normalization.py`](../src/kyc_router/normalization.py) | Shared Unicode, whitespace, case, number, sex, and explicit date rules |
| [`validation.py`](../src/kyc_router/validation.py) | Simple `alb_id` field and date-order checks; no guessed corrections |
| [`ocr.py`](../src/kyc_router/ocr.py) | Verify/load local artifacts once and run seven sequential recognition calls |
| [`evaluation.py`](../src/kyc_router/evaluation.py) | Exact Match, edit distance, CER, and complete-document scoring outside inference |

`rectangles_from_template` is a small geometry adapter for existing VIA records.
It reads field names and rectangle coordinates, not their `value` transcriptions.
The pipeline itself receives the resulting canonical rectangles, not an annotation
record. This separates dataset conventions from recognition without inventing a
dataset framework.

## Why these choices are enough

- **Functions for stateless operations.** Cropping and field rules have explicit
  arguments and return values. They are easy to read and test without a model.
- **One concrete class for expensive state.** `OcrPipeline` keeps a processor,
  model, device, and fixed policy alive across requests. It does not reload
  weights for every document. There is no base pipeline class, plugin registry,
  dependency container, or abstract factory.
- **Composition, not inheritance.** The extraction method calls the small
  functions in order. That is the whole coordination mechanism.
- **Explicit policy and failures.** Loading verifies versioned local artifacts;
  a missing checkpoint fails rather than silently downloading or substituting
  another model. The caller chooses CPU or CUDA. Imports do not change global
  thread pools or allocate GPU memory.
- **Preserve measured behavior.** The adopted path retains float32, greedy
  decoding, maximum length 20, disabled KV caching, sequential field calls, and
  the established crop/normalization rules. Existing pandas scalar missing-value
  semantics are retained instead of introducing a new compatibility change.
  Profiling variants remain experiments, not hidden defaults.

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

No Donut implementation, router, quality-feature pipeline, benchmark sweep,
HTTP server, queue, cache, distributed worker, scheduler, or deployment layer is
created here. The project does not currently need distributed infrastructure.
Later phases can call the same extraction boundary; they do not need speculative
interfaces or empty folders now.

Historical notebooks retain their executed code and provenance. The new package
is checked against that behavior; notebook results are not retroactively
regenerated to make the refactor appear correct. See the
[Phase 3 delivery record](phases/phase-3.md) for the actual verification and
[OCR usage guide](guides/ocr-usage.md) for setup and calls.

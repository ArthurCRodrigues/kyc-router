# OCR-free extraction: concepts and the executed foundations

**Updated 2026-10-15.** The full Phase 4 sequence has executed. Start with
[05a — Donut foundations](../../notebooks/05a_donut_foundations.ipynb); the
[experiment report](../experiments/donut-feasibility.md) owns measured results.
Continue with [B — targets and feasibility](../../notebooks/05b_donut_dataset.ipynb),
[C — fine-tuning](../../notebooks/05c_donut_finetuning.ipynb), and
[D — comparison/extraction](../../notebooks/05d_donut_evaluation.ipynb).
The [training guide](donut-training.md) develops adaptation/optimizer concepts;
the [phase plan](../workshops/phase-4.md) maps all delivered work.

## What changes from TrOCR?

Phase 3 gives the recognizer seven known crops and tells the surrounding code
which field each crop represents. Donut receives the whole rectified card and
must generate both field names and values. For example, it must associate the
visible date of birth with `date_of_birth` rather than `expiry_date`.

"OCR-free" means there is no separately executed OCR stage. The model still
has to recover text from pixels. It may misread a character, assign text to the
wrong field, or generate an incorrect structure. The
[Donut paper](https://arxiv.org/abs/2111.15664) describes this image-to-sequence
approach; the [Transformers documentation](https://huggingface.co/docs/transformers/model_doc/donut)
describes its Swin vision encoder and BART-family text decoder.

## Encoder, decoder, and attention

The image processor converts a canonical RGB array into the tensor expected by
the encoder. Resizing, padding, orientation, and numerical normalization matter:
a card can have the correct outer dimensions while its small text has become
unreadable. Sections 2–4 of [05a](../../notebooks/05a_donut_foundations.ipynb)
display the canonical and processed images, including a diagnostic text detail.
The common 1084 × 680 input remains the same for both pipelines; Donut's internal
processor policy is recorded separately in the investigation configuration.

The canonical array is height × width × channels in RGB uint8. Processing adds
a batch dimension and changes it to batch × channels × height × width in
floating point. Pixel rescaling/normalization prepares neural-network input;
field-string normalization standardizes dates and names after recognition. They
are different operations. An inverse normalization reconstructs a displayable
processed image so we can check what the network actually sees.

The pinned native processor uses a portrait canvas and aligns the long axis by
rotating landscape inputs. The investigation compares that behavior with a
smaller landscape canvas and rotation disabled. Setting a tensor's dimensions
is therefore not enough: inspect actual orientation, aspect ratio, centered
padding, and small text. The native dimensions appear as a historical list in
the saved file; the resolved named height/width and explicit PIL backend remove
ambiguity. In A these were exploratory settings; Phase 4 later selected the
landscape policy after complete-step resource checks and froze it in the
[adopted manifest](../../configs/donut_pipeline_v1.json).

Swin builds visual representations using local attention windows and a
hierarchy of spatial scales. The decoder uses cross-attention to access those
image representations while generating a token sequence. Tokens are vocabulary
units, not necessarily individual letters or whole words.

Generation is autoregressive: each selected token becomes context for the next.
An early wrong value or delimiter can affect later fields. A plausible name or
valid date can therefore be wrong even when the sequence parses successfully.
See the [Swin paper](https://arxiv.org/abs/2103.14030) for the image architecture;
sections 6–7 of [05a](../../notebooks/05a_donut_foundations.ipynb) trace the actual
encoder, first decoder decision, and generated tokens.

### Patches, windows, and hierarchical scales

For the notebook's landscape tensor, 4 × 4 patches begin a 160 × 256 spatial
grid. A patch representation describes a small pixel region; it is not a detected
character. Local attention connects positions within a window. Shifting windows
in alternating blocks lets positions interact across the preceding window
boundaries. This is a way to build contextual image representations while
restricting each local attention computation.

Patch merging then reduces grid height/width and increases the channel width.
In this checkpoint's four stages, the grids become 160 × 256, 80 × 128,
40 × 64, and 20 × 32. The final grid is flattened into a sequence for the text
decoder. Notebook A calculates the stages from configuration and checks the
real encoder output against the expected final shape.

`[1, 640, 1024]` at the encoder boundary means one image, 640 spatial feature
positions, and 1024 feature channels per position. Do not confuse it with
`[1, 3, 640, 1024]` at the image-input boundary. Neither 640 positions nor 1024
channels count recognized text characters. Other processor dimensions can
change grids/padding; derive and inspect them rather than hard-coding this example.

### Decoder cross-attention and the next-token distribution

The decoder uses the token prefix as context and queries the image features
through cross-attention. The encoder features supply information about the
whole card; the decoder must learn how to use it for the requested task. This
does not provide a guaranteed alignment between a field token and its printed
box, nor does it certify that a generated value came from visible text.

For one prompt token, notebook A's first decoder call has one logit per
vocabulary entry. Softmax converts those logits to a next-token distribution.
Greedy decoding selects the largest score, appends the selected token, and
continues. The displayed top-eight candidates explain that local decision;
their probabilities are not calibrated seven-field confidence. Small token
errors can accumulate through subsequent self-generated context. Template and
photo outputs in the experiment report illustrate that the same identity's
two views can produce very different sequences.

## Structured targets are still sequences

A Python dictionary is not directly a model label. Serialization converts the
seven-field mapping into a deterministic sequence. A task token tells the
decoder which output task to perform; field delimiters identify where values
start and end. The [official Donut implementation](https://github.com/clovaai/donut)
uses JSON-derived targets and conversion between structured objects and tokens.

For this project, field names come from
[`schema.REQUIRED_FIELDS`](../../src/kyc_router/schema.py). We must define field
order, delimiters, value representation, escaping, and an end token. New schema
tokens require corresponding model embeddings and must be saved with the
tokenizer. A correct training target must survive a serialize/parse round trip
without losing diacritics or punctuation. Token length must be measured so a
limit does not silently remove the last fields.

Section 5 of [05a](../../notebooks/05a_donut_foundations.ipynb) compares existing
pretraining task tokens with proposed `alb_id` task/field tags. It adds tokens
only to a separate tokenizer probe. That changes which symbol IDs the tokenizer
can emit; it does not change learned weights or teach the model to produce the
corresponding fields. New IDs require embedding rows in the neural model, then
training. Sending a new ID to the unchanged model can exceed its embedding
table. Model/tokenizer vocabulary sizes are checked explicitly before inference.

Inference reverses the process: generated token IDs become a raw sequence,
then a parser produces a mapping. Only after parsing do the existing
[normalization](../../src/kyc_router/normalization.py) and
[validation](../../src/kyc_router/validation.py) rules apply. Decoding must retain
the delimiters needed by that parser; blindly removing every special token can
erase structure. [Notebook B](../../notebooks/05b_donut_dataset.ipynb) exercises
that boundary, including token round trips and adversarial parser failures.

An unparseable sequence is a structural failure. Missing fields remain missing,
extra fields are recorded, and malformed dates are not guessed. Duplicate keys
need a declared policy because a dictionary can otherwise hide an earlier value.
The [shared scorer](../../src/kyc_router/evaluation.py) supplies the established
missing-field and structural-failure denominators; raw sequences remain evidence.

## Generation limits, end tokens, and cache

An end-of-sequence token (EOS) can tell generation to stop before its maximum
length. A length limit instead bounds computation even when the model has not
finished. Reaching that limit without EOS is evidence of truncation, which must
remain visible. `max_new_tokens` counts newly generated tokens; a total
`max_length` also counts the existing prefix. Section 7 of
[05a](../../notebooks/05a_donut_foundations.ipynb) uses an explicit new-token bound
and retains the prefix, every generated ID, and the observed stopping reason.

The pinned checkpoint also has a forced-EOS default: it can insert EOS at the
limit regardless of the model's preferred token. That cannot establish voluntary
termination. In the installed Transformers 5.19 implementation, `None` values
inside a supplied `GenerationConfig` are filled from checkpoint defaults; direct
generation keyword arguments apply afterward. Notebook A therefore passes the
explicit forced-EOS override directly and displays the inherited setting for
comparison. Its report preserves the initial faulty attempt and the corrected
result. This is a version-specific API observation, not a general rule for every
Transformers version; record and inspect defaults when reproducing elsewhere.

The decoder KV cache retains attention keys/values within a single generation,
avoiding some repeated computation as the token prefix grows. It is not a cache
of a previous document's answer and does not skip encoding a new document. It
uses memory, so cache/precision/length belong in the recorded inference policy.
For the shared autoregressive and cache concepts, see the existing
[generation/timing guide](ocr-generation-and-timing.md).

## Training loss and generated answers answer different questions

Fine-tuning starts with pretrained weights and adapts them to paired card images
and serialized answers. With teacher forcing, the decoder sees the correct
preceding target tokens while learning the next one. Padding positions are
ignored in the loss; label shifting aligns the decoder input with the next-token
target. Prompt handling must be explicit so the model learns the intended answer.

At inference the correct preceding answer is unavailable. The decoder uses its
own predictions, so a low teacher-forced validation loss does not guarantee a
correct generated document. Checkpoint selection uses free-running generation,
parsing, and the shared field metrics. A tiny training-only memorization exercise
checks wiring; accuracy on held-out identities checks a different property.
[Notebook C](../../notebooks/05c_donut_finetuning.ipynb) demonstrates that distinction
with the actual tiny diagnostic and complete retained training history.

## Why memory feasibility comes first

Inference needs weights and temporary computation. Training also needs gradients,
saved activations, and optimizer state. A successful forward pass therefore does
not establish that backward and the optimizer step fit. The Phase 3 GPU had
3.64 GiB visible memory; Phase 4 must measure its actual training requirement.

Smaller internal image tensors can reduce activation memory while sacrificing
text detail. Freezing parameters reduces trainable state while limiting
adaptation. Gradient checkpointing recomputes some activations during backward
to trade time for memory. Mixed precision changes numerical representation and
must be checked on the actual hardware. Gradient accumulation combines several
microbatch gradients; it cannot rescue a single example that already exceeds
memory. See [PyTorch checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
and [automatic mixed precision](https://docs.pytorch.org/docs/stable/amp.html).

The feasibility workshop measured complete optimizer steps and retained failed
attempts. Its selected compromise freezes the image encoder and uses all-decoder
Adafactor updates on deterministic features. See [training concepts](donut-training.md)
and the [feasibility evidence](../experiments/donut-feasibility.md) for why the
earlier AdamW probes were insufficient. Training time, inference time, and
end-to-end request time remain separate measurements.

### What the notebook's memory and timings actually measure

Section 1 defines and section 8 records process RSS, CUDA allocated/reserved
memory, peak counters, and device free memory. Allocated counts live tensors;
reserved also includes allocator blocks held for reuse. Device free memory can
change because of other processes, while process RSS includes notebook imports,
CPU buffers, and figures. A peak is measured since the last reset; a peak that
includes model weights is not an incremental activation measurement. See the
[resource-profiling guide](ocr-resource-profiling.md) for the shared accounting
concepts and PyTorch's memory-management references.

`eval()` sets model evaluation behavior such as disabling dropout;
`torch.inference_mode()` prevents gradient bookkeeping. CUDA operations can be
asynchronous, so synchronization completes them at timing boundaries. The
foundations notebook separates preprocessing, transfer, full model generation,
and string decoding. A's historical smoke timing excludes disk/geometry/model
setup and omits the later strict parser/validation. Two smoke calls, after earlier inspection, cannot
establish a stable latency distribution or a performance advantage over OCR.
The executed [comparison notebook](../../notebooks/05d_donut_evaluation.ipynb)
now includes the complete parsed/normalized/validated path under a matched,
warmed, repeated protocol, with both session measurements retained.

## What would make this phase useful?

The objective is a reproducible second path and an honest comparison. It may
recover fields OCR misses, introduce different errors, or fail to offer a useful
accuracy/cost tradeoff. We scored the same validation identities and views,
retained structural failures, and compared matched timings in
[05d](../../notebooks/05d_donut_evaluation.ipynb). The
[comparison report](../experiments/ocr-donut-comparison.md) records the resulting
tradeoffs; test stays sealed for the later frozen evaluation.

The [experimental contract](../../EXPERIMENT_DESIGN.md) owns the seven-field
metrics and comparison rules. Phase 4 provides the extraction path; Phase 5
investigates complementarity across the controlled benchmark, and Phase 6 asks
whether a router can exploit it.

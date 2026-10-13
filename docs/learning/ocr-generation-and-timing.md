# TrOCR generation and the cost of an OCR request

This note supports [Workshop 7 and the Workshop 8 processor audit](../../notebooks/04c_ocr_evaluation.ipynb).
Start with the [OCR overview](OCR.md) for detection, recognition, and processors.
Architecture details follow the [TrOCR documentation](https://huggingface.co/docs/transformers/model_doc/trocr)
and [original paper](https://arxiv.org/abs/2109.10282). Exact dimensions below come
from the project's pinned checkpoint configuration, not a generic TrOCR variant.

## Representations through the model

Our input is a small RGB crop containing one field. The processor converts it
into a float32 tensor shaped `[1, 3, 384, 384]`: one image, three channels, height,
and width. Resize and normalization are necessary to match the checkpoint's
training interface, even when the crop is much wider than it is tall.

The pinned DeiT encoder uses 16×16 patches. A 384×384 image supplies a 24×24 grid,
or 576 patch positions. Two special positions bring the encoder sequence length
to 578. The encoder's hidden size is 384, so its output is `[1, 578, 384]`. These
are visual feature vectors, not 578 recognized characters.

Patch embeddings combine learned image features with positional information.
Position matters because the same stroke in another location can belong to a
different character or a different part of the line. The encoder has 12 layers
and six attention heads per layer in this checkpoint.

Resizing a wide line to a square changes its aspect ratio. That is the established
processor behavior for this baseline. It is a possible sensitivity to examine,
not evidence of a bug by itself. The notebook reverses numerical normalization
for display so the actual resized input can be inspected. This visualization
does not reconstruct details lost during resizing.

## What attention computes

Attention lets one position combine information from other positions. Learned
linear projections turn vectors into queries, keys, and values. A query describes
what the current position seeks; keys help match it to other positions; values
carry the information being combined.

For one attention head:

```text
weights = softmax(Q Kᵀ / sqrt(key_dimension) + mask)
output  = weights V
```

The matrix product compares queries with keys. Softmax turns the scores in each
row into nonnegative weights summing to one. The weighted sum combines values.
The scale moderates the size of the dot products. A mask rules out prohibited
connections by making their weight zero.

For intuition, suppose a visual position gives three other positions weights
`[0.7, 0.2, 0.1]`. It incorporates their value vectors with those proportions.
The weights are computed from the current input; they are not a fixed character
dictionary. Multiple heads can learn different relationships. Layers also use
feed-forward transformations, residual connections, and normalization to refine
the representations.

An attention map alone does not prove why the model produced a character. It
shows one part of a much larger learned computation. Our failure analysis uses
observable images and strings before making causal claims.

## The decoder has two sources of context

The pinned decoder has six layers, hidden size 256, eight attention heads per
layer, and a vocabulary of 64,044 token IDs. It uses:

- causal self-attention over the generated text prefix;
- cross-attention to the encoded visual sequence.

Causal masking prevents an output position from reading future output tokens.
Cross-attention lets the decoder query the image representation. The decoder
can therefore combine what has already been generated with the visual evidence
for what comes next.

At each generation step the model produces vocabulary logits. With greedy
decoding, it selects the highest-scoring next token. It appends that token and
repeats until EOS or the configured length limit. Tokenization can represent
whole pieces of words, punctuation, and other fragments; token count is not
character count.

The start token and EOS both have ID 2 in this checkpoint. Seeing ID 2 at the
first position therefore does not mean generation has immediately finished.
The notebook checks the generated suffix when diagnosing a length limit. Padding
and start/end markers are removed during string decoding, while the original
token IDs are saved for audit.

Context can help interpret ambiguous strokes, but it can also favor a plausible
sequence unsupported by the pixels. A generated name or date must still be
compared with truth. Softmax probabilities describe the model's distribution;
they are not automatically calibrated probabilities of field correctness.

## Why seven fields cost seven generations

The pipeline crops each field and invokes the recognizer sequentially. Every
field gets its own processor conversion, encoder pass, autoregressive decoding,
and string decoding. One document request is therefore seven recognition calls,
not one.

Longer generated sequences generally require more decoder steps. The encoder
cost and other overhead remain even for a one-character sex field. Hardware
state, tokenization, stopping behavior, and CPU work also affect latency, so
field length alone does not determine request time.

Batching several crops could improve throughput, but it would be a separate
execution strategy. The baseline uses a single crop per model call and a single
document request at a time. Later comparisons need to state whether they change
that policy.

## Decoder caching and the preserved baseline

At inference time, previous text tokens do not change. A key/value cache can
reuse attention projections for earlier decoder positions rather than rebuilding
them from the full growing prefix each step. Depending on implementation,
encoder-side cross-attention projections may also be reused. Caching trades
memory for less repeated computation. It is not a cache of recognized documents.

The checkpoint's original generation configuration has `use_cache=False`; `04b`
inherits it. Workshops 5–8 preserve that behavior, make it explicit, and record
the complete generation configuration. The decoder's underlying configuration
also records this setting.

Turning caching on would be a legitimate subsequent performance experiment.
It should retain the uncached baseline, verify predictions, record memory, and
repeat the same latency protocol. It is not assumed to solve recognition errors
or to improve accuracy. This notebook does not silently substitute that variant.

## Evaluation mode, inference mode, and precision

`model.eval()` changes training-specific layer behavior, such as dropout. It does
not by itself disable gradient tracking. `torch.inference_mode()` disables
autograd bookkeeping and some tensor tracking overhead for inference. It does
not by itself switch the model into evaluation mode. We use both.

The baseline uses float32 weights and inputs on the GTX 1650. Lower precision
can reduce memory and sometimes computation, but hardware support and numerical
behavior matter. A precision change requires its own recorded run and output
comparison. Peak allocated and reserved GPU memory are retained in metadata;
they measure PyTorch's allocator, not all memory consumed by the desktop or other
processes.

## Why GPU timing needs synchronization

CPU code can enqueue GPU work and continue before the GPU finishes. A wall-clock
timer that stops immediately after enqueueing can measure submission rather than
completed computation. `torch.cuda.synchronize(device)` waits for queued work
on that device to finish. See the
[PyTorch synchronization reference](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html).

The inference timer begins after processor conversion and transfer, with prior
GPU work synchronized. It surrounds `model.generate`, then synchronizes before
stopping. Seven such intervals are summed. Those intervals include both the
vision encoder and text generation; they are not decoder-only measurements.

The outer request timer starts from a preloaded canonical RGB card and includes
crop extraction, processor work, transfer, all generation calls, decoding,
normalization, and prediction validation. It stops after a structured result is
ready. File loading, model loading, rectification to the common canonical input,
comparison with ground truth, and artifact writing are excluded.

Nested synchronization adds some overhead and can prevent overlap. The protocol
is transparent rather than presented as maximum hardware throughput. Before
comparing OCR with another pipeline, use the same input boundary and measurement
conditions. A later service benchmark can add raw-image processing and queueing.

## Warm-up, repetitions, and interpretation

First requests can include lazy initialization, allocator growth, or other cold
costs. We discard three full-document warm-ups per source. We then measure every
validation identity/source observation three times, shuffling order with a fixed
seed each repetition. All repeated outputs are checked against the accuracy run.

Mean and median describe different aspects of the measured workload. A few slow
requests pull up the mean; the median describes its middle. We retain every raw
timing, the repetition, and order position to inspect variation rather than
discarding inconvenient values.

These are warm, sequential, local measurements. They do not estimate queueing
latency under concurrent load, cold-start latency, or reliable p95/p99 service
tails. Those questions belong to the later serving phase. Repeating inference
does not turn 15 validation identities into a larger accuracy dataset.

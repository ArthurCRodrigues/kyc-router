# Training Donut on seven fields

This guide accompanies the Phase 4 data/training work. The notebooks expose
concrete examples; the experiment reports own measured results. The experimental
contract still owns normalization, the identity split, and scoring.

Execution map: [B](../../notebooks/05b_donut_dataset.ipynb) covers grammar, token
round trips, label alignment, and actual resource records; [C](../../notebooks/05c_donut_finetuning.ipynb)
reviews real optimization, free generation, selection, and all epoch replays;
[D](../../notebooks/05d_donut_evaluation.ipynb) verifies saved/source inference.
See [fine-tuning evidence](../experiments/donut-finetuning.md) for measured results.
The extracted protocol is in [serialization.py](../../src/kyc_router/serialization.py),
and the public whole-image request is in [donut.py](../../src/kyc_router/donut.py).

## Serialization is part of the task

A network predicts token IDs rather than dictionaries. We turn the normalized
seven-field mapping into a fixed-order tagged sequence. `<s_alb_id>` identifies
the task, `<s_surname>...</s_surname>` bounds one value, and `</s>` ends the
document. Existing vocabulary pieces still express names, numbers, and dates;
15 additional tokens express the task and seven pairs of field boundaries.

The same tokenizer must be saved with the adapted model. Registering schema
tokens increases vocabulary size, so the decoder embedding/output tables must
be resized. New rows start without learned task meaning. The training objective
teaches when those tokens belong and how to generate their contents from pixels.

Targets use the existing normalization rules, including NFC and preserved
diacritics. XML-like delimiters are structural, so literal `&`, `<`, and `>` in
values are escaped as canonical HTML entities. A serialize/parse round trip
must recover every character. No identity values become categorical lookup tokens.
See `experiments/donut_protocol.py` for the investigation's explicit functions.

The parser accepts well-formed fields in any order, retaining missing and extra
keys for the shared scorer. It rejects duplicate keys and malformed or incomplete
envelopes as whole-sequence structural failures. A generated date can be
structurally parseable and still fail semantic validation. This distinction keeps
grammatical errors, schema omissions, and incorrect values observable separately.

## Teacher forcing and the single shift

For tokens `[task, open_surname, name_piece, close_surname, EOS]`, decoder inputs
are all but the last token. Labels are all but the first. Each position learns
to predict the next token using the correct preceding target context and the
image's encoded features. The initial task token is context, not a predicted
answer token; EOS is a supervised target.

The installed mBART causal decoder compares aligned logits/labels, so our
explicitly shifted input/label pair must not be shifted again. Cross entropy
penalizes a wrong next-token distribution. Notebook B inspects the alignment
and later training code uses that same function. If batching introduces padding,
padding labels must be replaced by `-100` and the decoder attention mask must
identify real positions. Our initial microbatch is one variable-length sequence,
so there is no padding to ignore and no hidden truncation.

At inference, preceding target tokens are unknown: the model sees its own
generated prefix. Low teacher-forced loss can coexist with repeated text, broken
tags, or wrong fields. Checkpoint selection therefore uses free-running generated
answers through the actual parser and shared Exact Match/CER scorer.

## What full training needs in memory

Float32 weights use four bytes per parameter. A trainable parameter generally
also needs a gradient and AdamW's two moment estimates: roughly sixteen bytes
per parameter before saved activations, buffers, and temporary workspaces.
The moments are often initialized on the first optimizer step, so forward or
backward fitting is insufficient evidence. Measure multiple complete steps.

The resource investigation also found that a two-step success with AdamW was
not enough headroom for a later diagnostic under changing desktop GPU use.
The selected follow-up uses **Adafactor**: matrix-shaped second-moment statistics
are approximated with row/column summaries rather than a full matrix, and no
first-moment estimate is used. This greatly reduces optimizer-state storage for
the large vocabulary embedding while keeping all decoder parameters trainable.
It changes the optimization experiment, so it has its own configuration and
ten-step resource check; it is not described as AdamW with identical behavior.

We use explicit learning rates with `scale_parameter=False` and
`relative_step=False`, plus an external warm-up/decay schedule. Adafactor's
`clip_threshold=1` clips the RMS of its preconditioned update. We record gradient
norms but do not additionally apply AdamW-style global gradient clipping. See the
[optimizer documentation](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.Adafactor)
and [Adafactor paper](https://arxiv.org/abs/1804.04235). Lower state memory is a
resource finding; it does not establish better field accuracy.

Gradient checkpointing retains fewer activations and recomputes them during
backward. It trades computation for memory; it does not remove parameter or
optimizer-state storage. Gradient accumulation combines microbatch gradients
to achieve a larger effective batch; it does not make one oversized microbatch
fit. Clipping bounds the total gradient norm to reduce unusually large updates;
the recorded norm is the value before clipping.

If full-model float32 training cannot fit, freezing the image encoder removes
its gradients/moments and prevents its weights from changing. The decoder can
still learn structured extraction from pretrained whole-image features. This
is a different adaptation policy, not equivalent to full fine-tuning, and its
limitations must appear in the report and model manifest.

A fixed encoder with deterministic preprocessing permits precomputing its
features for training. Such features are a compute cache of permitted images,
not reference labels. They cannot be reused if encoder weights or preprocessing
change, and inference still needs to encode each new image. Repeated timing
comparisons must not use this training-only feature cache. A feature produced
under inference mode must be materialized as a normal detached tensor before
it participates in an operation that saves inputs for decoder backpropagation.

An early smoke script in this project selected a feature by positional index
without binding it to the requested image. Its matching generated tokens were
therefore invalid as input/cache parity evidence; the audit preserves that
correction. Cache keys must include identity, source view, split, and pixel hash,
then bind the encoder/preprocessor version too. The corrected check compares the
encoder feature tensor itself before comparing tokens. A model can emit the
same answer for different inputs, so output agreement alone cannot validate a
cache. Main training's complete metadata check and the final 30-view inference
verification used the proper bindings; see the experiment report for the distinction.

## Tiny overfit checks and bounded selection

Deliberately memorizing a very small training-only set checks whether target
tokens, gradients, and generation are wired together. It does not demonstrate
generalization; paired views can even share every target. Reset pretrained
weights, vocabulary initialization, seed, and optimizer before the main run so
this diagnostic is not an undeclared extra training stage.

The main experiment uses only training identities for updates. Validation
selects the best generated-field metric, then pooled CER, then the earlier
candidate. Epoch budgets and early stopping are declared before selection.
Training loss remains diagnostic. One layout, constant nationality, only 15
validation identities, and repeated development selection limit any broad claim.
No field-correction table or identity lookup is introduced when output is wrong.

Primary references: [Donut](https://github.com/clovaai/donut),
[vision encoder-decoder models](https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder),
[PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html),
[activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html).

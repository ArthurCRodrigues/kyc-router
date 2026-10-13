# Evaluating OCR: from characters to complete records

This note accompanies Workshops 5–8 in
[04c_ocr_evaluation.ipynb](../../notebooks/04c_ocr_evaluation.ipynb).
The authoritative rules are in the [experiment contract](../../EXPERIMENT_DESIGN.md#12-accuracy-metrics).

## What is one observation?

An observation is one identity document from one source: a clean template or a
rectified photograph. It has seven required fields. An identity contributes two
observations in this validation experiment, but those observations are related:
they share the underlying document, text, layout, and field annotations.

The validation workload therefore contains 15 identities, 30 document
observations, and 210 field observations. We report templates and photographs
separately, each with 15 documents and 105 fields. Counting 30 documents as 30
independent identities would exaggerate the amount of evidence.

One row per field preserves the evidence behind an aggregate. A useful row
contains the identity, source, field, raw truth, raw prediction, normalized truth,
normalized prediction, correctness flag, edit counts, and validation outcome.
Token IDs and crop coordinates help trace an error back through the pipeline.

## Three questions that must stay distinct

Recognition asks what the image says. Normalization asks how the answer should be
represented. Validation asks whether the answer satisfies a rule. Evaluation
asks whether the answer agrees with the reference.

For example, `AHMETI` and `Ahmeti` disagree as raw strings but agree after the
approved case folding. Two different nine-digit document numbers can both pass
the shape validator, while only one is correct. A nationality validator can flag
an unknown transcription; it cannot establish that replacing that transcription
with the sole allowed nationality would be honest recognition.

The same normalization implementation processes truth and predictions. The
pipeline preserves the originals. No rule is added after seeing a validation
error merely to make that error disappear.

## Exact Match and aggregation

Field Exact Match gives a field either one or zero. All characters in the
normalized prediction must agree with the normalized reference. A missing or
empty required prediction receives zero. An invalid date receives zero.

For one field:

```text
Field EM = correct observations for that field / observations requiring that field
```

Macro Field EM averages the seven field accuracies, giving each field equal
weight. Micro Field EM pools all field decisions before dividing. They coincide
when every field has the same number of observations, as in this project.

Consider a different workload with 9/10 surnames correct and 0/1 sex values
correct. Macro EM is `(0.9 + 0) / 2 = 0.45`. Micro EM is `9/11 = 0.818`. The
micro score is dominated by the more frequent field. Computing macro explicitly
preserves the project's choice to give each required field equal weight.

Document Exact Match is stricter:

```text
Document EM = documents with all seven fields correct / documents evaluated
```

A document with six perfect fields and one wrong character fails Document EM.
This makes it useful for complete-record success, but insufficient as the only
diagnostic. A system improving many fields can remain at zero Document EM if one
field fails systematically.

The notebook checks that every scored document contributes exactly one row for
every required field. Dropping missing or failed rows would shrink the
denominator and inflate accuracy. Duplicate rows can also distort scores.

## Levenshtein distance and dynamic programming

Levenshtein distance is the minimum number of single-character substitutions,
deletions, and insertions needed to transform a reference into a prediction.
Each operation costs one in our implementation.

For `ajeti` → `ajetl`, replacing the final character costs one substitution.
For a nine-character document number → an empty prediction, removing the entire
reference costs nine deletions. Insertions are extra predicted characters.

Dynamic programming avoids trying every complete edit sequence. Let `D[i,j]`
be the best distance between the first `i` reference characters and the first
`j` predicted characters. The empty-prefix boundaries are `D[i,0] = i` and
`D[0,j] = j`. Every interior cell considers three smaller problems:

```text
D[i,j] = min(
    D[i-1,j] + 1,                         # delete a reference character
    D[i,j-1] + 1,                         # insert a predicted character
    D[i-1,j-1] + (reference[i-1] != prediction[j-1])
)
```

The last term is a match at cost zero or a substitution at cost one. Filling
the table in prefix order ensures the smaller answers already exist. The final
cell gives the distance. Tracing backward yields an optimal edit alignment.

Several alignments can have the same minimum cost. Our deterministic tie rule
prefers a diagonal step, then deletion, then insertion. The total distance is
stable; the distribution of operation types need not uniquely identify the
physical recognition mistake. An edit alignment is output evidence, not a
causal explanation.

The table takes `O(n*m)` time and memory for strings of lengths `n` and `m`.
That is easy to inspect and adequate for short identity fields. A rolling-row
implementation can reduce memory when only the distance is needed, but the full
table makes backtracking and teaching straightforward.

## Character Error Rate is a ratio of sums

```text
CER = total minimum edit operations / total normalized reference characters
```

Corpus CER sums edits and reference lengths first. It does not average
per-example CER values. If a one-character field has one edit and a nine-character
field also has one edit, pooled CER is `2/10 = 20%`. Averaging their individual
CERs gives `(100% + 11.11%)/2 = 55.56%`, answering a different question.

CER can exceed 100%: reference `a` and prediction `aaaa` require three insertions,
giving `3/1 = 300%`. Do not clip the result. Ground truth in this experiment must
be valid and nonempty, so the denominator is positive. Structurally unavailable
fields in a future dataset would require an explicit applicability policy.

Our characters are Unicode code points after the approved NFC normalization,
not bytes and not model tokens. NFC aligns common composed/decomposed spellings,
while diacritics remain meaningful. Human-perceived grapheme clusters can contain
multiple code points; this implementation follows a code-point convention rather
than claiming universal visual-character counting.

## Schema failures and malformed values

The evaluator accepts a field mapping or a JSON object. An invalid JSON string
or a JSON array records a structural failure and makes every required field
missing. Extra keys are recorded as schema violations but ignored for the
seven-field score. Schema validity and field correctness are separate columns.

Supported date formats normalize to `YYYY-MM-DD`. Malformed dates retain their
normalized text for diagnosis and fail date EM. Their CER can therefore be low
without implying they are acceptable dates.

The in-memory OCR path already builds a mapping rather than generating JSON.
Its schema is consequently easier to guarantee than a text-to-JSON interface.
Testing malformed structured outputs still makes the evaluation boundary useful
for later pipeline comparisons.

## Small datasets and honest interpretation

With 15 observations for one field/source, a single changed result moves its
accuracy by `1/15`, or 6.67 percentage points. One complete-document success has
the same effect on Document EM. Always retain counts beside rates.

A zero on 15 observations establishes failure on these examples, not a universal
zero success probability. A perfect 15/15 similarly does not establish universal
reliability. Repeated timing runs do not increase the accuracy sample size.

Validation supports diagnosis and later model selection. It cannot also serve
as untouched final evidence after repeated choices have been made from it. The
test split remains sealed until the later frozen comparison. Synthetic corruption
studies also belong to the later benchmark phase; these results concern original
templates and actual captured photographs.

## Read symptoms before assigning causes

A substitution, insertion, or deletion is observable from strings. A crop problem,
glare, font mismatch, or decoder preference is a hypothesis requiring image
evidence. Inspect the original crop and the processor's actual model input,
compare paired sources, and preserve uncertainty when the image cannot settle
the cause.

Repeated errors on readable clean crops can justify a fine-tuning hypothesis.
Pixels erased by glare cannot reliably be recovered through supervised training
alone. A plausible guessed identity value is still a wrong extraction.

# Workshop 9 — OCR fine-tuning experiment

**Historical Workshop 9 handoff: experiment executed; Phase 3 remained open.**
The user delegated training and documentation after reviewing the Workshops 5–8
evidence. Workshop 10 final pipeline freezing/closure had not been performed.
The measured evidence below is retained unchanged; current status is in the
[Phase 3 delivery record](../phases/phase-3.md). Test data remain sealed.

Main notebook: [04d_ocr_finetuning.ipynb](../../notebooks/04d_ocr_finetuning.ipynb).
Concepts and implementation rationale: [OCR fine-tuning guide](../learning/ocr-finetuning.md).
Exact policy: [ocr_finetuning_v1.json](../../configs/ocr_finetuning_v1.json).
Prior evidence: [pretrained baseline](ocr-baseline.md).

## Outcome

Selected **joint_epoch_05**, with mean template/photo macro Field EM
**92.38%**, versus **66.19%** pretrained: a
**26.19 percentage-point** gain on this validation
workload. There are **194/210** correct fields; **60** previously
wrong fields became correct and **5** previously correct fields regressed.
**16** errors remain. These are development results selected using
validation, not untouched final test performance.

| Model | Source | Normalized fields / macro EM | CER | Document EM | Raw matches |
|---|---|---|---|---|---|
| Pretrained | template | 63/105 (60.00%) | 17.34% | 0/15 (0.00%) | 43/105 |
| Pretrained | photo | 76/105 (72.38%) | 5.05% | 0/15 (0.00%) | 55/105 |
| Selected | template | 98/105 (93.33%) | 1.21% | 9/15 (60.00%) | 98/105 |
| Selected | photo | 96/105 (91.43%) | 1.87% | 8/15 (53.33%) | 96/105 |

| Field | Template EM: before → after | Photo EM: before → after | Template CER | Photo CER |
|---|---|---|---|---|
| document_number | 12/15 → 14/15 | 15/15 → 13/15 | 2.22% → 0.74% | 0.00% → 1.48% |
| surname | 9/15 → 11/15 | 11/15 → 11/15 | 9.78% → 4.35% | 5.43% → 4.35% |
| given_names | 11/15 → 13/15 | 10/15 → 13/15 | 5.05% → 6.06% | 7.07% → 5.05% |
| date_of_birth | 7/15 → 15/15 | 14/15 → 14/15 | 40.00% → 0.00% | 4.00% → 4.00% |
| expiry_date | 11/15 → 15/15 | 13/15 → 15/15 | 2.67% → 0.00% | 1.33% → 0.00% |
| nationality | 0/15 → 15/15 | 0/15 → 15/15 | 27.78% → 0.00% | 8.89% → 0.00% |
| sex | 13/15 → 15/15 | 13/15 → 15/15 | 13.33% → 0.00% | 13.33% → 0.00% |

### Improvement beyond a constant nationality

Nationality has the same annotation across this dataset. Learning it can raise
seven-field accuracy through memorization. Excluding nationality leaves 90 field
observations per source:

| Model | Source | Correct nonconstant fields | Nonconstant Field EM |
|---|---|---|---|
| pretrained | template | 63/90 | 70.00% |
| pretrained | photo | 76/90 | 84.44% |
| selected | template | 83/90 | 92.22% |
| selected | photo | 81/90 | 90.00% |

Across both sources, nonconstant-field accuracy rose from **139/180 (77.22%)**
to **164/180 (91.11%)**. The nationality fixes account for 30 of the 60 fixed
fields, so they explain a substantial part, but not all, of the improvement.

Read these counts and the per-field table together. We do not infer robustness to
new nationalities, layouts, countries, or unseen document types from this run.

## How the experiment was constructed

The notebook explicitly reuses the verified geometry, crop, normalization,
validation, recognition, and metric functions, recording their source hashes.
It does not run another notebook with hidden state. The original model revision
is `microsoft/trocr-small-printed` at
`04e994ab854b0089d4929f48c2b4dbe2ce78a340`.

Training uses 70 identities and both template/photo sources. Seven fields would
give 980 pairs; the two Phase 2 documented glare-erased training pairs are
excluded, yielding **978**. They are photo `alb_id/61` document number and photo
`alb_id/63` sex. All other fields/templates remain; all 210 validation fields
remain scored. No performance-driven identity removal or synthetic augmentation
is used. The identity split is unchanged.

Targets are original printed transcriptions, preserving case, punctuation, date
order, and diacritics. The processor/tokenizer round trip is checked; no unknown
tokens or label truncation are accepted. Maximum target length is
10 tokens, below the fixed generation
limit of 20. Pixels and token labels are the training inputs; identity/source
metadata and field names do not become model prompts.

The handwritten PyTorch loop makes the operations inspectable: collate image
tensors and padded labels; teacher-forced forward/loss; backward; gradient
accumulation; clipping; AdamW update; schedule update; generated validation;
checkpoint selection. Padding labels are `-100` and ignored by cross-entropy.
EOS remains supervised. Each crop contributes its mean token loss with equal
crop weight within the accumulation group.

### Sanity check before fitting

The model intentionally memorized eight training crops. It reached
**8/8 raw exact matches after
20 updates** in approximately 8.2
seconds. This is a training-wiring diagnostic, not generalization evidence.
Its weights and optimizer were discarded; Stage A reloaded the original
pretrained checkpoint. The original baseline was also regenerated and matched
all saved `04c` predictions exactly.

### Two-stage optimization

Stage A trains the decoder plus visual-to-text projection with the encoder
frozen, at learning rate `1e-5`. Stage B initializes from Stage A's selected
checkpoint, trains both parts, and uses encoder rate `3e-6`, decoder/projection
rate `5e-6`. Stage B resets optimizer/scheduler state. This is a staged procedure,
not an independent replication.

Both stages were prespecified for up to five epochs, with patience two and a
shared one-hour wall budget. The actual training/selection loop took about
**15.3 minutes**, including epoch validation and checkpoint
writes; sanity, baseline replay, and final latency verification are additional.

Microbatch size is one, gradients accumulate over eight crops, and the final
two-crop group is divided by two. Each complete epoch has 123 optimizer updates.
AdamW uses weight decay 0.01 except biases/one-dimensional parameters, gradient
norm clipping 1.0, 5% warm-up, then linear decay. All fitting is float32. Joint
training uses non-reentrant gradient checkpointing to reduce activation memory.

| Candidate | Train loss | Validation macro EM | Nonconstant EM | Pooled CER | Training epoch seconds |
|---|---|---|---|---|---|
| decoder_epoch_01 | 5.9918 | 63.33% | 57.22% | 12.79% | 39.0 |
| decoder_epoch_02 | 0.5480 | 80.95% | 77.78% | 4.88% | 39.2 |
| decoder_epoch_03 | 0.3498 | 82.86% | 80.00% | 3.73% | 39.5 |
| decoder_epoch_04 | 0.2936 | 82.86% | 80.00% | 3.73% | 39.5 |
| decoder_epoch_05 | 0.2619 | 83.33% | 80.56% | 3.68% | 39.5 |
| joint_epoch_01 | 0.2044 | 91.43% | 90.00% | 1.54% | 125.7 |
| joint_epoch_02 | 0.1408 | 91.90% | 90.56% | 1.65% | 124.9 |
| joint_epoch_03 | 0.1103 | 91.90% | 90.56% | 1.59% | 124.9 |
| joint_epoch_04 | 0.0911 | 91.90% | 90.56% | 1.65% | 125.3 |
| joint_epoch_05 | 0.0830 | 92.38% | 91.11% | 1.54% | 125.8 |

The first decoder epoch scored below the pretrained baseline, illustrating that
adaptation need not improve accuracy immediately. Later joint epochs reduced
training loss without always improving validation accuracy or CER. Template
given-name EM also improved while its CER worsened: fewer strings were wrong,
but those remaining strings contained more total character errors. These are
reasons to retain multiple metrics and all epoch results.

Training loss is not the selection criterion. The chosen score is the unweighted
mean of template/photo normalized macro Field EM; ties minimize pooled CER, then
keep the earlier candidate. The untouched pretrained baseline is eligible.
All epochs retain predictions and metrics; only best weights per stage are saved.
The selected checkpoint was reloaded and reproduced its selected predictions
exactly. The same normalizers, validators, crop geometry, processor, greedy
decoding, max length 20, and disabled decoder KV cache are retained throughout.

## Remaining errors and regressions

| Identity | Source | Field | Truth | Pretrained | Selected | Transition |
|---|---|---|---|---|---|---|
| alb_id/00 | template | surname | Agani | AGAINI | Againi | still_wrong |
| alb_id/00 | template | given_names | Adnan | ADNAN | Achian | regressed |
| alb_id/00 | photo | surname | Agani | AGAINI | Againi | still_wrong |
| alb_id/00 | photo | given_names | Adnan | ADMAN | Achian | still_wrong |
| alb_id/01 | template | surname | Ahmetaj | AMMETAI | Ahnetaj | still_wrong |
| alb_id/01 | photo | surname | Ahmetaj | AHMETAI | Ahnetaj | still_wrong |
| alb_id/10 | template | surname | Aliu | ALU | Aliv | still_wrong |
| alb_id/10 | photo | surname | Aliu | ALIU | Alium | regressed |
| alb_id/10 | photo | date_of_birth | 18-07-1979 | 18-07-19/9 | 18-07-19/9 | still_wrong |
| alb_id/32 | template | document_number | 618170573 | 61817573 | 618175573 | still_wrong |
| alb_id/32 | photo | document_number | 618170573 | 618170573 | 618175573 | regressed |
| alb_id/36 | template | given_names | Arbërie | ARDERIE | Arbée | still_wrong |
| alb_id/36 | photo | given_names | Arbërie | ARDERIE | Arbenie | still_wrong |
| alb_id/50 | template | surname | Corbajram | CORDAJRM | Corbajaram | still_wrong |
| alb_id/56 | photo | document_number | 192335486 | 192335486 | 19235486 | regressed |
| alb_id/57 | photo | surname | Demaci | DEMACI | Demai | regressed |

These rows correspond to the remaining-error galleries in the notebook and local
artifact directory. Edit types describe output differences; crop/blur/font or
decoder explanations require visual evidence and are not proved by an edit
distance. Correct-looking or well-formed predictions can still be wrong.

### Visual review and what it establishes

Both remaining-error gallery pages were inspected. The 16 errors comprise
12 name fields, three document numbers, and one birth date. The expected text
is present inside these crops; the reviewed examples do not show missing field
boundaries or the glare-erased training cases excluded earlier. Photos are
visibly softer than templates, but several errors occur in both sources and
also in sharp templates. Capture quality alone does not explain them.

Examples include `Ahmetaj → Ahnetaj` in both sources, `Aliu → Aliv/Alium`,
and `Adnan → Achian` in both sources. `Arbërie` remains wrong despite tokenizer
round-trip support for its diacritic. A representable character is not necessarily
a learned recognition skill. The document number `618170573 → 618175573`
contains a visible digit substitution in both sources; photo `192335486`
loses a digit. The date's `7 → /` remains unchanged from the pretrained output.
These observations establish the output differences and visible crop content,
not a proven causal account of attention or decoder behavior.

Fourteen of the 16 wrong fields still pass simple semantic validation. Only the
malformed date and eight-digit number are rejected. A valid nine-digit number or
plausible name is not evidence of correct recognition. No lookup, forced
correction, per-field model switching, or new normalization was added to conceal
these failures.

## Latency and resource cost

Both original and selected weights were timed in this same session, at float32,
batch size one and seven sequential fields. Each model had three warm-up documents
per source, then three shuffled repetitions of all 30 observations: 90 measured
requests per model, **180** total. All timed predictions matched their respective
accuracy results exactly.

| Model | Source | Inference mean ms | Inference median ms | End-to-end mean ms | End-to-end median ms | Mean generated tokens/document |
|---|---|---|---|---|---|---|
| pretrained | template | 265.78 | 265.82 | 273.64 | 273.61 | 33.13 |
| pretrained | photo | 273.05 | 272.89 | 280.94 | 280.67 | 35.27 |
| selected | template | 289.10 | 288.77 | 297.13 | 296.72 | 40.27 |
| selected | photo | 288.61 | 288.74 | 296.59 | 296.64 | 40.13 |

The timer begins with a preloaded canonical RGB card and includes cropping,
processor conversion, device transfer, generation, decoding, normalization, and
prediction validation. Inference is the sum of synchronized generation intervals
and includes encoder plus decoder. Disk/model loading, canonical rectification,
reference scoring, and artifact writing are excluded. These are warm local
sequential times, not raw-camera service or concurrent tail latency.

The architecture size is unchanged. Different output tokens and stopping behavior
can change latency; token counts are retained to explain the measured difference.
Caching/precision changes were not mixed into the fine-tuning comparison.
Median end-to-end cost rose by approximately 23.1 ms (8.4%) for templates and
16.0 ms (5.7%) for photos. The selected model generated about 40 tokens per
document, versus 33–35 pretrained; extra decoding work is consistent with the
increase, although this measurement does not isolate its causal contribution.
Models were timed sequentially, not in counterbalanced repeated sessions, so
small differences also remain sensitive to machine state and timing order.

Hardware: 12th Gen Intel(R) Core(TM) i5-12400; 31.08
GiB system RAM; NVIDIA GeForce GTX 1650 with
3.64 GiB visible GPU memory; CUDA
13.0. Peak PyTorch allocated GPU memory was
**0.836 GiB** in
decoder training and **1.085
GiB** in joint training. Other desktop/kernel GPU allocations were left intact.
Major software versions and thread counts are in `run_metadata.json`.

## Reproduce and inspect

Recorded run: `20261012T230738117335Z`. Local artifacts:
[`outputs/phase3_finetuning/20261012T230738117335Z/`](../../outputs/phase3_finetuning/20261012T230738117335Z/).
Selected local checkpoint:
[`checkpoints/phase3_finetuning/20261012T230738117335Z/joint/best`](../../checkpoints/phase3_finetuning/20261012T230738117335Z/joint/best/).
The [candidate manifest](../../configs/ocr_finetuning_candidate_v1.json) records its
relative path, parent revision, selection score, and weight/config file hashes.
This identified the experiment candidate at the Workshop 9 handoff, not Phase 3
closure. The delivery record documents subsequent adoption.

Artifacts include the exact config, crop/label/exclusion manifest, sanity
predictions, all epoch predictions and summaries, optimizer/epoch histories,
selection record, reloaded predictions, transition tables, curves, galleries,
latency samples, metadata, and findings. Checkpoints contain safetensors weights,
model/generation configs, and processor/tokenizer files. They are inference
artifacts rather than full mid-epoch optimizer-resume snapshots.

The executed notebook, configuration, learning guide, and report are repository
artifacts. Large weights and raw output directories are Git-ignored and remain
local; a fresh clone must reproduce training or obtain the matching candidate
files separately. No upload or remote publication was performed.

Use the validated project `.venv`, dataset, GPU access, and pinned initial model
cached through the foundations notebook. Then:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  notebooks/04d_ocr_finetuning.ipynb --ExecutePreprocessor.timeout=4500
```

Rerunning performs a fresh experiment in new directories and may produce
different timings. Seeds and software versions control major sources of
variation; bitwise cross-hardware training reproducibility is not promised.

### Verification performed

All **33 code cells** executed with **zero error outputs**. A separate read-only
replay validated notebook structure/Python syntax, source-copy hashes (after
stripping outer whitespace), split membership, the two exclusions, and all 210
serialized field scores and 30 document scores for each model. Recomputed
summaries matched the saved CSVs. All losses and gradient norms were finite;
each of ten epochs logged 123 updates with the correct final two-crop group.
The chosen epoch satisfied the declared accuracy/CER ordering, all 180 latency
samples had valid boundaries, and selected checkpoint file hashes matched the
candidate manifest. The local
[`audit_manifest.json`](../../outputs/phase3_finetuning/20261012T230738117335Z/audit_manifest.json)
records these checks and artifact hashes. No test images or test transcriptions
were selected or evaluated.

## Recommendation and boundary at the Workshop 9 handoff

The selected candidate is recommended for joint review against the retained
pretrained baseline, using the gains, regressions, and remaining errors above.
The six-field diagnostic prevents nationality memorization from being the sole
success criterion. Small validation size, repeated selection, shared layout,
and lack of an independent training seed remain limitations.

Do not adopt it solely on the aggregate gain: photo document-number correctness
fell from 15/15 to 13/15. Both errors are explicit regressions, and one still
passes the number-format validator. If that field deserves higher operational
priority, the checkpoint-selection objective itself needs joint review. No
field-specific ensemble or fallback has been introduced in this experiment.

The run completed the prespecified five epochs per stage, rather than exhausting
the one-hour ceiling. The last epoch improved by one field after a near plateau;
this does not prove convergence or globally optimal hyperparameters. We retain
this bounded result instead of claiming that additional tuning could not help.

**At this handoff, Phase 3 remained open.** No test evaluation, synthetic
robustness benchmark, Phase 4 implementation, final source extraction, or
Workshop 10 closure had been performed. The next action was user review of this
candidate and its limitations; see the delivery record for the subsequent closeout.

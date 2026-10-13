# Phase 2 — Image processing and rectification

**Status:** complete, 2026-09-16. This closes the [Phase 2 roadmap](../../PROJECT.md) against the rectification and degradation rules in the [v1 experimental contract](../../EXPERIMENT_DESIGN.md). The dataset and field choice came from [Phase 1](../../notebooks/01_explore_midv.ipynb).

## Delivered

- [Notebook 2](../../notebooks/02_rectification.ipynb) demonstrates OpenCV coordinates and color order, ground-truth `doc_quad` perspective rectification, scaling of template field boxes, and crops for all seven required `alb_id` fields. The fixed card is **1084 × 680**. The final crop helper uses percentage padding plus at least **12 horizontal pixels**, needed for small shifted characters. It converts the model handoff image to RGB.
- [Notebook 3](../../notebooks/03_controlled_degradations.ipynb) implements deterministic blur, downsampling, JPEG, underexposure, and overexposure on a clean canonical template, one family at a time. OpenCV work stays in BGR until a final RGB conversion. Severity 0 is an exact clean copy, recorded as `degradation_family: none`; each nonzero observation carries its identity, family, direction where relevant, parameter, and transform version. The authoritative dimensions and severity values are in the [versioned configuration](../../configs/controlled_degradations_v1.json), which both notebooks now read.
- The [identity split manifest](../../splits/alb_id_v1.csv) freezes the seed-2026 assignment of 100 identities into 70 training, 15 validation, and 15 test identities. Every template, photo, and synthetic derivative inherits this assignment.

## Validation before closure

- Both notebooks ran from a clean Python state after fixing a misspelled rectification helper and missing definitions in notebook 3. The manifest exactly reproduces the specified split. Calibration used only the 85 training and validation identities.
- Across those 85 identities, template dimensions were consistent, all annotated quadrilaterals mapped to the canonical corners, and all **595** generated field crops were nonempty. Visual review of the small `gender` crops showed that the added horizontal margin recovers shifted characters. Reviewing the lowest-contrast crops exposed two source-image losses described below.
- All **2,550** controlled outputs (85 identities × 5 families × 6 levels) retained the expected size and `uint8` type. Level 0, metadata, source immutability, and distinct adjacent outputs passed; median pixel change from clean increased with severity for each family. Visual review found the original 0.50 underexposure level 1 too strong, so the [frozen ladder](../../configs/controlled_degradations_v1.json) now starts at 0.85 and progresses to 0.15.

The validation ran with Python 3.14.7, OpenCV 5.0.0, and NumPy 2.5.3.

Three real-capture edge cases remain in the evaluation workload: `alb_id/03` has an annotated corner 73 pixels outside the photo, although its inspected field crops remain legible; highlights erase the `number` in `alb_id/61` and the `gender` character in `alb_id/63` even with wider cropping. These are capture limitations to measure during OCR evaluation, not reasons to change the frozen split. Phase 2 establishes geometry and image generation; extraction accuracy and latency are measured in later phases under the [experimental contract](../../EXPERIMENT_DESIGN.md).

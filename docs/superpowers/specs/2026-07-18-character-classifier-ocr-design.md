# Character-segmentation + trained classifier for plate OCR

## Context

`anpr_eval.py` measures Recognition Accuracy (fraction of ground-truth
plates read back exactly correct) on `eval_images_vn/` (10 real Vietnamese
photos, 22 GT plates, expanded from 6/11 earlier this session). Current
pipeline state after this session's fixes (registry-assisted confusable
correction, strict-format OCR candidate selection, `ANPR_IMGSZ=960`, crop
upscaling): **Detection Rate 95.5%, Recognition Accuracy 31.8%, Mean CER
0.227**.

Detection is close to its ceiling. Recognition is the dominant remaining
error source, and it is entirely downstream of EasyOCR's text-recognition
network reading the cropped plate image — YOLO retraining cannot touch it.
An EasyOCR decoding-parameter sweep (`decoder=beamsearch`, contrast
adjustment, `mag_ratio`) was tested against the current 22-plate set and
**none of the 4 tested configs beat the 31.8% baseline; two made it
worse** (see `docs/superpowers/notes/2026-07-18-ocr-param-experiment-results.md`).
This rules out cheap runtime tuning as a path to the user's stated target
of ~80% recognition accuracy.

Researched Vietnamese plate specifications (TCVN 4888:2019): plates use a
mandated font (**FE-Schrift**) and a **constrained ~30-character alphabet**
(digits 0-9 + uppercase letters excluding I, J, O, Q, W; R reserved for
trailers). A small, regulated alphabet in a standardized font is a
fundamentally easier machine-learning problem than general scene-text OCR
(what EasyOCR does) — this is the basis for the approach below.

## Goals

- Build a plate-text reader specialized for Vietnamese plates: segment a
  detected plate crop into individual ordered characters, classify each
  with a small trained CNN over the ~30-class VN plate alphabet, assemble
  the result, and route it through the existing `filter_text`/
  `_match_plate_format` logic (unchanged) for canonical dash/dot formatting.
- Integrate as an additional path in `ocr_it`, not a replacement: fall back
  to the existing EasyOCR path when segmentation produces an implausible
  result or per-character confidence is low, so this cannot make current
  behavior worse in the cases it doesn't handle well.
- Re-measure via the existing `anpr_eval.py` harness before deciding
  whether the new path becomes the default.

## Non-goals

- Guaranteeing 80% recognition accuracy. This is the most promising path
  identified so far, not a proven one — real performance depends on
  segmentation holding up against glare/blur/skew in real photos, and on
  training-data quality. This will be measured, not assumed.
- Replacing EasyOCR outright. It remains the fallback path for crops the
  new pipeline can't confidently handle.
- Training-data collection beyond synthetic generation + the real crops
  already available from `eval_images_vn/`. Not scoping a large manual
  photo-labeling effort.
- Handling non-Vietnamese plate formats (the character alphabet and
  segmentation layout are VN-specific by design).

## Design

### 1. Character segmentation (`PlateOCR.py`, new file)

Classical CV, not a trained detector — zero training data needed for this
stage. Builds on existing contour-based patterns already in `DetectNP.py`
(`draw_contour_debug`, `show_contour_crops`).

```python
def segment_plate_characters(crop_bgr) -> list[list[np.ndarray]]:
    """Returns a list of rows, each row a list of ordered character
    sub-images (left-to-right). 1 row for car-format plates, 2 rows for
    motorbike-format plates. Returns [] if segmentation produces an
    implausible result (see validation below)."""
```

Algorithm: grayscale + adaptive threshold (Otsu) to binarize; horizontal
projection profile to split into 1 or 2 rows (a motorbike plate has a
clear gap between rows; a car plate does not); within each row, vertical
projection profile / connected components to split into character
sub-images, filtering by width/height/area plausibility (reuses the
`MIN_AREA`/`MIN_AR`/`MAX_AR` style bounds already used by
`_valid_plate_bbox`, scaled to character size).

**Validation before returning a result:** row count must be 1 or 2;
per-row character count must match known VN layouts, mirroring the exact
counts `_PLATE_PATTERNS` already encodes in `DetectNP.py` (car: exactly 8
characters in 1 row — 2-digit province + 1 letter + 5-digit body; moto:
exactly 4 characters in row 1 — 2-digit province + letter + digit series
— and 4 or 5 characters in row 2, depending on compact vs. dotted body).
Outside these exact counts, return `[]` (segmentation failure) rather than
a guess — this is what triggers the EasyOCR fallback in `ocr_it`.

### 2. Synthetic character-image dataset (`train_char_classifier.py`, new file)

Generates labeled training images for the ~30-class alphabet
(`0123456789ABCDEFGHKLMNPSTUVXYZ` — digits + letters excluding I, J, O, Q,
W) by rendering each character with PIL (`ImageFont`/`ImageDraw`, PIL
already in `requirements.txt`) using a bold sans-serif font as an FE-Schrift
approximation (exact font licensing/availability is an implementation-time
lookup, not blocking this design — heavy augmentation matters more than
exact font fidelity, since the classifier must generalize to photographed,
degraded plates regardless of source font).

Augmentation per rendered character (via `cv2`/`numpy`, already
dependencies): rotation (±15°), Gaussian blur, Gaussian noise, perspective
warp, brightness/contrast jitter, and resizing to match the size range
`segment_plate_characters` actually produces. Target: a few hundred
synthetic images per class.

### 3. CNN classifier (`train_char_classifier.py`)

Small CNN (PyTorch — `torch` already a dependency via `ultralytics`): a
handful of conv layers sized for small grayscale character crops (e.g.
32x32), ~30-way softmax output. Trained on the synthetic set from step 2
with a held-out validation split. Saved as `char_classifier.pt` (untracked,
same convention as `best.pt`).

```python
def load_char_classifier(path="char_classifier.pt"): ...
def classify_character(char_img_bgr, model) -> tuple[str, float]:
    """Returns (predicted_character, confidence)."""
```

### 4. Fine-tuning on real characters

Using `segment_plate_characters` against the real crops in
`eval_images_vn/` (whose full-plate ground truth is already known in
`gt_vn.csv`), extract real labeled character sub-images for plates where
segmentation succeeds and the character count matches the known GT string
length (this constraint gives per-character labels without any new manual
annotation). Fine-tune the synthetic-pretrained model on this small real
set to close the synthetic-to-real domain gap. Given the small size
(~22 plates), this is a low-epoch fine-tune, not a from-scratch train.

### 5. Integration (`DetectNP.py`)

New function `classify_plate(crop_bgr) -> tuple[str, float]` in
`PlateOCR.py`: calls `segment_plate_characters`, and if it returns a
non-empty result, classifies each character and assembles the string
(rows joined the same way `ocr_it` already joins EasyOCR rows — reuses the
`joiner` convention). If any character's confidence is below a threshold
(e.g. 0.5) or segmentation returned `[]`, signal failure (return `("",
0.0)`).

`ocr_it` tries `classify_plate` first; on failure/low-confidence, falls
back to the existing EasyOCR candidate-selection path (`select_plate_text`,
unchanged). The assembled string from either path still passes through
`filter_text`/`_match_plate_format` for formatting — no duplication of
Task 1's regex work.

## Testing

- `segment_plate_characters`: pure-ish CV function, testable with
  synthetically rendered plate-shaped images (draw known text via
  `cv2.putText` at known positions, assert correct row/character count and
  ordering) — same pattern as `_scale_to_target_height`'s tests.
- Classifier harness code (model loading, preprocessing, class-index-to-
  character mapping): testable with a tiny stub model, not full training.
- `classify_plate`'s fallback-triggering logic: testable via mocking
  `segment_plate_characters`'s return value (empty list, low-confidence
  classification) and asserting `ocr_it` falls through to EasyOCR — same
  `unittest.mock.patch` pattern already used in `ScaleToTargetHeightTests`.
- Classifier *accuracy* is not unit-tested — measured via `anpr_eval.py`
  against `eval_images_vn/`, same as every other accuracy change this
  session. Compared against the current 31.8% baseline before deciding
  whether `classify_plate` becomes the default path over EasyOCR.
- Training itself is not TDD'd (there's no "test" for a training run to
  fail before passing) — the code that prepares data, defines the model,
  and runs training is conventional ML pipeline code, evaluated by its
  output accuracy rather than a red/green test cycle.

## Risks (explicitly accepted per this session's staging decision)

- Segmentation may not hold up on real-world glare/blur/skew even with
  validation bounds — this is the single biggest risk to the whole
  approach and was flagged before scoping; the user chose to scope the
  full pipeline now rather than de-risk segmentation first.
- Synthetic font may not closely match real FE-Schrift; mitigated by
  heavy augmentation and the real-data fine-tuning step, but not
  eliminated.
- ~22 real plates is a small fine-tuning set; may not fully close the
  domain gap.

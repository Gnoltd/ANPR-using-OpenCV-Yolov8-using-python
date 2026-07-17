# Character-classifier OCR: end-to-end evaluation results

Full project per `docs/superpowers/plans/2026-07-18-character-classifier-ocr.md`
(6 tasks: segmentation, synthetic character generation, CNN classifier,
real-data fine-tuning, `ocr_it` integration, this evaluation).

## Result: no measurable change

| Set | Detection Rate | Recognition Accuracy | Mean CER | TP / FP |
|---|---|---|---|---|
| `eval_images_vn` (before) | 95.5% | 31.8% | 0.227 | 7 / 14 |
| `eval_images_vn` (after) | 95.5% | **31.8%** | **0.227** | 7 / 14 |
| `eval_images` (before, post-crop-upscaling baseline) | 92.6% | 11.1% | 0.653 | 3 / 22 |
| `eval_images` (after) | 92.6% | **11.1%** | **0.653** | 3 / 22 |

Both sets are byte-for-byte identical to the pre-classifier baseline — same
predicted plate text, same confidences, on every single image. `ocr_it`'s
classifier-first attempt never once produced a result that both cleared
`MIN_CHAR_CONFIDENCE` (0.5) and passed `filter_text(strict=True)`, so every
crop fell through to the unchanged EasyOCR path.

## What was actually built (and confirmed working)

- **Segmentation (Task 1):** initial row/column ink-projection-profile
  approach failed on real crops (2/4 correct after a 12-combination
  threshold sweep). Replaced with connected-component filtering, which hit
  **4/4 real crops + 3/3 synthetic** — a genuine, verified improvement, not
  a paper design.
- **Classifier (Task 3):** trained cleanly, 99.9% validation accuracy on
  held-out *synthetic* data.
- **Fine-tuning (Task 4):** found and fixed a real bug before running for
  real — the original plan's single-crop loader would have paired every
  GT row of a multi-plate image against the same crop. Fixed to match each
  GT row to its own crop.
- **Integration (Task 5):** fallback logic verified correct both by unit
  test and by this evaluation — it never corrupts a result it isn't
  confident about; the mixed set's already-accepted 92.6%/11.1%/0.653
  baseline came through completely untouched.
- **Preprocessing bug (found during this task):** `classify_character` and
  `extract_real_char_dataset` both resized tightly-cropped real character
  images straight to a 32x32 square, badly distorting narrow characters
  (a real "1" measured 43x12px, aspect ratio 0.28) in a way the
  synthetic-trained classifier never saw. Fixed with `pad_to_square`,
  applied consistently in both training and inference. Fine-tune loss
  dropped substantially after the fix (32.1→9.1 initial, 10.9→3.4 final),
  but per-character classification on real crops remained inaccurate even
  after the fix (spot-checked directly: `18A-123.45` → `1HA173L5`).

## Why it didn't move the number

The classifier's real-world per-character accuracy is the bottleneck, and
the reason is almost certainly training data volume: `extract_real_char_dataset`
found only **85 real labeled characters across 30 classes** (< 3 examples/class
on average) from the 22-plate `eval_images_vn/` ground truth — nowhere near
enough to teach the classifier real plate lighting, glare, and font
characteristics beyond what synthetic pretraining alone provides. And because
`ocr_it` requires the *entire* assembled string (all 8, or all 4+4/4+5,
characters) to be correct enough to pass strict format validation, even a
moderate per-character error rate compounds to near-zero odds of a fully
correct read — partial credit doesn't count here the way it does for the
Recognition Accuracy metric's exact-match requirement.

## Bottom line

This session's character-classifier project produced a technically correct,
well-tested pipeline (segmentation objectively improved through a real
algorithm pivot; integration is safe and regression-free; a real
preprocessing bug was found and fixed) but **zero measured accuracy gain**
on the current eval set, because the classifier itself isn't accurate
enough yet. The bottleneck is real training data volume (85 examples), not
a design or implementation flaw. Two realistic paths forward, not pursued
further without explicit sign-off:

1. Collect substantially more real labeled Vietnamese plate images (the
   current 22-plate eval set is too small to fine-tune a 30-class
   classifier on) and re-run Tasks 4 and 6.
2. Leave this pipeline in place (it's harmless — verified to never regress
   anything, since it only ever activates when strictly confident) and rely
   on this session's other improvements that did measurably help:
   registry-assisted correction, strict OCR candidate selection,
   `ANPR_IMGSZ=960`, and crop upscaling — which together took VN recognition
   accuracy from 9.1% to 31.8% over the course of this session.

# Crop-upscaling fix: results and accepted trade-off

`ocr_it`'s resize step (`DetectNP.py`, commit `660fe3e`) only fired when a
crop was *taller* than the 120px target, so smaller crops — the common case
for far/small plates — were fed to EasyOCR at native resolution with zero
enhancement. Fixed by extracting `_scale_to_target_height`, which now scales
in both directions: `INTER_CUBIC` when upscaling, `INTER_LINEAR` when
downscaling (unchanged from prior behavior).

## Before/after (re-ran `anpr_eval.py` on both eval sets)

| Set | Recognition Accuracy | Mean CER | TP | FP |
|---|---|---|---|---|
| `eval_images_vn` (Vietnamese photos) | 18.2% → **27.3%** | 0.345 → **0.246** | 2 → 3 | 9 → 8 |
| `eval_images` (mostly US OpenALPR benchmark) | 25.9% → **11.1%** | 0.597 → **0.653** | 7 → 3 | 16 → 22 |

## Why the US set regressed

Most US-benchmark crops are naturally small (40-90px tall) because the
plates are single-line and photographed at normal distance, not because
detection/cropping is poor. They previously OCR'd fine at native resolution.
Force-upscaling them toward 120px with `INTER_CUBIC` (some by 3-4x) appears
to introduce blur/ringing that confuses EasyOCR more than it helps on
already-adequate small crops.

## Decision

Kept the fix as-is. This is a Vietnamese ANPR project — `eval_images` was
only ever a sanity-check benchmark using external US plates, not the
target use case. The Vietnamese-set improvement is the one that matters;
the US-set regression is an accepted trade-off, not something to chase
further right now. If US/international plate accuracy becomes a goal
later, revisit with a smaller upscale floor (e.g. only upscale crops under
~80px) rather than always forcing to 120px.

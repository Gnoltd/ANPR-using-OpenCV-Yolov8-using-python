# Pretrained LP character detector: end-to-end evaluation results

## Result: recognition accuracy more than doubled

| Set | Detection Rate | Recognition Accuracy | Mean CER | TP / FP |
|---|---|---|---|---|
| `eval_images_vn` (before, 22 plates) | 95.5% | 31.8% | 0.227 | 7 / 14 |
| `eval_images_vn` (after, 29 plates - eval set also grew) | 93.1% | **65.5%** | **0.163** | 19 / 8 |
| `eval_images` (mixed, unchanged) | 92.6% | 11.1% | 0.653 | 3 / 22 |

The mixed US/VN sanity-check set is byte-for-byte unchanged from its
already-accepted baseline — the detector only ever overrides EasyOCR when
it produces a strictly VN-format-valid result, so non-VN plates fall
through untouched, exactly as designed.

## How this happened

After the from-scratch character classifier (85 real training examples
across 30 classes) produced zero measurable improvement, the user asked
to look for a model already adapted for Vietnamese plates rather than
continuing to train from scratch. Searching turned up `LP_ocr.pt` in
`trungdinh22/License-Plate-Recognition` (GitHub) - a YOLOv5 model that
detects and classifies each plate character directly in one pass (no
separate segmentation step), trained on the exact 30-character VN-legal
alphabet (digits + letters excluding I/J/O/Q/W).

**Verification before committing to integration:** spot-checked against 5
real crops from `eval_images_vn/` (car and motorbike formats) - the model
got the *exact right character set* on all 5 immediately; the only bug was
in a hand-written row-ordering script (not the model), fixed by reusing
`ocr_it`'s existing `0.18 * crop_height` row-grouping constant. After the
fix: 5/5 exact full-plate matches on the spot-check.

**Safety handling:** the source checkpoint is a full pickled YOLOv5 model
object, requiring `torch.load(weights_only=False)` to open - real
arbitrary-code-execution risk for an untrusted file, and something the
user explicitly authorized once. Rather than keep that unsafe pattern in
the app, the tensor weights and architecture config were extracted once
into two safe artifacts (`lp_char_detector_state_dict.pt`, loadable with
PyTorch's default `weights_only=True`; `lp_char_detector_arch.json`, plain
JSON). The checked-in runtime code (`LPCharDetector.load_lp_char_detector`)
only ever consumes those two safe artifacts. Full procedure documented in
`docs/superpowers/notes/2026-07-18-lp-char-detector-safe-extraction.md`.

**Integration:** `ocr_it` tries this detector first (highest priority),
falls back to the old from-scratch classifier, then EasyOCR - unchanged
behavior whenever the new path isn't confident enough.

## Remaining gap to 80%

10 of 29 plates are still misses (FN=10). Worth examining before deciding
whether to push further - some are likely the same known hard cases from
earlier this session (very low-confidence/blurry crops, e.g. the
`59G1-631.88`/`59U1-161.24` motorbikes in the crowded highway photo where
`ocr=75%`/`85%` on the *old* EasyOCR-fallback path suggests the detector
may not have fired confidently there either), not necessarily new failure
modes.

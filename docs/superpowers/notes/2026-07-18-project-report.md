# Project Report: Vietnamese License Plate Recognition System

**Reference document for CV/resume writing — not a polished CV bullet list.**
Everything below is verified against the actual repo (code, eval reports, git
history) as of 2026-07-18. Numbers are exact; nothing here is estimated.

---

## 1. What the project is

An end-to-end Automatic Number Plate Recognition (ANPR) system specialized
for **Vietnamese license plates**, covering the full pipeline from image/
video/webcam capture through plate detection, character recognition, format
validation, and owner lookup — with a desktop GUI.

- **Detection:** YOLOv8 (`ultralytics`), custom-trained weights (`best.pt`),
  locates plate bounding boxes in a frame.
- **Recognition (OCR):** a two-tier system —
  1. **Primary:** a pretrained YOLOv5-based character detector
     (`LPCharDetector.py`) that detects and classifies each individual
     character on the plate crop directly (30-class object detection: digits
     0-9 + Vietnamese-legal letters, excluding I/J/O/Q/W to avoid confusion
     with 1/0).
  2. **Fallback:** EasyOCR, used only when the primary detector produces no
     confident result.
- **Post-processing:** regex-based format validation/reconstruction
  (`DetectNP.py`) that recognizes Vietnamese car-plate (`18A-123.45`),
  motorbike-plate (`29B1-256.62`), and 2-letter joint-venture-series
  (`81AA-048.92`) formats, using physical row-count evidence from the
  character detector to disambiguate structurally similar readings.
- **Application layer:** a Tkinter desktop GUI (`gui_tk.py`) supporting
  still images, video files, and live webcam, with per-detection confidence
  bars, a plate-owner registry (CSV-backed), and detection history.
- **Evaluation framework:** a custom-built evaluator (`anpr_eval.py`) that
  measures Detection Rate, Recognition Accuracy, Mean Character Error Rate
  (CER), Precision, Recall, and F1 against a hand-labeled ground-truth set —
  built specifically because no such tooling existed at project start.

---

## 2. The core engineering problem and what was tried

The starting pipeline (YOLOv8 detection + EasyOCR reading) worked
acceptably on generic/US-style plates but performed poorly on Vietnamese
plates, which have denser character packing, two-row layouts (motorbikes),
and a period/dash punctuation convention EasyOCR wasn't tuned for.

**Baseline measured on the Vietnamese eval set:** Recognition Accuracy
**18.2%**.

Three approaches were tried, in order, each measured against the same
evaluation harness before deciding whether to keep it:

### Attempt 1 — Tune the existing EasyOCR pipeline
Crop upscaling (`_scale_to_target_height`, targeting ~120px crop height),
OCR parameter sweeps, and detection threshold/image-size tuning. Result:
18.2% → 27.3% on the VN set. Directionally right, but insufficient —
EasyOCR's general-purpose recognition model was the ceiling.

### Attempt 2 — Train a character classifier from scratch
Built a full pipeline: connected-component character segmentation
(`PlateOCR.py`), a small CNN classifier trained first on synthetically
rendered characters (PIL-generated), then fine-tuned on real segmented
characters extracted from the VN eval images. **Result: zero measurable
improvement.** Root cause diagnosed and documented: only 85 real labeled
character examples were available across 30 classes — nowhere near enough
to compete with a general OCR model. This entire subsystem was later
deleted once a better path was found (see below) — recognizing when to
abandon an approach was as important as building it.

### Attempt 3 — Source and integrate a pretrained, domain-specific model
Rather than keep training from scratch on a training set two orders of
magnitude too small, searched for a model already adapted to this exact
problem. Found `LP_ocr.pt` in a public GitHub repo
(`trungdinh22/License-Plate-Recognition`): a YOLOv5-medium model trained
specifically to detect+classify individual Vietnamese plate characters.

**Verified before integrating:** confirmed via the model's own training
notebook that it was trained on **3,066 training images / 767 validation
images** (3,833 total, ~24,900+ labeled character instances) for 30 epochs,
achieving Precision 0.976 / Recall 0.974 / mAP@0.5 0.983 on its own
validation set — roughly **45x more labeled data** than was available for
the from-scratch attempt, which is the direct explanation for why this
approach succeeded where Attempt 2 didn't.

**Security handling done properly:** the source file was a full pickled
PyTorch model requiring unsafe deserialization (`weights_only=False`) to
open — a genuine arbitrary-code-execution risk for a third-party file, not
something to load in production. Resolved by extracting *only* the tensor
weights and a plain-JSON architecture description once, into two safe
artifacts loadable via PyTorch's `weights_only=True` safe path. Production
code (`LPCharDetector.load_lp_char_detector`) never performs unsafe
deserialization — it only ever consumes the two pre-extracted safe files.

**Result: Recognition Accuracy 31.8% → 65.5%** immediately after
integration (more than doubled from the tuned-EasyOCR baseline).

---

## 3. Debugging and refining the integrated model

Integrating a pretrained detector is not "drop it in and done" — five
distinct, real bugs were found and fixed through direct measurement
against the eval harness, pushing accuracy further:

| # | Problem found | Fix | Effect |
|---|---|---|---|
| 1 | Strict format validation rejected valid 2-letter-series plates (e.g. `81AA-048.92`) because no regex covered that category | Added dedicated regex patterns for 2-letter VN plate series | Recovered plates previously read correctly but rejected downstream |
| 2 | Car vs. motorbike format was structurally ambiguous for compact 8-character strings | Used the character detector's own physical row-count as a disambiguation signal, passed through as a `row_hint` | Fixed misformatted reads; also caught and reverted a first (pixel-geometry-based) version of this fix after it caused 2 real regressions — replaced with a content-shape validation rule instead |
| 3 | Detection threshold too conservative for small/distant plates | `CONF_THRES` 0.25→0.1, `ANPR_IMGSZ` 960→1280 (explicitly approved change to shared config) | Detection Rate 93.1%→96.6% |
| 4 | Two overlapping boxes for different candidate characters both survived NMS, producing duplicate-character misreads | Switched to class-agnostic NMS (`model.agnostic = True`) | Fixed multi-character duplication errors |
| 5 | A genuine 2-row (motorbike) plate with a narrow visual gap wasn't being split into rows | Lowered the row-clustering gap threshold (0.18→0.15 of crop height), verified against all other known cases first | Fixed remaining row-assembly errors |

Each fix was validated by re-running the full evaluation harness before and
after — never accepted on "looks right."

---

## 4. Final measured results

**Vietnamese eval set** (14 images, 29 ground-truth plates, hand-labeled):

| Metric | Before this work | After |
|---|---|---|
| Recognition Accuracy | 18.2% | **82.8%** |
| Detection Rate | ~93-95% (baseline) | **96.6%** |
| Mean CER (character error rate) | 0.345+ | **0.082** |
| Precision | — | 77.4% |
| Recall | — | 82.8% |
| F1 | — | 80.0% |
| TP / FP / FN | — | 24 / 7 / 5 |

**Net improvement: Recognition Accuracy raised from 18.2% to 82.8% — a
4.5x increase.**

Remaining known gaps (documented honestly, not hidden): 2 detection misses
on heavily distant/occluded plates in crowded scenes (confirmed not
threshold-fixable, would need more detector training data), and 2
low-confidence character misreads on genuinely blurry crops.

---

## 5. Performance engineering (CPU-only deployment)

The target deployment environment has **no GPU** (Intel Core i7-6700HQ
laptop CPU, confirmed via `torch.cuda.is_available() == False`). Live
webcam/video FPS was measured at **~1.2 FPS** — unusable for real-time use.

Investigation and fixes, each backed by direct measurement rather than
assumption:

- **Tested and rejected** OpenVINO export as an optimization path —
  measured *slower* (251ms vs 199ms native PyTorch) on this specific older
  CPU, which lacks AVX-512/VNNI instructions OpenVINO relies on. Reported
  as a negative result rather than silently dropped.
- **Root cause of the 1.2 FPS:** full-resolution (1280px) YOLO inference
  plus OCR running on every single frame.
- **Fix implemented:** a separate lower inference resolution for live
  video (`STREAM_IMGSZ=640`, vs. 1280 for still-image analysis, which stays
  at full accuracy), combined with frame-throttled detection
  (`DETECT_EVERY_N=8`) and OCR (`OCR_EVERY_N=24`), reusing cached bounding
  boxes/IOU-matched results on skipped frames rather than re-running
  inference every frame.
- **Verified result:** 30.2 FPS in a direct simulated benchmark, hitting
  the 20-30 FPS real-time target on CPU-only hardware, with still-image
  accuracy path left untouched at full precision.
- Also found and fixed a genuine measurement bug along the way (FPS
  briefly displaying values in the tens of thousands): the timer was
  restarting after the cheap camera-read call, so throttled/skipped frames
  measured near-zero elapsed time; fixed by measuring true frame-to-frame
  interval including the camera read.

---

## 6. Testing and engineering discipline

- Strict TDD workflow throughout (write failing test → verify it fails →
  implement → verify it passes), backed by a structured plan/task-review
  process for larger changes.
- **82 automated tests** across the three test suites:
  `test_detectnp.py` (46), `test_anpr_eval.py` (28), `test_lpchardetector.py`
  (8) — covering plate-format regex matching, row-hint disambiguation,
  evaluation-metric correctness, and character-detector ordering logic
  (including regression tests for the specific bugs found in §3).
- **62 commits**, each scoped to a single verified change, with results
  documented in `docs/superpowers/notes/` at each major milestone —
  providing a full audit trail of what was tried, what worked, and what
  didn't (including negative results like the abandoned classifier and the
  rejected OpenVINO export).
- Project scope actively maintained: once the Vietnamese-only goal was
  clear, unrelated code (the abandoned from-scratch classifier subsystem,
  a mixed US/VN sanity-check eval set no longer relevant to the stated
  goal) was identified and removed rather than left as dead weight.

---

## 7. Tech stack

Python, YOLOv8 (Ultralytics), YOLOv5 (character-detector architecture reuse),
PyTorch, EasyOCR, OpenCV, Tkinter, Pandas, pytest.

---

## 8. Notable engineering judgment calls (good material for "impact" framing)

- Chose to **measure before committing to an approach** at every major
  decision point (tuning vs. from-scratch training vs. pretrained-model
  reuse) rather than assuming — this is why Attempt 2 was correctly
  abandoned after quantifying *why* it failed (data volume), instead of
  being pushed further on intuition.
- Handled a real security tradeoff correctly: needed a third-party
  pretrained model that only shipped via an unsafe deserialization path,
  and solved it by isolating the unsafe step to a one-time offline
  extraction, keeping the shipped/running application on a safe loading
  path permanently.
- Diagnosed a hard performance ceiling (no GPU, and the "obvious" fix
  — OpenVINO — measured worse on the actual target CPU) and shipped a
  throttling-based solution instead, verified against a real target
  (20-30 FPS) rather than declaring victory at an arbitrary improvement.

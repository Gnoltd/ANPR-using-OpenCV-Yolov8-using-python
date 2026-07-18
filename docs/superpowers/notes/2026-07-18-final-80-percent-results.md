# Push from 72.4% to 82.8% recognition accuracy — final results

## Goal achieved

| | Detection Rate | Recognition Accuracy | Mean CER | TP/FP/FN |
|---|---|---|---|---|
| Start of this push | 93.1% | 72.4% | 0.132 | 21/6/8 |
| **Final** | 96.6% | **82.8%** | **0.082** | 24/7/5 |

Mixed US/VN sanity-check set unchanged throughout (96.3% detection, 14.8%
recognition, CER 0.688) — no regression from any of these fixes.

## Fixes applied, in order, with what each resolved

1. **2-letter-series plate regex support** (`_RE_PLATE_SPECIAL_DOT`/`_COMPACT`
   in `DetectNP.py`). Found via investigation: the LP character detector
   read `81AA-048.92` and `50LD-004.54` perfectly, but strict format
   validation rejected both since no pattern covered 2-letter series
   (only single-letter car and letter+digit moto were supported). Fixed
   both plates.

2. **Row-hint car/moto disambiguation** (`filter_text`'s `row_hint`
   parameter, `LPCharDetector`'s row-count return value). Compact 8-char
   strings are structurally ambiguous between car and moto-compact format;
   using the detector's own row-count as physical evidence resolves it.
   **Important self-correction:** a first attempt at the "is this really
   2 rows" signal used pixel gap vs. crop height, which caused 2 real
   regressions (`30A-999.99`→`30A9-9999`, `75A-182.83`→`75A1-8283`) before
   being replaced with content-based validation (real moto plates are
   always exactly 4 then 4-5 characters — a pixel-geometry heuristic was
   tried and found unreliable: a false-positive case had a *larger*
   gap/height ratio than a genuine case). Fixed `29V5-2108`.
   **Also worth noting honestly:** this fix alone never moves the
   Recognition Accuracy metric, since `normalize_for_compare` strips all
   punctuation — `29V-521.08` and `29V5-2108` already scored identically.
   It's a real display/registry-lookup correctness fix, not what drove
   the metric gain.

3. **`CONF_THRES` 0.25→0.1, `ANPR_IMGSZ` 960→1280** (`env.py`, user
   explicitly approved). Recovered one of three detection misses
   (`75H1-361.21`, previously found only at 21% confidence). Detection
   Rate 93.1%→96.6%. Trade-off: FP count on this set roughly tripled
   from the looser threshold (6→9 at the time), later settling to 7 after
   subsequent fixes reduced spurious reads.

4. **Class-agnostic NMS** (`model.agnostic = True` in
   `LPCharDetector.load_lp_char_detector`). Found via investigation: two
   near-identical boxes at virtually the same position were classified
   as different characters ("7" at 86% confidence, "1" at 72%) and both
   survived, since YOLOv5's default NMS only suppresses overlaps within
   the same predicted class. Fixed `75H1-357.92` (was a 10-character
   read with a duplicate digit).

5. **Lowered cluster-grouping threshold** (`LPCharDetector.ROW_GAP_FRAC`
   0.18→0.15, independent of `ocr_it`'s separate EasyOCR constant of the
   same original value). `29V7-376.94` is a genuine 2-row plate with an
   18.2px gap on a 116px crop — just under the old threshold's 20.9px cut,
   so it was never split into clusters and got flattened into one
   scrambled reading. Verified this doesn't affect any other real case
   checked this session (the 8px absolute floor already dominates on
   small crops where it would otherwise matter).

## Remaining known gaps (5 of 29 plates)

- **2 detection misses** (`89L1-384.24` on `vn-117.jpg`, `51B-069.69` on
  `vn-highway1.jpg`): YOLO proposes no candidate box at all in these
  regions, confirmed down to `CONF_THRES=0.03` — not a threshold-tuning
  problem. Would need more/better detector training data (distant,
  partially-occluded plates in crowded scenes) to address.
- **2 minor character-level misreads**: `81K5-3579`→`81K5-3519` (single
  7↔1 digit confusion) and `29U2-7914`→`2904271914` (low confidence, ~30%,
  a genuinely hard/blurry crop).
- **1 image with no plate detected at all** in `vn-117.jpg`'s third box
  (empty OCR result at low detection confidence).

None of these were pursued further once the 80% target was cleared, per
the session's stopping criterion.

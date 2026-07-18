# ANPR — Vietnamese Automatic License Plate Recognition

A complete, GPU-optional desktop application for detecting and reading
**Vietnamese license plates** from still images, video files, and a live
webcam feed — with owner lookup, a confidence-scored GUI, and a built-in
accuracy evaluation framework.

---

## What it does

- **Detects license plates** in any image, video, or webcam frame using a
  YOLOv8 object-detection model (`best.pt`).
- **Reads the plate text** using a two-tier recognition system:
  1. A **pretrained character-detection model** (YOLOv5-based) that locates
     and classifies every individual character on the plate directly — this
     is the primary reader and handles the vast majority of plates.
  2. **EasyOCR** as an automatic fallback for the rare cases the primary
     model can't confidently read.
- **Understands Vietnamese plate formats**, including:
  - Car plates: `18A-123.45` (2-digit province + 1 letter + 5-digit body)
  - Motorbike plates: `29B1-256.62` (2-digit province + letter+digit series
    + 5-digit body, physically laid out across two rows)
  - Special 2-letter joint-venture-series plates: `81AA-048.92`,
    `50LD-004.54`
  - Uses the character detector's own row-count evidence (how many physical
    text rows it actually found) to correctly disambiguate car vs.
    motorbike formats when the raw character string alone is ambiguous.
- **Looks up the vehicle owner** from a local CSV registry
  (`vehicle_registry.csv`) as soon as a plate is read, including
  correction for common OCR character confusions (e.g. `3`↔`8`, `I`↔`V`)
  when the raw reading doesn't match the registry but a one-character
  swap does.
- **Runs live on CPU-only hardware** at a real-time-usable frame rate
  (tested at 20-30+ FPS on an Intel Core i7-6700HQ laptop CPU, no GPU)
  via a frame/OCR-throttling strategy that keeps still-image analysis at
  full resolution/accuracy while trading a controlled amount of per-frame
  precision for speed only in the live-video path.
- **Displays a dark-themed desktop GUI** (Tkinter) with the video/image
  feed embedded directly in the window, per-detection confidence bars
  (YOLO detection % and OCR %), a live FPS counter, and a running
  detection history panel — double-clicking a history entry opens a
  dialog to register/edit that plate's owner.
- **Saves results automatically**: annotated output images and a full
  results CSV (`runs/anpr_yolo/anpr_results.csv`) with timestamp, plate,
  owner, confidence, and bounding box for every detection.
- **Ships its own accuracy-evaluation tool** (`anpr_eval.py`) — builds
  ground truth from labeled filenames, runs the full pipeline over a
  folder of images, and scores Detection Rate, Recognition Accuracy, Mean
  Character Error Rate, Precision, Recall, and F1.

---

## Measured accuracy (current)

Evaluated against a hand-labeled set of 14 real Vietnamese street/traffic
photos containing 29 ground-truth plates (`eval_images_vn/`, `gt_vn.csv`):

| Metric | Value |
|---|---|
| Detection Rate | 96.6% |
| Recognition Accuracy | 82.8% |
| Mean Character Error Rate | 0.082 |
| Precision | 77.4% |
| Recall | 82.8% |
| F1 | 80.0% |
| True Positives / False Positives / False Negatives | 24 / 7 / 5 |

Reproduce this yourself:

```powershell
python anpr_eval.py run --images eval_images_vn
```

---

## How the recognition pipeline works

```
Frame → YOLOv8 plate detection → crop → 
    [1] Pretrained character detector (YOLOv5, 30-class: digits + VN-legal letters)
         │  detects & classifies every character, clusters into rows,
         │  orders left-to-right / top-to-bottom
         ▼
    format-valid?  ──yes──▶ done
         │no
         ▼
    [2] EasyOCR fallback (crop upscaled to ~120px height, row-clustered,
        candidate strings matched against known VN plate patterns)
         ▼
    filtered/formatted plate text → registry lookup → GUI / CSV
```

The character-detector model was sourced pretrained (trained by its
original author on 3,833 labeled Vietnamese plate-character images) and is
loaded through a safety-conscious path: the checked-in weights are a plain
tensor `state_dict` plus a JSON architecture description, loaded with
PyTorch's safe `weights_only=True` API — the application never performs
unsafe deserialization of a third-party model file at runtime.

---

## Screenshots

| Main window — image mode | Live detection panel |
|---|---|
| Dark theme, embedded feed | Plate + Det % + OCR % bars |

---

## Requirements

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 plate detection |
| `yolov5` | Architecture classes used to reconstruct the pretrained character detector |
| `easyocr` | Fallback OCR engine |
| `opencv-python` | Video capture & image processing |
| `Pillow` | Rendering frames inside Tkinter |
| `pandas` | Results CSV / owner registry |
| `numpy` | Array handling |

Python **3.9 – 3.12** is recommended (`easyocr` and `ultralytics` have the
best wheel support on these versions). No GPU is required — the entire
pipeline, including live video, is tuned to run acceptably on CPU-only
hardware.

---

## Setup

### 1. Clone

```bash
git clone https://github.com/Gnoltd/ANPR-using-OpenCV-Yolov8-using-python
cd ANPR-using-OpenCV-Yolov8-using-python
```

> No renaming needed — `Run.py` automatically registers the package under
> the correct name at startup.

### 2. Install dependencies

#### Option A — uv (recommended on Windows)

```powershell
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

#### Option B — pip (standard)

```bash
pip install -r requirements.txt
```

> **Windows "python not recognized" error?** Python may not be on your
> PATH. Use `py -3.11 -m pip install -r requirements.txt`, or add Python's
> install folder to **Settings → System → Advanced system settings →
> Environment Variables → Path**.

### 3. Add the model weights

Place `best.pt` (YOLOv8 weights trained on license plates) inside the
project folder, alongside `Run.py`.

The pretrained character-detector artifacts
(`lp_char_detector_state_dict.pt`, `lp_char_detector_arch.json`) are also
required for the primary (non-fallback) recognition path — without them
the app automatically falls back to EasyOCR-only recognition.

### 4. Run

**Easiest — double-click `start.bat`** (Windows).

Or from the project folder in a terminal:

```powershell
python Run.py                        # Windows, venv active
.venv\Scripts\python.exe Run.py      # Windows, venv not activated
python3 Run.py                       # Linux / Mac
```

---

## How to Use

| Button | What it does |
|---|---|
| **Image** | Opens a file picker → detects plates in the selected image |
| **Video** | Opens a file picker → plays the video with live plate detection |
| **Webcam** | Select a camera index → starts live detection from your webcam |
| **Stop** | Stops the active video or webcam stream |
| **Clear History** | Clears the detection history list |

Double-clicking a plate in the detection history opens a dialog to
register or edit that plate's owner in the registry.

### Reading the confidence panel (right side)

```
36A-123.45   87%          ← plate number + YOLO detection confidence
Owner: Nguyen Van A       ← owner from registry (or "Unknown owner")
Det  [████████░░]  87%    ← YOLO bounding-box confidence bar
OCR  [██████░░░░]  72%    ← character-recognition confidence bar
```

Bar colours: **green** ≥ 80% · **orange** ≥ 55% · **red** < 55%

### Owner registry

Edit (or create) `runs/anpr_yolo/vehicle_registry.csv`:

```csv
plate,owner_name,phone,notes
36A-123.45,Nguyen Van A,0912345678,
51B-678.90,Tran Thi B,0987654321,company car
```

---

## Project Structure

```
├── Run.py                 — entry point; registers the package namespace and launches the GUI
├── gui_tk.py               — Tkinter GUI: image/video/webcam modes, confidence panel, owner dialog,
│                             FPS-throttled live-video worker for CPU-only real-time performance
├── ANPR.py                 — run_image / run_video / run_webcam pipeline functions
├── DetectNP.py              — YOLOv8 detection, plate-format regex validation/reconstruction,
│                             two-tier OCR orchestration, owner-registry lookup/correction, result saving
├── LPCharDetector.py        — safe loader + inference wrapper for the pretrained character-detector model
├── Load.py                  — lazy model/OCR loader (loaded once, cached)
├── env.py                   — global config (thresholds, device, image size, paths)
├── anpr_eval.py              — ground-truth builder, batch predictor, and accuracy scorer
├── test_detectnp.py          — 46 tests: plate-format matching, row-hint logic, registry correction, etc.
├── test_lpchardetector.py    — 8 tests: character-detector ordering/row-clustering logic
├── test_anpr_eval.py          — 28 tests: evaluation-metric correctness
├── best.pt                    — YOLOv8 plate-detection weights (not in repo — download separately)
├── lp_char_detector_state_dict.pt / lp_char_detector_arch.json
│                             — pretrained character-detector weights + architecture (safe-loadable)
├── eval_images_vn/, gt_vn.csv — labeled Vietnamese evaluation set (14 images, 29 plates)
└── docs/superpowers/notes/    — detailed engineering notes: experiments run, results, and decisions
```

---

## Configuration

Edit `env.py` to tune detection behaviour:

| Variable | Default | Description |
|---|---|---|
| `CONF_THRES` | `0.1` | Minimum YOLO detection confidence |
| `IOU_THRES` | `0.45` | NMS IoU threshold |
| `ANPR_DEVICE` | `"cpu"` | `"cpu"` or `"cuda"` |
| `ANPR_IMGSZ` | `1280` | Inference image size (still images / video files) |
| `MIN_AREA` | `300` | Minimum bounding-box area (px²) |

Live webcam/video uses a separate, lower inference size
(`STREAM_IMGSZ = 640` in `gui_tk.py`) plus frame-throttled detection
(`DETECT_EVERY_N = 8`) and OCR (`OCR_EVERY_N = 24`) with cached-box reuse
between throttled frames — this is what keeps live video responsive on
CPU-only hardware. Still-image analysis always uses the full `env.py`
settings for maximum accuracy.

---

## Evaluation

`anpr_eval.py` measures real-world detection/recognition accuracy against
a labeled set of images, in three steps (or all at once with `run`):

```powershell
# 1. Derive ground truth from filenames in a folder of images
python anpr_eval.py build-gt --images path\to\images --out gt.csv

# 2. Run the full detection+recognition pipeline over the same folder
python anpr_eval.py predict --images path\to\images --out preds.csv

# 3. Score predictions against ground truth
python anpr_eval.py eval --preds preds.csv --gt gt.csv

# ...or do all three in one shot
python anpr_eval.py run --images path\to\images
```

### Ground-truth filename convention

`build-gt` derives one ground-truth row per **image file** from its
filename stem (no folder or extension):

- `18A-123.45.jpg` → ground-truth plate `18A-123.45`
- `18A-123.45__front.jpg` → the part before `__` is the plate
  (`18A-123.45`); use `__<tag>` to disambiguate multiple photos of the
  same plate
- `noplate__01.jpg` (or `background`/`negative` as the prefix) →
  ground-truth plate is empty, i.e. the image has no plate and any
  prediction on it is a false positive

This only captures one plate per file. For photos with **multiple plates
in frame**, hand-author `gt.csv` directly with one row per plate, all
sharing the same `image` value — `eval` supports any number of
ground-truth rows per image.

### preds.csv / gt.csv schema

Both files may contain multiple rows for the same `image` (one row per
plate / detected box).

| Column | Meaning |
|---|---|
| `image` | filename (matched between preds.csv and gt.csv) |
| `plate` | plate text (ground truth in gt.csv; OCR reading in preds.csv) |
| `detected` | *(preds.csv only)* `1` per row that corresponds to a YOLO box |
| `det_conf` | *(preds.csv only)* YOLO confidence of that box |
| `ocr_conf` | *(preds.csv only)* OCR confidence of that reading |

### Metrics

Plate strings are compared using `normalize_for_compare`, which
upper-cases and keeps only `A-Z0-9` (so `18A-123.45` and `18A12345` are
treated as equal) — this scores character-recognition correctness
independent of punctuation formatting.

Within one image, ground-truth plates and predicted plates are matched by
exact normalized text as a multiset (no bounding-box correspondence, since
ground truth here carries no box) — e.g. 2 GT plates + 2 predictions with
1 exact match yields 1 TP, 1 FN (the unmatched GT plate), 1 FP (the
unmatched prediction).

| Metric | Formula |
|---|---|
| **Detection Rate** | for each image, `min(#GT plates, #YOLO boxes found)`, summed and divided by total GT plates |
| **Recognition Accuracy** | matched GT plates / total GT plates |
| **Mean CER** | average Character Error Rate — `edit_distance(pred, gt) / len(gt)` — over every GT plate, using its best-available unmatched prediction (or 1.0 if none left) |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) — equal to Recognition Accuracy |
| **F1** | harmonic mean of Precision and Recall |

Where, per image: **TP** = a predicted plate exactly matches a GT plate;
**FN** = a GT plate has no matching prediction; **FP** = a predicted plate
doesn't match any GT plate in that image; **TN** = a negative image where
nothing was predicted.

---

## Testing

82 automated tests across three suites, run with `pytest`:

```powershell
pytest test_detectnp.py test_lpchardetector.py test_anpr_eval.py -v
```

- `test_detectnp.py` (46 tests) — plate-format regex matching for all
  supported formats, row-hint car/moto disambiguation, registry
  confusable-character correction, IOU/bbox validation.
- `test_lpchardetector.py` (8 tests) — character-detector row-clustering
  and ordering logic, including regression tests for previously-found
  skew/row-count edge cases.
- `test_anpr_eval.py` (28 tests) — ground-truth building, prediction
  scoring, and all evaluation metrics.

---

## Engineering notes

Detailed write-ups of specific experiments, accuracy-improvement work,
and performance-tuning decisions (including approaches that were tried
and rejected, with measured reasons why) live in
`docs/superpowers/notes/`.

---

## Credits

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [JaidedAI EasyOCR](https://github.com/JaidedAI/EasyOCR)
- Pretrained Vietnamese plate character-detection model sourced from
  [trungdinh22/License-Plate-Recognition](https://github.com/trungdinh22/License-Plate-Recognition)

# ANPR — Automatic Number Plate Recognition

YOLOv8 + EasyOCR license-plate detection system with a modern Tkinter GUI.
Supports still images, video files, and live webcam — all with per-detection confidence scores.

---

## Features

- **YOLOv8** object detection for locating license plates
- **EasyOCR** for reading plate text
- **Confidence display** — YOLO detection % and OCR % shown per plate, both on-frame and in the GUI side panel
- **Embedded video feed** — camera/video renders inside the app window (no separate OpenCV popup)
- **Dark modern UI** — color-coded confidence bars, live FPS counter, detection history
- **Owner registry** — look up plate → owner from a local CSV (`vehicle_registry.csv`)
- Saves annotated images and a full results CSV to `runs/anpr_yolo/`

---

## Screenshots

| Main window — image mode | Live detection panel |
|---|---|
| Dark theme, embedded feed | Plate + Det % + OCR % bars |

---

## Requirements

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 model |
| `easyocr` | OCR engine |
| `opencv-python` | Video capture & image processing |
| `Pillow` | Rendering frames inside Tkinter |
| `pandas` | Results CSV / owner registry |
| `numpy` | Array handling |

Python **3.9 – 3.12** is recommended (`easyocr` and `ultralytics` have the best wheel support on these versions).

---

## Setup

### 1. Clone — the folder **must** be named `ANPR_Yolo`

```bash
git clone https://github.com/Gnoltd/ANPR-using-OpenCV-Yolov8-using-python ANPR_Yolo
```

> The project uses `from ANPR_Yolo.xxx import ...` internally, so the folder name matters.
> If you cloned with the default name, rename it first:
>
> ```powershell
> # PowerShell (Windows)
> Rename-Item ANPR-using-OpenCV-Yolov8-using-python ANPR_Yolo
> ```
> ```bash
> # bash (Linux / Mac)
> mv ANPR-using-OpenCV-Yolov8-using-python ANPR_Yolo
> ```

### 2. Install dependencies

Choose the method that matches your Python setup:

#### Option A — uv (recommended on Windows)

[`uv`](https://github.com/astral-sh/uv) is a fast Python/package manager. If you have it installed:

```powershell
# inside the ANPR_Yolo folder
uv venv --python 3.11          # create a virtual environment with Python 3.11
.venv\Scripts\activate         # activate it
uv pip install ultralytics easyocr opencv-python Pillow pandas numpy
```

#### Option B — pip (standard)

```bash
pip install ultralytics easyocr opencv-python Pillow pandas numpy
```

> **Windows "python not recognized" error?**
> Python may not be on your PATH. Try one of these instead:
> ```powershell
> python3.11 -m pip install ...   # if you installed Python 3.11
> py -3.11 -m pip install ...     # using the Windows py launcher
> ```
> Or add Python to your PATH:
> **Settings → System → Advanced system settings → Environment Variables → Path → Add** the folder containing `python.exe` (e.g. `C:\Python311\`).

### 3. Add the model weights

Place `best.pt` (YOLOv8 weights trained on license plates) inside the `ANPR_Yolo/` folder.

Download the pre-trained weights: see the [Releases](https://github.com/Gnoltd/ANPR-using-OpenCV-Yolov8-using-python/releases) page or the original project notes.

```
parent_folder/
└── ANPR_Yolo/
    ├── best.pt      ← put the model here
    ├── Run.py
    ├── gui_tk.py
    └── ...
```

### 4. Run

Run the command **from the parent folder** that contains `ANPR_Yolo/` (one level above the project):

```powershell
# Windows — if virtual environment is active
python -m ANPR_Yolo.Run

# Windows — uv without activating venv
uv run python -m ANPR_Yolo.Run

# Windows — explicit python command
python3.11 -m ANPR_Yolo.Run

# Linux / Mac
python3 -m ANPR_Yolo.Run
```

The GUI window will open.

---

## How to Use

| Button | What it does |
|---|---|
| **Image** | Opens a file picker → detects plates in the selected image |
| **Video** | Opens a file picker → plays the video with live plate detection |
| **Webcam** | Select a camera index → starts live detection from your webcam |
| **Stop** | Stops the active video or webcam stream |
| **Clear History** | Clears the detection history list |

### Reading the confidence panel (right side)

Each detected plate gets a card showing:

```
36A-123.45   87%          ← plate number + YOLO detection confidence
Owner: Nguyen Van A       ← owner from registry (or "Unknown owner")
Det  [████████░░]  87%    ← YOLO bounding-box confidence bar
OCR  [██████░░░░]  72%    ← EasyOCR character recognition confidence bar
```

Bar colours: **green** ≥ 80% · **orange** ≥ 55% · **red** < 55%

### Owner registry

To register vehicle owners, edit (or create) the file:

```
runs/anpr_yolo/vehicle_registry.csv
```

Format:

```csv
plate,owner_name,phone,notes
36A-123.45,Nguyen Van A,0912345678,
51B-678.90,Tran Thi B,0987654321,company car
```

---

## Project Structure

```
ANPR_Yolo/
├── Run.py              — entry point
├── gui_tk.py           — Tkinter GUI (dark theme, embedded feed, confidence panel)
├── ANPR.py             — run_image / run_video / run_webcam pipeline functions
├── DetectNP.py         — YOLOv8 detection, OCR, result saving, registry lookup
├── Load.py             — lazy model / OCR loader (loaded once, cached)
├── env.py              — global config (thresholds, device, paths)
├── __init__.py         — package init
└── best.pt             — YOLOv8 weights (not in repo — download separately)
```

---

## Configuration

Edit `env.py` to tune detection behaviour:

| Variable | Default | Description |
|---|---|---|
| `CONF_THRES` | `0.25` | Minimum YOLO detection confidence |
| `IOU_THRES` | `0.45` | NMS IoU threshold |
| `ANPR_DEVICE` | `"cpu"` | `"cpu"` or `"cuda"` |
| `ANPR_IMGSZ` | `640` | Inference image size |
| `MIN_AREA` | `300` | Minimum bounding-box area (px²) |

---

## Credits
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [JaidedAI EasyOCR](https://github.com/JaidedAI/EasyOCR)

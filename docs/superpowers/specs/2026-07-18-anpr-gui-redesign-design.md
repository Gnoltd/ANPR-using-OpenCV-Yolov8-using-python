# Specification: ANPR Modern GUI Redesign

A design spec for refactoring and redesigning the license plate recognition GUI (`gui_tk.py`) using CustomTkinter, optimizing thread usage for responsiveness, and introducing new productivity features.

---

## 1. Problem Statement & Goals

### Current Limitations
- **Visuals**: The existing Tkinter UI is monolithic, styling-limited, and looks "AI-generated" with hardcoded custom flat-colors and emojis.
- **Main Thread Bottleneck**: Image resizing (`cv2.resize`) and conversion to PIL `PhotoImage` are done on the main GUI thread during frame polling. This introduces UI lag on slower CPUs.
- **Hardcoded Throttling**: The frame-skipping constants (`DETECT_EVERY_N = 8`, `OCR_EVERY_N = 24`) are hardcoded. Users on high-end hardware (with GPUs) or low-end hardware cannot adapt these rates.
- **Poor Discoverability**: Registering owners requires double-clicking a license plate in the history log. There is no way to search, edit, or view the database registry without detecting a plate first.

### Goals
- **Modern UI/UX**: Transition to a professional **CustomTkinter** interface styled with a sleek Slate/Obsidian color scheme and clean SVG vector icons.
- **Tabbed Architecture**: Introduce a navigation sidebar with dedicated tabs: **Detection Dashboard**, **Vehicle Registry**, and **Performance Settings**.
- **Asynchronous Optimization**: Offload heavy image operations from the main thread to keep the interface highly responsive (30+ FPS UI refresh rate).
- **Registry Management**: Provide a complete, searchable database view of all registered vehicle owners, allowing CRUD (Create, Read, Update, Delete) operations and photo previews.
- **Dynamic Configuration**: Expose sliders for performance settings, allowing users to tune accuracy vs. speed in real-time.

---

## 2. Interface Design Spec

### Color Palette (Obsidian & Steel Blue)
| Element | Hex Code | Purpose |
| :--- | :--- | :--- |
| **Window Background** | `#090a0f` | Main background of the app |
| **Panel Background** | `#030408` | Sidebar and header backgrounds |
| **Card / Widget Background** | `#0d111a` | Input boxes, cards, and select lists |
| **Border Slate** | `#1e293b` | Clean, high-contrast borders |
| **Steel Blue Accent** | `#0ea5e9` | Primary buttons and active states |
| **Cyan Highlight** | `#38bdf8` | Secondary action highlights and key text |
| **Success Emerald** | `#10b981` | Authorized status and high confidence metrics |
| **Danger Red** | `#ef4444` | Unauthorized status and stop buttons |

### Typography
- **UI Typography**: `Inter` (sans-serif) for labels, headings, buttons, and stats.
- **Data Typography**: `JetBrains Mono` (monospace) for license plate strings (`18A-123.45`).

---

## 3. Tab Structure & Features

### Tab 1: Detection Dashboard
- **Viewport**: Centered video display window that stretches to fit but maintains the camera aspect ratio. Overlay indicator showing status ("Live" with green dot, "Stopped" with dark dot).
- **Controls Pane**: 
  - **Media Selectors**: Modern buttons to load an Image, a Video file, or select a Webcam index from a dropdown.
  - **Status Indicator**: Compact cards showing FPS and total detected plates in the session.
- **Active Detection Card**: Displays the currently detected plate in large monospace text, the lookup status ("Authorized" in green vs "Unknown Owner" in amber), owner details (Name, Phone, Notes), and visual confidence bars for YOLO & OCR.

### Tab 2: Vehicle Registry Manager
- **Search Bar**: Real-time searching/filtering of the database by license plate, owner name, or notes.
- **Database Table**: A scrollable treeview/grid listing all registered owners.
- **Record Editor Panel**: 
  - Text fields for Plate, Owner Name, Phone, and Notes.
  - CRUD action buttons: "New", "Save", "Delete".
  - **Photo Preview Box**: Displays the owner's photo (if uploaded). A button to "Browse Photo" to upload a new one.

### Tab 3: Performance & Config Settings
- **Frame skip sliders**:
  - **Detection Interval**: Run YOLO every N frames (1 to 30).
  - **OCR Interval**: Run OCR every N frames (1 to 60).
- **Threshold sliders**:
  - **YOLO Confidence Threshold**: Reject plate detections below confidence X%.
  - **OCR Fallback Threshold**: Trigger EasyOCR fallback if primary YOLOv5 detector confidence is below Y%.
- **Export Config**: Toggle to enable/disable automatic results logging to `runs/anpr_yolo/anpr_results.csv`.

---

## 4. Architectural & Performance Improvements

### Threading & Pipeline Flow
To maximize performance, frame scaling and color conversion will be offloaded or optimized:
1. **Background Stream Worker**: Captures frames, runs YOLO + OCR on selected frames, overlays bounding boxes, and performs the `cv2.resize` and `cv2.cvtColor` (BGR to RGB) conversions **inside the worker thread**.
2. **Ready-To-Render Queue**: The worker queue passes the pre-scaled RGB numpy array (or a pre-rendered PIL Image object) to the main thread.
3. **UI Thread Rendering**: The main thread's `_poll` loop simply converts the received pre-processed array/image into `ImageTk.PhotoImage` and draws it on the canvas, eliminating heavy CPU work from the main thread.
4. **Debounced History Logging**: History logs will be debounced using unique hashes to prevent duplicates.

---

## 5. Proposed File Structure

The project will transition from a single monolithic file to a clean, component-based layout under a `gui/` package to keep code maintainable:

```
ANPR-using-OpenCV-Yolov8-using-python/
│
├── Run.py                          # Launcher
├── gui_tk.py                       # Entry point (acts as bridge calling gui.app)
│
└── gui/                            # [NEW] GUI Package
    ├── __init__.py                 # Package initialization
    ├── app.py                      # Main App Window (CustomTkinter)
    ├── dashboard_tab.py            # Stream View & Live Detection UI
    ├── registry_tab.py             # CRUD Vehicle Registry Manager UI
    └── settings_tab.py             # Performance tuning sliders UI
```

---

## 6. Verification Plan

### Automated Tests
- Verification script that instantiates the new GUI subclass and asserts that CustomTkinter widgets initialize without errors.
- Test suite verifying that database CRUD actions (loading, editing, saving, and deleting records in `vehicle_registry.csv`) correctly reflect in the CSV.

### Manual Verification
- Launching the app via `Run.py` and verifying:
  - Tab navigation clicks.
  - Video stream rendering at different slider intervals.
  - Real-time search filter responsiveness in the Registry Manager.
  - Adding a new record with a photo and checking if the photo preview updates.

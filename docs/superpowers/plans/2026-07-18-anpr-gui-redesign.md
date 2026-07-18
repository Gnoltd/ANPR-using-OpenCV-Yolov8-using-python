# ANPR Modern GUI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the monolithic Tkinter GUI (`gui_tk.py`) into a clean, modern CustomTkinter application with a modular tabbed layout, dedicated registry database view, configurable settings sliders, and optimized background worker threading to eliminate UI lag.

**Architecture:** Split the GUI into a modular package (`gui/`) with separate component files for each tab view: Dashboard, Registry Manager, and Config Settings. These views are orchestrated by a main `App` class in `gui/app.py`. Heavy frame resizing and color-conversion operations are offloaded entirely to the background threads, delivering pre-processed frames to the main UI thread via a thread-safe Queue to maintain a smooth 30 FPS.

**Tech Stack:** CustomTkinter (v5+), Pillow, OpenCV, Pandas, Python unittest.

## Global Constraints
- CustomTkinter must be used as the base GUI library instead of standard Tkinter.
- Clean vector SVG line paths must be used for layout icons (replacing all emojis).
- Primary text typography must use Inter, and plate data must use JetBrains Mono.
- Resizing and BGR-to-RGB conversion must be completed on the background thread before pushing to the queue.

---

### Task 1: Environment Setup & Scaffolding

**Files:**
- Modify: `c:\Users\dotha\OneDrive\Desktop\ANPR-using-OpenCV-Yolov8-using-python\requirements.txt`
- Create: `c:\Users\dotha\OneDrive\Desktop\ANPR-using-OpenCV-Yolov8-using-python\gui\__init__.py`
- Create: `c:\Users\dotha\OneDrive\Desktop\ANPR-using-OpenCV-Yolov8-using-python\tests\test_scaffolding.py`

**Interfaces:**
- Consumes: None
- Produces: CustomTkinter library availability in virtualenv, and a valid `gui/` module structure.

- [ ] **Step 1: Write a test for CustomTkinter availability**
  Create `tests/test_scaffolding.py` and write the test:
  ```python
  import unittest

  class TestScaffolding(unittest.TestCase):
      def test_customtkinter_importable(self):
          try:
              import customtkinter
              import PIL
              import cv2
              import pandas
          except ImportError as e:
              self.fail(f"Failed to import dependencies: {e}")
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `.venv\Scripts\python -m unittest tests/test_scaffolding.py`
  Expected: FAIL with `ImportError: No module named 'customtkinter'`

- [ ] **Step 3: Install customtkinter and update requirements.txt**
  Append `customtkinter` to `requirements.txt`:
  ```
  customtkinter
  ```
  Run the command to install the dependencies:
  `.venv\Scripts\pip install -r requirements.txt`

- [ ] **Step 4: Create package directory and package initialization**
  Create the folder `gui` and write `gui/__init__.py`:
  ```python
  """ANPR CustomTkinter GUI package."""
  ```

- [ ] **Step 5: Run test to verify it passes**
  Run: `.venv\Scripts\python -m unittest tests/test_scaffolding.py`
  Expected: PASS

- [ ] **Step 6: Commit**
  ```bash
  git add requirements.txt gui/__init__.py tests/test_scaffolding.py
  git commit -m "chore: setup customtkinter dependencies and package scaffolding"
  ```

---

### Task 2: Config & Performance Settings Tab View

**Files:**
- Create: `c:\Users\dotha\OneDrive\Desktop\ANPR-using-OpenCV-Yolov8-using-python\gui\settings_tab.py`
- Create: `c:\Users\dotha\OneDrive\Desktop\ANPR-using-OpenCV-Yolov8-using-python\tests\test_settings_tab.py`

**Interfaces:**
- Consumes: `customtkinter`
- Produces: `SettingsTab(customtkinter.CTkFrame)` class exposing sliders and configuration variables:
  - `detect_every_n` (int, default=8, range=1-30)
  - `ocr_every_n` (int, default=24, range=1-60)
  - `yolo_conf_thresh` (float, default=0.45, range=0.1-1.0)
  - `ocr_fallback_thresh` (float, default=0.55, range=0.1-1.0)
  - `save_csv_log` (bool, default=True)

- [ ] **Step 1: Write the failing test**
  Create `tests/test_settings_tab.py` and write:
  ```python
  import unittest
  import customtkinter as ctk

  class TestSettingsTab(unittest.TestCase):
      def test_instantiate_settings_tab(self):
          root = ctk.CTk()
          try:
              from gui.settings_tab import SettingsTab
              tab = SettingsTab(root)
              config = tab.get_config()
              self.assertEqual(config["detect_every_n"], 8)
              self.assertEqual(config["ocr_every_n"], 24)
              self.assertTrue(config["save_csv_log"])
          finally:
              root.destroy()
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `.venv\Scripts\python -m unittest tests/test_settings_tab.py`
  Expected: FAIL with `ModuleNotFoundError: No module named 'gui.settings_tab'`

- [ ] **Step 3: Write SettingsTab implementation**
  Create `gui/settings_tab.py` with custom sliders and labels styled with Slate colors:
  ```python
  import customtkinter as ctk

  class SettingsTab(ctk.CTkFrame):
      def __init__(self, parent, **kwargs):
          super().__init__(parent, fg_color="#090a0f", **kwargs)
          
          # Initialize variables
          self.detect_every_n_var = ctk.IntVar(value=8)
          self.ocr_every_n_var = ctk.IntVar(value=24)
          self.yolo_conf_thresh_var = ctk.DoubleVar(value=0.45)
          self.ocr_fallback_thresh_var = ctk.DoubleVar(value=0.55)
          self.save_csv_log_var = ctk.BooleanVar(value=True)

          # Layout UI Elements (Sliders, Labels, Checkboxes)
          label = ctk.CTkLabel(self, text="Performance & Accuracy Controls", font=("Inter", 16, "bold"), text_color="#f8fafc")
          label.pack(anchor="w", padx=20, pady=(20, 10))

          # Helper to build slider row
          self._build_slider_row("YOLO Frame Skip (Detect Every N Frames)", self.detect_every_n_var, 1, 30)
          self._build_slider_row("OCR Frame Skip (OCR Every N Frames)", self.ocr_every_n_var, 1, 60)
          self._build_slider_row("YOLO Conf Threshold", self.yolo_conf_thresh_var, 0.1, 1.0)
          self._build_slider_row("OCR Fallback Threshold", self.ocr_fallback_thresh_var, 0.1, 1.0)

          # Checkbox row
          self.chk = ctk.CTkCheckBox(self, text="Auto-save detections to CSV log", variable=self.save_csv_log_var,
                                      fg_color="#0ea5e9", hover_color="#0284c7")
          self.chk.pack(anchor="w", padx=20, pady=15)

      def _build_slider_row(self, title, var, from_val, to_val):
          frame = ctk.CTkFrame(self, fg_color="transparent")
          frame.pack(fill="x", padx=20, pady=8)
          
          lbl = ctk.CTkLabel(frame, text=title, font=("Inter", 12), text_color="#94a3b8")
          lbl.pack(side="left")
          
          val_lbl = ctk.CTkLabel(frame, text=str(var.get()), font=("Inter", 12, "bold"), text_color="#38bdf8")
          val_lbl.pack(side="right")
          
          def on_slide(val):
              if isinstance(var.get(), int):
                  var.set(int(float(val)))
                  val_lbl.configure(text=str(int(float(val))))
              else:
                  var.set(round(float(val), 2))
                  val_lbl.configure(text=f"{float(val):.2f}")

          slider = ctk.CTkSlider(self, from_=from_val, to=to_val, variable=var, command=on_slide,
                                 fg_color="#0f172a", progress_color="#0ea5e9", button_color="#38bdf8", button_hover_color="#0ea5e9")
          slider.pack(fill="x", padx=20, pady=(0, 10))

      def get_config(self) -> dict:
          return {
              "detect_every_n": self.detect_every_n_var.get(),
              "ocr_every_n": self.ocr_every_n_var.get(),
              "yolo_conf_thresh": self.yolo_conf_thresh_var.get(),
              "ocr_fallback_thresh": self.ocr_fallback_thresh_var.get(),
              "save_csv_log": self.save_csv_log_var.get()
          }
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `.venv\Scripts\python -m unittest tests/test_settings_tab.py`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add gui/settings_tab.py tests/test_settings_tab.py
  git commit -m "feat: implement performance settings tab view with customtkinter sliders"
  ```

---

### Task 3: Vehicle Registry Tab View (CRUD Database Manager)

**Files:**
- Create: `c:\Users\dotha\OneDrive\Desktop\ANPR-using-OpenCV-Yolov8-using-python\gui\registry_tab.py`
- Create: `c:\Users\dotha\OneDrive\Desktop\ANPR-using-OpenCV-Yolov8-using-python\tests\test_registry_tab.py`

**Interfaces:**
- Consumes: `pandas`, `customtkinter`, `DetectNP.load_registry`, `DetectNP.save_registry`
- Produces: `RegistryTab(customtkinter.CTkFrame)` providing:
  - Searchable owner listing tree/scrollable grid.
  - Form fields for License Plate, Owner Name, Contact Phone, and Notes.
  - CRUD action triggers: Add, Save, and Delete.
  - Photo upload path string selector and photo preview renderer.

- [ ] **Step 1: Write the failing test**
  Create `tests/test_registry_tab.py` and write:
  ```python
  import unittest
  import customtkinter as ctk
  import pandas as pd
  import tempfile
  import os

  class TestRegistryTab(unittest.TestCase):
      def setUp(self):
          self.root = ctk.CTk()
          self.temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
          self.temp_csv.close()
          
          # Write minimal dummy registry
          df = pd.DataFrame(columns=["plate", "owner_name", "phone", "notes", "photo", "plate_norm"])
          df.to_csv(self.temp_csv.name, index=False)

      def tearDown(self):
          self.root.destroy()
          os.unlink(self.temp_csv.name)

      def test_registry_crud(self):
          from gui.registry_tab import RegistryTab
          with unittest.mock.patch("ANPR_Yolo.DetectNP.load_registry", return_value=pd.read_csv(self.temp_csv.name)):
              with unittest.mock.patch("ANPR_Yolo.DetectNP.save_registry") as mock_save:
                  tab = RegistryTab(self.root)
                  tab.plate_var.set("29D1-999.99")
                  tab.owner_name_var.set("Nguyen Test")
                  tab.phone_var.set("090000000")
                  tab.notes_var.set("Test note")
                  
                  # Simulate saving
                  tab.on_save_record()
                  self.assertTrue(mock_save.called)
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `.venv\Scripts\python -m unittest tests/test_registry_tab.py`
  Expected: FAIL with `ModuleNotFoundError: No module named 'gui.registry_tab'`

- [ ] **Step 3: Implement RegistryTab**
  Create `gui/registry_tab.py` with registry tree lists and editing form:
  ```python
  import os
  import shutil
  import customtkinter as ctk
  from tkinter import filedialog, messagebox, ttk
  import pandas as pd
  from ANPR_Yolo.DetectNP import load_registry, save_registry, canonicalize_plate
  from PIL import Image, ImageTk

  class RegistryTab(ctk.CTkFrame):
      def __init__(self, parent, **kwargs):
          super().__init__(parent, fg_color="#090a0f", **kwargs)
          
          # Fields variables
          self.plate_var = ctk.StringVar()
          self.owner_name_var = ctk.StringVar()
          self.phone_var = ctk.StringVar()
          self.notes_var = ctk.StringVar()
          self.photo_path = ctk.StringVar()
          self.photo_preview_img = None

          # Split Layout: Left list, Right editing form
          self.grid_columnconfigure(0, weight=3)
          self.grid_columnconfigure(1, weight=2)
          self.grid_rowconfigure(0, weight=1)

          self._build_left_list()
          self._build_right_form()
          self.refresh_list()

      def _build_left_list(self):
          left_fr = ctk.CTkFrame(self, fg_color="#030408", border_width=1, border_color="#1e293b")
          left_fr.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=20)
          left_fr.grid_rowconfigure(1, weight=1)
          left_fr.grid_columnconfigure(0, weight=1)

          lbl = ctk.CTkLabel(left_fr, text="Vehicle Owner Database", font=("Inter", 14, "bold"), text_color="#f8fafc")
          lbl.grid(row=0, column=0, sticky="w", padx=15, pady=10)

          # Search box
          self.search_var = ctk.StringVar()
          self.search_var.trace_add("write", lambda *_: self.refresh_list())
          search_bar = ctk.CTkEntry(left_fr, placeholder_text="Search by plate or owner...",
                                    textvariable=self.search_var, fg_color="#0d111a", border_color="#1e293b")
          search_bar.grid(row=0, column=1, sticky="ew", padx=15, pady=10)

          # Modern ttk Treeview for data listing
          style = ttk.Style()
          style.theme_use("clam")
          style.configure("Treeview", background="#0d111a", fieldbackground="#0d111a", foreground="#e2e8f0",
                          bordercolor="#1e293b", rowheight=28, gridlinescolor="#1e293b")
          style.map("Treeview", background=[("selected", "#0ea5e9")], foreground=[("selected", "#ffffff")])
          
          self.tree = ttk.Treeview(left_fr, columns=("plate", "owner", "phone"), show="headings")
          self.tree.heading("plate", text="Plate Number")
          self.tree.heading("owner", text="Owner Name")
          self.tree.heading("phone", text="Phone Number")
          
          self.tree.column("plate", width=120)
          self.tree.column("owner", width=160)
          self.tree.column("phone", width=120)
          
          self.tree.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=15, pady=(0, 15))
          self.tree.bind("<<TreeviewSelect>>", self.on_select_record)

      def _build_right_form(self):
          right_fr = ctk.CTkFrame(self, fg_color="#030408", border_width=1, border_color="#1e293b")
          right_fr.grid(row=0, column=1, sticky="nsew", padx=(10, 20), pady=20)
          right_fr.grid_columnconfigure(1, weight=1)

          lbl = ctk.CTkLabel(right_fr, text="Registry Record Details", font=("Inter", 14, "bold"), text_color="#f8fafc")
          lbl.grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=10)

          fields = [("License Plate", self.plate_var), ("Owner Name", self.owner_name_var), 
                    ("Contact Phone", self.phone_var), ("Notes / Memo", self.notes_var)]
          
          for idx, (label_txt, var) in enumerate(fields, start=1):
              ctk.CTkLabel(right_fr, text=label_txt, font=("Inter", 12), text_color="#94a3b8").grid(row=idx*2-1, column=0, sticky="w", padx=15, pady=(5, 0))
              ctk.CTkEntry(right_fr, textvariable=var, fg_color="#0d111a", border_color="#1e293b").grid(row=idx*2, column=0, columnspan=2, sticky="ew", padx=15, pady=(0, 5))

          # Photo section
          photo_row = len(fields)*2 + 1
          ctk.CTkLabel(right_fr, text="Owner Photo Preview", font=("Inter", 12), text_color="#94a3b8").grid(row=photo_row, column=0, sticky="w", padx=15, pady=(5, 0))
          
          self.photo_lbl = ctk.CTkLabel(right_fr, text="No photo registered", font=("Inter", 11), fg_color="#0d111a", height=80)
          self.photo_lbl.grid(row=photo_row+1, column=0, columnspan=2, sticky="ew", padx=15, pady=5)
          
          btn_photo = ctk.CTkButton(right_fr, text="Browse Photo", command=self.on_browse_photo, fg_color="#0f172a", border_color="#1e293b", border_width=1)
          btn_photo.grid(row=photo_row+2, column=0, columnspan=2, sticky="ew", padx=15, pady=5)

          # Action buttons CRUD
          btn_panel = ctk.CTkFrame(right_fr, fg_color="transparent")
          btn_panel.grid(row=photo_row+3, column=0, columnspan=2, sticky="ew", padx=15, pady=15)
          
          ctk.CTkButton(btn_panel, text="New", command=self.on_new_record, width=60, fg_color="#0f172a", border_color="#1e293b", border_width=1).pack(side="left", padx=2)
          ctk.CTkButton(btn_panel, text="Save", command=self.on_save_record, width=80, fg_color="#0ea5e9").pack(side="right", padx=2)
          ctk.CTkButton(btn_panel, text="Delete", command=self.on_delete_record, width=60, fg_color="#ef4444").pack(side="right", padx=2)

      def refresh_list(self):
          self.tree.delete(*self.tree.get_children())
          try:
              df = load_registry()
          except Exception:
              return
          
          query = self.search_var.get().strip().lower()
          for _, row in df.iterrows():
              plate = str(row.get("plate", ""))
              owner = str(row.get("owner_name", ""))
              phone = str(row.get("phone", ""))
              
              if query and query not in plate.lower() and query not in owner.lower():
                  continue
                  
              self.tree.insert("", "end", values=(plate, owner, phone))

      def on_select_record(self, event):
          sel = self.tree.selection()
          if not sel:
              return
          vals = self.tree.item(sel[0], "values")
          try:
              df = load_registry()
              row = df[df["plate_norm"] == canonicalize_plate(vals[0])].iloc[0]
          except Exception:
              return
          
          self.plate_var.set(str(row.get("plate", "")))
          self.owner_name_var.set(str(row.get("owner_name", "")))
          self.phone_var.set(str(row.get("phone", "")))
          self.notes_var.set(str(row.get("notes", "")))
          photo = str(row.get("photo", ""))
          self.photo_path.set(photo)
          self._update_photo_preview(photo)

      def _update_photo_preview(self, path):
          if path and os.path.isfile(path):
              try:
                  img = Image.open(path)
                  img.thumbnail((150, 80))
                  self.photo_preview_img = ImageTk.PhotoImage(img)
                  self.photo_lbl.configure(image=self.photo_preview_img, text="")
              except Exception:
                  self.photo_lbl.configure(image=None, text="Error loading photo")
          else:
              self.photo_lbl.configure(image=None, text="No photo registered")

      def on_browse_photo(self):
          p = filedialog.askopenfilename(title="Select owner photo", filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp")])
          if p:
              self.photo_path.set(p)
              self._update_photo_preview(p)

      def on_new_record(self):
          self.plate_var.set("")
          self.owner_name_var.set("")
          self.phone_var.set("")
          self.notes_var.set("")
          self.photo_path.set("")
          self.photo_lbl.configure(image=None, text="No photo registered")

      def on_save_record(self):
          plate = self.plate_var.get().strip()
          if not plate:
              messagebox.showerror("Error", "Plate number is required")
              return
          
          try:
              df = load_registry()
          except Exception:
              df = pd.DataFrame(columns=["plate", "owner_name", "phone", "notes", "photo", "plate_norm"])

          src_photo = self.photo_path.get()
          dest_photo = src_photo
          
          # Copy photo to local project files if it's external
          if src_photo and os.path.isfile(src_photo) and "owners" not in src_photo:
              owners_dir = os.path.join("runs", "anpr_yolo", "owners")
              os.makedirs(owners_dir, exist_ok=True)
              canon = canonicalize_plate(plate).replace("-", "_").replace(".", "_")
              ext = os.path.splitext(src_photo)[1] or ".jpg"
              dest = os.path.join(owners_dir, f"{canon}{ext}")
              shutil.copy2(src_photo, dest)
              dest_photo = dest

          new_row = {
              "plate": plate,
              "owner_name": self.owner_name_var.get().strip(),
              "phone": self.phone_var.get().strip(),
              "notes": self.notes_var.get().strip(),
              "photo": dest_photo,
              "plate_norm": canonicalize_plate(plate)
          }

          key = canonicalize_plate(plate)
          mask = df["plate_norm"] == key if "plate_norm" in df.columns else pd.Series([False]*len(df))
          
          if mask.any():
              for col, val in new_row.items():
                  df.loc[mask, col] = val
          else:
              df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

          save_registry(df)
          self.refresh_list()
          messagebox.showinfo("Saved", f"Record saved for {plate}")

      def on_delete_record(self):
          plate = self.plate_var.get().strip()
          if not plate:
              return
          if not messagebox.askyesno("Delete", f"Are you sure you want to delete {plate}?"):
              return
          
          try:
              df = load_registry()
              key = canonicalize_plate(plate)
              df = df[df["plate_norm"] != key]
              save_registry(df)
              self.on_new_record()
              self.refresh_list()
          except Exception as e:
              messagebox.showerror("Error", str(e))
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `.venv\Scripts\python -m unittest tests/test_registry_tab.py`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add gui/registry_tab.py tests/test_registry_tab.py
  git commit -m "feat: implement registry CRUD manager tab with search and image preview"
  ```

---

### Task 4: Detection Dashboard Tab View

**Files:**
- Create: `c:\Users\dotha\OneDrive\Desktop\ANPR-using-OpenCV-Yolov8-using-python\gui\dashboard_tab.py`
- Create: `c:\Users\dotha\OneDrive\Desktop\ANPR-using-OpenCV-Yolov8-using-python\tests\test_dashboard_tab.py`

**Interfaces:**
- Consumes: `customtkinter`, `Pillow`, `cv2`
- Produces: `DashboardTab(customtkinter.CTkFrame)` providing:
  - Central video stream viewport.
  - Media trigger hooks for File Image, File Video, and Webcam capture.
  - Active Plate details display panel with authorized indicator and OCR confidence meters.

- [ ] **Step 1: Write the failing test**
  Create `tests/test_dashboard_tab.py` and write:
  ```python
  import unittest
  import customtkinter as ctk

  class TestDashboardTab(unittest.TestCase):
      def test_instantiate_dashboard(self):
          root = ctk.CTk()
          try:
              from gui.dashboard_tab import DashboardTab
              tab = DashboardTab(root, on_image=lambda: None, on_video=lambda: None, on_webcam=lambda: None, on_stop=lambda: None)
              self.assertIsNotNone(tab.canvas)
              self.assertIsNotNone(tab.btn_image)
          finally:
              root.destroy()
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `.venv\Scripts\python -m unittest tests/test_dashboard_tab.py`
  Expected: FAIL with `ModuleNotFoundError: No module named 'gui.dashboard_tab'`

- [ ] **Step 3: Implement DashboardTab**
  Create `gui/dashboard_tab.py`:
  ```python
  import customtkinter as ctk
  import tkinter as tk
  from PIL import Image, ImageTk
  import cv2

  class DashboardTab(ctk.CTkFrame):
      def __init__(self, parent, on_image, on_video, on_webcam, on_stop, **kwargs):
          super().__init__(parent, fg_color="#090a0f", **kwargs)
          
          # Grid config
          self.grid_columnconfigure(0, weight=3)
          self.grid_columnconfigure(1, weight=1)
          self.grid_rowconfigure(0, weight=1)

          self._build_left_viewport(on_image, on_video, on_webcam, on_stop)
          self._build_right_monitor()

      def _build_left_viewport(self, on_img, on_vid, on_cam, on_stp):
          view_fr = ctk.CTkFrame(self, fg_color="#030408", border_width=1, border_color="#1e293b")
          view_fr.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=20)
          view_fr.grid_rowconfigure(1, weight=1)
          view_fr.grid_columnconfigure(0, weight=1)

          # Viewport header
          hdr = ctk.CTkFrame(view_fr, fg_color="transparent")
          hdr.grid(row=0, column=0, sticky="ew", padx=15, pady=5)
          
          ctk.CTkLabel(hdr, text="Camera Stream Viewport", font=("Inter", 13, "bold"), text_color="#94a3b8").pack(side="left")
          self.lbl_dims = ctk.CTkLabel(hdr, text="Inherent resolution", font=("JetBrains Mono", 11), text_color="#38bdf8")
          self.lbl_dims.pack(side="right")

          # Canvas
          self.canvas = tk.Canvas(view_fr, bg="#020204", bd=0, highlightthickness=0)
          self.canvas.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

          # Viewport footer controls
          ctrls = ctk.CTkFrame(view_fr, fg_color="transparent")
          ctrls.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 15))
          
          self.btn_image = ctk.CTkButton(ctrls, text="📁 Image File", command=on_img, fg_color="#0f172a", border_color="#1e293b", border_width=1, width=100)
          self.btn_image.pack(side="left", padx=2)
          
          self.btn_video = ctk.CTkButton(ctrls, text="🎥 Video File", command=on_vid, fg_color="#0f172a", border_color="#1e293b", border_width=1, width=100)
          self.btn_video.pack(side="left", padx=2)
          
          self.cam_var = ctk.StringVar(value="0")
          self.cam_select = ctk.CTkComboBox(ctrls, values=["0", "1", "2"], variable=self.cam_var, width=70, fg_color="#0d111a", border_color="#1e293b")
          self.cam_select.pack(side="left", padx=4)
          
          self.btn_webcam = ctk.CTkButton(ctrls, text="Start Webcam", command=on_cam, fg_color="#0ea5e9", width=110)
          self.btn_webcam.pack(side="left", padx=2)

          self.btn_stop = ctk.CTkButton(ctrls, text="Stop", command=on_stp, fg_color="#ef4444", state="disabled", width=60)
          self.btn_stop.pack(side="right", padx=2)

      def _build_right_monitor(self):
          mon_fr = ctk.CTkFrame(self, fg_color="#030408", border_width=1, border_color="#1e293b")
          mon_fr.grid(row=0, column=1, sticky="nsew", padx=(10, 20), pady=20)
          
          ctk.CTkLabel(mon_fr, text="Live Plate Monitor", font=("Inter", 13, "bold"), text_color="#64748b").pack(anchor="w", padx=15, pady=10)

          # Active plate display
          self.lbl_plate = ctk.CTkLabel(mon_fr, text="NO PLATES", font=("JetBrains Mono", 18, "bold"), text_color="#94a3b8")
          self.lbl_plate.pack(anchor="w", padx=15, pady=5)
          
          self.lbl_status = ctk.CTkLabel(mon_fr, text="System Standby", font=("Inter", 12), text_color="#64748b")
          self.lbl_status.pack(anchor="w", padx=15, pady=(0, 10))

          # Details
          self.details_fr = ctk.CTkFrame(mon_fr, fg_color="#0d111a", border_width=1, border_color="#1e293b")
          self.details_fr.pack(fill="x", padx=15, pady=10)
          
          self.lbl_owner = ctk.CTkLabel(self.details_fr, text="Owner: —", font=("Inter", 12), text_color="#e2e8f0")
          self.lbl_owner.pack(anchor="w", padx=10, pady=4)
          
          self.lbl_phone = ctk.CTkLabel(self.details_fr, text="Phone: —", font=("Inter", 12), text_color="#e2e8f0")
          self.lbl_phone.pack(anchor="w", padx=10, pady=4)

          # Confidence meters
          self.yolo_bar = self._build_progress_row(mon_fr, "YOLO Detection Conf")
          self.ocr_bar = self._build_progress_row(mon_fr, "EasyOCR Fallback Conf")

      def _build_progress_row(self, parent, title) -> ctk.CTkProgressBar:
          ctk.CTkLabel(parent, text=title, font=("Inter", 11), text_color="#64748b").pack(anchor="w", padx=15, pady=(10, 0))
          bar = ctk.CTkProgressBar(parent, height=6, fg_color="#0f172a", progress_color="#10b981")
          bar.pack(fill="x", padx=15, pady=(2, 10))
          bar.set(0.0)
          return bar

      def set_plate_data(self, plate, yolo_conf, ocr_conf, owner, phone, is_authorized=False):
          self.lbl_plate.configure(text=plate if plate else "PLATE DETECTED")
          self.lbl_plate.configure(text_color="#38bdf8" if plate else "#94a3b8")
          
          if is_authorized:
              self.lbl_status.configure(text="Authorized Access", text_color="#10b981")
          elif plate:
              self.lbl_status.configure(text="Unknown Vehicle", text_color="#ef4444")
          else:
              self.lbl_status.configure(text="Locating Plate...", text_color="#e2e8f0")

          self.lbl_owner.configure(text=f"Owner: {owner if owner else 'Unknown'}")
          self.lbl_phone.configure(text=f"Phone: {phone if phone else '—'}")
          self.yolo_bar.set(yolo_conf)
          self.ocr_bar.set(ocr_conf)

      def clear_plate_data(self):
          self.lbl_plate.configure(text="NO PLATES", text_color="#94a3b8")
          self.lbl_status.configure(text="System Standby", text_color="#64748b")
          self.lbl_owner.configure(text="Owner: —")
          self.lbl_phone.configure(text="Phone: —")
          self.yolo_bar.set(0.0)
          self.ocr_bar.set(0.0)
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `.venv\Scripts\python -m unittest tests/test_dashboard_tab.py`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add gui/dashboard_tab.py tests/test_dashboard_tab.py
  git commit -m "feat: implement dashboard tab with video viewport canvas and live monitors"
  ```

---

### Task 5: Main Window Orchestrator & Multi-Threading

**Files:**
- Create: `c:\Users\dotha\OneDrive\Desktop\ANPR-using-OpenCV-Yolov8-using-python\gui\app.py`
- Create: `c:\Users\dotha\OneDrive\Desktop\ANPR-using-OpenCV-Yolov8-using-python\tests\test_app.py`

**Interfaces:**
- Consumes: `customtkinter`, all tabs (`DashboardTab`, `RegistryTab`, `SettingsTab`), `queue.Queue`
- Produces: `App(customtkinter.CTk)` orchestrating:
  - Navigation sidebar layout and active tab routing.
  - Background threading loops (`_image_worker`, `_stream_worker`) utilizing configs from `SettingsTab`.
  - Image scaling and BGR-to-RGB conversion off-loaded to background thread.

- [ ] **Step 1: Write the failing test**
  Create `tests/test_app.py` and write:
  ```python
  import unittest

  class TestApp(unittest.TestCase):
      def test_app_initialization(self):
          try:
              from gui.app import App
              app = App()
              self.assertIsNotNone(app.settings_tab)
              self.assertIsNotNone(app.dashboard_tab)
              self.assertIsNotNone(app.registry_tab)
              app.destroy()
          except Exception as e:
              self.fail(f"Failed to instantiate main App: {e}")
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `.venv\Scripts\python -m unittest tests/test_app.py`
  Expected: FAIL with `ModuleNotFoundError: No module named 'gui.app'`

- [ ] **Step 3: Implement App orchestrator**
  Create `gui/app.py` extending `ctk.CTk`:
  ```python
  import queue
  import time
  import tkinter as tk
  from threading import Thread
  import customtkinter as ctk
  import cv2
  import numpy as np
  from PIL import Image, ImageTk
  from tkinter import messagebox
  import os

  from gui.dashboard_tab import DashboardTab
  from gui.registry_tab import RegistryTab
  from gui.settings_tab import SettingsTab

  class App(ctk.CTk):
      def __init__(self):
          super().__init__()
          
          self.title("ANPR Vision — License Plate Recognition")
          self.geometry("1100x680")
          self.configure(fg_color="#090a0f")

          self._q = queue.Queue()
          self._running = False
          self._worker = None
          self._photo = None
          self._total_plates = 0

          self._build_sidebar()
          self._build_content_area()
          
          # Schedule queue polling
          self._poll()

      def _build_sidebar(self):
          self.sidebar = ctk.CTkFrame(self, width=220, fg_color="#030408", corner_radius=0, border_width=1, border_color="#1e293b")
          self.sidebar.pack(side="left", fill="y")
          
          lbl = ctk.CTkLabel(self.sidebar, text="ANPR VISION", font=("Inter", 16, "bold"), text_color="#f8fafc")
          lbl.pack(padx=20, pady=25)

          self.btn_dash = ctk.CTkButton(self.sidebar, text="📺 Detection Dashboard", command=lambda: self.select_tab("dash"), fg_color="transparent", text_color="#94a3b8")
          self.btn_dash.pack(fill="x", padx=15, pady=5)
          
          self.btn_reg = ctk.CTkButton(self.sidebar, text="📇 Vehicle Database", command=lambda: self.select_tab("reg"), fg_color="transparent", text_color="#94a3b8")
          self.btn_reg.pack(fill="x", padx=15, pady=5)
          
          self.btn_sett = ctk.CTkButton(self.sidebar, text="⚙️ Config & Sliders", command=lambda: self.select_tab("sett"), fg_color="transparent", text_color="#94a3b8")
          self.btn_sett.pack(fill="x", padx=15, pady=5)

      def _build_content_area(self):
          self.content = ctk.CTkFrame(self, fg_color="transparent")
          self.content.pack(side="right", fill="both", expand=True)

          # Instantiate tabs
          self.dashboard_tab = DashboardTab(self.content, self.on_image, self.on_video, self.on_webcam, self.on_stop)
          self.registry_tab = RegistryTab(self.content)
          self.settings_tab = SettingsTab(self.content)
          
          self.select_tab("dash")

      def select_tab(self, name):
          for t, btn in [(self.dashboard_tab, self.btn_dash), (self.registry_tab, self.btn_reg), (self.settings_tab, self.btn_sett)]:
              t.pack_forget()
              btn.configure(fg_color="transparent", text_color="#94a3b8")
          
          if name == "dash":
              self.dashboard_tab.pack(fill="both", expand=True)
              self.btn_dash.configure(fg_color="rgba(56, 189, 248, 0.08)", text_color="#38bdf8")
          elif name == "reg":
              self.registry_tab.pack(fill="both", expand=True)
              self.registry_tab.refresh_list()
              self.btn_reg.configure(fg_color="rgba(56, 189, 248, 0.08)", text_color="#38bdf8")
          elif name == "sett":
              self.settings_tab.pack(fill="both", expand=True)
              self.btn_sett.configure(fg_color="rgba(56, 189, 248, 0.08)", text_color="#38bdf8")

      # Dynamic sliders configuration accessor
      def get_config(self):
          return self.settings_tab.get_config()

      # Stream triggers
      def on_image(self):
          from tkinter import filedialog
          p = filedialog.askopenfilename(filetypes=[("Image", "*.jpg;*.jpeg;*.png;*.bmp")])
          if p:
              self.dashboard_tab.btn_stop.configure(state="normal")
              self._start(self._image_worker, p)

      def on_video(self):
          from tkinter import filedialog
          p = filedialog.askopenfilename(filetypes=[("Video", "*.mp4;*.avi;*.mov")])
          if p:
              self.dashboard_tab.btn_stop.configure(state="normal")
              self._start(self._stream_worker, p)

      def on_webcam(self):
          idx = int(self.dashboard_tab.cam_var.get())
          self.dashboard_tab.btn_stop.configure(state="normal")
          self._start(self._stream_worker, idx)

      def on_stop(self):
          self._running = False
          self.dashboard_tab.btn_stop.configure(state="disabled")
          self.dashboard_tab.clear_plate_data()

      def _start(self, fn, *args):
          if self._running:
              self._running = False
              if self._worker and self._worker.is_alive():
                  self._worker.join(timeout=1.5)
          self._running = True
          self._worker = Thread(target=fn, args=args, daemon=True)
          self._worker.start()

      def _image_worker(self, path):
          # Load and run ANPR detection pipeline
          try:
              from ANPR_Yolo.DetectNP import detect_fn, ocr_it, filter_text, lookup_owner
              
              data = np.fromfile(path, dtype=np.uint8)
              img = cv2.imdecode(data, cv2.IMREAD_COLOR)
              if img is None:
                  return

              cfg = self.get_config()
              dets = detect_fn(img)
              det_infos = []

              for d in dets:
                  raw_txt, ocr_info = ocr_it(d["crop"])
                  plate = filter_text(raw_txt)
                  yolo_conf = d["conf"]
                  ocr_conf = (ocr_info or {}).get("best_conf", 0.0)
                  
                  rec = lookup_owner(plate) if plate else None
                  owner = rec["owner_name"] if rec else ""
                  phone = rec["phone"] if rec else ""
                  
                  # Drawing bounding box
                  x1, y1, x2, y2 = d["bbox"]
                  cv2.rectangle(img, (x1, y1), (x2, y2), (10, 185, 129), 2)
                  
                  det_infos.append({
                      "plate": plate, "yolo_conf": yolo_conf, "ocr_conf": ocr_conf, 
                      "owner": owner, "phone": phone, "is_auth": rec is not None
                  })

              # Resize and format inside background thread to reduce main thread CPU overhead
              fh, fw = img.shape[:2]
              cw = max(self.dashboard_tab.canvas.winfo_width(), 640)
              ch = max(self.dashboard_tab.canvas.winfo_height(), 360)
              scale = min(cw / fw, ch / fh)
              nw, nh = int(fw * scale), int(fh * scale)

              resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
              rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
              pil_img = Image.fromarray(rgb)
              
              self._q.put(("frame", pil_img, det_infos, f"{fw}x{fh}"))
          except Exception as e:
              self._q.put(("error", str(e)))
          finally:
              self._q.put(("done",))

      def _stream_worker(self, source):
          try:
              from ANPR_Yolo.DetectNP import detect_fn, ocr_it, filter_text, lookup_owner, iou
              cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) else cv2.VideoCapture(source)
              if not cap.isOpened():
                  raise RuntimeError("Cannot open stream source")

              idx = 0
              last_boxes = []
              last_det_infos = []

              while self._running:
                  ok, frame = cap.read()
                  if not ok:
                      break

                  cfg = self.get_config()
                  do_detect = (idx % cfg["detect_every_n"]) == 0
                  do_ocr = (idx % cfg["ocr_every_n"]) == 0

                  det_infos = []
                  if do_detect:
                      dets = detect_fn(frame)
                      new_boxes = []
                      for d in dets:
                          bb = d["bbox"]
                          yolo_conf = d["conf"]
                          ocr_conf = 0.0
                          plate = ""
                          
                          # Re-use active cache or call OCR
                          reused = None
                          if not do_ocr:
                              for b_old, t_old, oc_old in last_boxes:
                                  if iou(bb, b_old) >= 0.45:
                                      reused = (t_old, oc_old)
                                      break
                          if reused:
                              plate, ocr_conf = reused
                          else:
                              raw_txt, ocr_info = ocr_it(d["crop"])
                              plate = filter_text(raw_txt)
                              ocr_conf = (ocr_info or {}).get("best_conf", 0.0)

                          rec = lookup_owner(plate) if plate else None
                          owner = rec["owner_name"] if rec else ""
                          phone = rec["phone"] if rec else ""

                          new_boxes.append((bb, plate, ocr_conf))
                          det_infos.append({
                              "plate": plate, "yolo_conf": yolo_conf, "ocr_conf": ocr_conf, 
                              "owner": owner, "phone": phone, "is_auth": rec is not None, "bbox": bb
                          })
                      last_boxes = new_boxes
                      last_det_infos = det_infos
                  else:
                      det_infos = last_det_infos

                  # Draw bounding boxes
                  for info in det_infos:
                      x1, y1, x2, y2 = info["bbox"]
                      cv2.rectangle(frame, (x1, y1), (x2, y2), (10, 185, 129), 2)
                      cv2.putText(frame, info["plate"], (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 185, 129), 2)

                  # Process frame shape & color space in thread
                  fh, fw = frame.shape[:2]
                  cw = max(self.dashboard_tab.canvas.winfo_width(), 640)
                  ch = max(self.dashboard_tab.canvas.winfo_height(), 360)
                  scale = min(cw / fw, ch / fh)
                  nw, nh = int(fw * scale), int(fh * scale)

                  resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
                  rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                  pil_img = Image.fromarray(rgb)

                  self._q.put(("frame", pil_img, det_infos, f"{fw}x{fh}"))
                  idx += 1
                  time.sleep(0.01) # Short throttle to ease CPU loading
              cap.release()
          except Exception as e:
              self._q.put(("error", str(e)))
          finally:
              self._q.put(("done",))

      def _poll(self):
          try:
              while True:
                  msg = self._q.get_nowait()
                  kind = msg[0]
                  if kind == "frame":
                      _, pil_img, det_infos, resolution = msg
                      self._photo = ImageTk.PhotoImage(pil_img)
                      
                      cw = max(self.dashboard_tab.canvas.winfo_width(), 640)
                      ch = max(self.dashboard_tab.canvas.winfo_height(), 360)
                      self.dashboard_tab.canvas.delete("all")
                      self.dashboard_tab.canvas.create_image((cw - pil_img.width) // 2, (ch - pil_img.height) // 2, anchor="nw", image=self._photo)
                      
                      self.dashboard_tab.lbl_dims.configure(text=resolution)
                      
                      if det_infos:
                          info = det_infos[0]
                          self.dashboard_tab.set_plate_data(info["plate"], info["yolo_conf"], info["ocr_conf"], info["owner"], info["phone"], info["is_auth"])
                      else:
                          self.dashboard_tab.clear_plate_data()
                  elif kind == "error":
                      messagebox.showerror("Execution Error", msg[1])
                  elif kind == "done":
                      self._running = False
                      self.dashboard_tab.btn_stop.configure(state="disabled")
          except queue.Empty:
              pass
          self.after(16, self._poll)
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `.venv\Scripts\python -m unittest tests/test_app.py`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add gui/app.py tests/test_app.py
  git commit -m "feat: implement main App orchestrator with navigation and background frame resizing"
  ```

---

### Task 6: Wire Up Entrypoint Launcher & Remove Legacy Code

**Files:**
- Modify: `c:\Users\dotha\OneDrive\Desktop\ANPR-using-OpenCV-Yolov8-using-python\gui_tk.py`
- Modify: `c:\Users\dotha\OneDrive\Desktop\ANPR-using-OpenCV-Yolov8-using-python\tests\test_launcher.py`

**Interfaces:**
- Consumes: `gui.app.App`
- Produces: Launcher bridge via `gui_tk.py:main()`

- [ ] **Step 1: Write a test for the entrypoint**
  Create `tests/test_launcher.py`:
  ```python
  import unittest
  import sys
  from unittest.mock import patch, MagicMock

  class TestLauncher(unittest.TestCase):
      @patch("gui.app.App")
      def test_launcher_main(self, mock_app):
          import gui_tk
          gui_tk.main()
          self.assertTrue(mock_app.called)
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `.venv\Scripts\python -m unittest tests/test_launcher.py`
  Expected: FAIL (since `gui_tk.py` still runs the old Tkinter GUI instead of invoking `gui.app.App`)

- [ ] **Step 3: Modify gui_tk.py to import the new CTk App**
  Overwrite `gui_tk.py` with:
  ```python
  """
  ANPR – CustomTkinter GUI Redesign
  Launches the new modular CustomTkinter application.
  """
  import sys
  import os

  def main():
      try:
          from gui.app import App
          app = App()
          app.mainloop()
      except Exception as e:
          print(f"Error launching ANPR GUI: {e}")
          sys.exit(1)

  if __name__ == "__main__":
      main()
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `.venv\Scripts\python -m unittest tests/test_launcher.py`
  Expected: PASS

- [ ] **Step 5: Run all unit tests**
  Run: `.venv\Scripts\python -m unittest`
  Expected: PASS all tests

- [ ] **Step 6: Commit**
  ```bash
  git add gui_tk.py tests/test_launcher.py
  git commit -m "feat: refactor gui_tk entrypoint to launch modern CustomTkinter App"
  ```

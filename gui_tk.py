"""
ANPR – Modern Tkinter GUI
Embeds video feed inside the window; shows YOLO & OCR confidence per detection.
"""
from __future__ import annotations

import os
import queue
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from threading import Thread

import cv2
import numpy as np

try:
    from PIL import Image, ImageTk
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# ─── Colour palette (GitHub-dark) ────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
CARD    = "#21262d"
BORDER  = "#30363d"
ACCENT  = "#58a6ff"
SUCCESS = "#3fb950"
WARNING = "#d29922"
DANGER  = "#f85149"
TEXT    = "#e6edf3"
SUBTEXT = "#8b949e"
SEL     = "#1f6feb"

# ─── Fonts ────────────────────────────────────────────────────────────────────
F_TITLE = ("Segoe UI", 16, "bold")
F_HEAD  = ("Segoe UI", 11, "bold")
F_BODY  = ("Segoe UI", 10)
F_SMALL = ("Segoe UI", 9)
F_MONO  = ("Consolas", 11, "bold")


def _conf_color(v: float) -> str:
    if v >= 0.80:
        return SUCCESS
    if v >= 0.55:
        return WARNING
    return DANGER


# ─── Confidence bar widget ────────────────────────────────────────────────────
class ConfBar(tk.Canvas):
    """Thin horizontal progress bar coloured by confidence level."""

    def __init__(self, parent, value: float = 0.0, **kw):
        kw.setdefault("height", 13)
        kw.setdefault("width", 100)
        kw.setdefault("bg", CARD)
        kw.setdefault("highlightthickness", 0)
        kw.setdefault("relief", "flat")
        super().__init__(parent, **kw)
        self._v = max(0.0, min(1.0, value))
        self.bind("<Configure>", lambda _: self._draw())

    def set(self, value: float) -> None:
        self._v = max(0.0, min(1.0, value))
        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        W = self.winfo_width()
        H = self.winfo_height()
        if W <= 1:
            try:
                W = int(self.cget("width"))
            except Exception:
                W = 100
        if H <= 1:
            try:
                H = int(self.cget("height"))
            except Exception:
                H = 13
        self.create_rectangle(0, 0, W, H, fill=BORDER, outline="")
        fw = max(0, int(W * self._v))
        if fw:
            self.create_rectangle(0, 0, fw, H,
                                  fill=_conf_color(self._v), outline="")
        self.create_text(W // 2, H // 2, text=f"{self._v:.0%}",
                         fill=TEXT, font=("Segoe UI", 8))


# ─── Detection card ───────────────────────────────────────────────────────────
class DetCard(tk.Frame):
    def __init__(self, parent, plate: str, yolo_conf: float,
                 ocr_conf: float, owner: str, **kw):
        kw.setdefault("bg", CARD)
        super().__init__(parent, **kw)

        inner = tk.Frame(self, bg=CARD, padx=10, pady=8)
        inner.pack(fill="x")

        # Plate number + YOLO conf inline
        row1 = tk.Frame(inner, bg=CARD)
        row1.pack(fill="x", pady=(0, 2))
        tk.Label(row1, text=plate or "—",
                 font=F_MONO, fg=ACCENT, bg=CARD).pack(side="left")
        tk.Label(row1, text=f"  {yolo_conf:.0%}",
                 font=F_SMALL, fg=_conf_color(yolo_conf), bg=CARD).pack(side="left")

        # Owner
        owner_txt = owner if owner else "Unknown owner"
        tk.Label(inner, text=owner_txt,
                 font=F_SMALL, fg=TEXT, bg=CARD).pack(anchor="w", pady=(0, 4))

        # Det bar
        row_det = tk.Frame(inner, bg=CARD)
        row_det.pack(fill="x", pady=(2, 1))
        tk.Label(row_det, text="Det ", font=F_SMALL, fg=SUBTEXT,
                 bg=CARD, width=4, anchor="e").pack(side="left")
        ConfBar(row_det, value=yolo_conf,
                height=13).pack(side="left", fill="x", expand=True, padx=(4, 0))

        # OCR bar
        row_ocr = tk.Frame(inner, bg=CARD)
        row_ocr.pack(fill="x", pady=(1, 0))
        tk.Label(row_ocr, text="OCR", font=F_SMALL, fg=SUBTEXT,
                 bg=CARD, width=4, anchor="e").pack(side="left")
        ConfBar(row_ocr, value=ocr_conf,
                height=13).pack(side="left", fill="x", expand=True, padx=(4, 0))

        tk.Frame(self, height=1, bg=BORDER).pack(fill="x")


# ─── Main application ─────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ANPR  —  License Plate Recognition")
        self.geometry("1200x720")
        self.minsize(900, 560)
        self.configure(bg=BG)

        self._q: queue.Queue = queue.Queue()
        self._running = False
        self._worker: Thread | None = None
        self._photo = None           # keep PhotoImage reference from GC
        self._total_plates = 0
        self._last_panel_upd = 0.0
        self._last_det_hash = ""

        self._apply_style()
        self._build_ui()
        self._poll()

    # ── Style ─────────────────────────────────────────────────────────────────
    def _apply_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TScrollbar", background=BORDER, troughcolor=PANEL,
                    bordercolor=PANEL, arrowcolor=SUBTEXT)
        s.configure("TCombobox", fieldbackground=CARD,
                    background=CARD, foreground=TEXT, arrowcolor=SUBTEXT)
        s.map("TCombobox", fieldbackground=[("readonly", CARD)],
              foreground=[("readonly", TEXT)])

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_header()
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=8, pady=(4, 4))
        self._build_sidebar(body)
        self._build_right_panel(body)
        self._build_canvas(body)
        self._build_statusbar()

    def _build_header(self):
        hdr = tk.Frame(self, bg=PANEL, height=54)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="  ANPR",
                 font=F_TITLE, fg=ACCENT, bg=PANEL).pack(side="left", padx=2)
        tk.Label(hdr, text="License Plate Recognition",
                 font=F_BODY, fg=SUBTEXT, bg=PANEL).pack(side="left", padx=6)

        self._live_dot = tk.Label(hdr, text="●",
                                  font=("Segoe UI", 18), fg=BORDER, bg=PANEL)
        self._live_dot.pack(side="right", padx=(0, 16))
        tk.Label(hdr, text="LIVE",
                 font=F_SMALL, fg=SUBTEXT, bg=PANEL).pack(side="right")

        self._lbl_fps = tk.Label(hdr, text="FPS: —",
                                 font=F_SMALL, fg=SUBTEXT, bg=PANEL)
        self._lbl_fps.pack(side="right", padx=18)

        self._lbl_plates = tk.Label(hdr, text="Plates: 0",
                                    font=F_SMALL, fg=SUBTEXT, bg=PANEL)
        self._lbl_plates.pack(side="right", padx=10)

    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=PANEL, width=216)
        sb.pack(side="left", fill="y", padx=(0, 6))
        sb.pack_propagate(False)

        self._section(sb, "INPUT SOURCE")

        self._btn_image = self._mk_btn(sb, "  Image", self.on_image, ACCENT)
        self._btn_image.pack(fill="x", padx=10, pady=(2, 2))

        self._btn_video = self._mk_btn(sb, "  Video", self.on_video, ACCENT)
        self._btn_video.pack(fill="x", padx=10, pady=2)

        cam_row = tk.Frame(sb, bg=PANEL)
        cam_row.pack(fill="x", padx=10, pady=2)

        self._cam_var = tk.StringVar(value="0")
        cams = self._detect_cameras()
        ttk.Combobox(cam_row, textvariable=self._cam_var, values=cams,
                     width=4, state="readonly").pack(side="left", ipady=4)

        self._btn_webcam = self._mk_btn(cam_row, "  Webcam", self.on_webcam, ACCENT)
        self._btn_webcam.pack(side="left", fill="x", expand=True, padx=(6, 0))

        self._btn_stop = self._mk_btn(sb, "  Stop", self.on_stop, DANGER)
        self._btn_stop.pack(fill="x", padx=10, pady=(8, 2))
        self._btn_stop.config(state="disabled")

        tk.Frame(sb, height=1, bg=BORDER).pack(fill="x", padx=8, pady=8)

        # History
        self._section(sb, "DETECTION HISTORY")

        hist_fr = tk.Frame(sb, bg=PANEL)
        hist_fr.pack(fill="both", expand=True, padx=6)

        scr = ttk.Scrollbar(hist_fr, orient="vertical")
        self._hist_box = tk.Listbox(
            hist_fr, bg=CARD, fg=TEXT, selectbackground=SEL,
            font=("Consolas", 9), relief="flat", bd=0,
            yscrollcommand=scr.set, activestyle="none",
            highlightthickness=0, cursor="arrow",
        )
        scr.config(command=self._hist_box.yview)
        scr.pack(side="right", fill="y")
        self._hist_box.pack(fill="both", expand=True)
        self._hist_box.bind("<Double-Button-1>", self._on_hist_dbl_click)

        tk.Label(sb, text="Double-click to register owner",
                 font=("Segoe UI", 8), fg=BORDER, bg=PANEL).pack(anchor="w", padx=10)

        self._mk_btn(sb, "Clear History", self._clear_history,
                     BORDER, fg=SUBTEXT).pack(fill="x", padx=10, pady=(4, 10))

    def _build_right_panel(self, parent):
        rp = tk.Frame(parent, bg=PANEL, width=266)
        rp.pack(side="right", fill="y", padx=(6, 0))
        rp.pack_propagate(False)

        self._section(rp, "DETECTIONS")

        det_cv = tk.Canvas(rp, bg=PANEL, bd=0, highlightthickness=0)
        det_scr = ttk.Scrollbar(rp, orient="vertical", command=det_cv.yview)
        det_cv.configure(yscrollcommand=det_scr.set)
        det_scr.pack(side="right", fill="y")
        det_cv.pack(fill="both", expand=True, padx=4, pady=(0, 6))

        self._det_inner = tk.Frame(det_cv, bg=PANEL)
        _win = det_cv.create_window((0, 0), window=self._det_inner, anchor="nw")

        self._det_inner.bind(
            "<Configure>",
            lambda _: det_cv.configure(scrollregion=det_cv.bbox("all")))
        det_cv.bind(
            "<Configure>",
            lambda e: det_cv.itemconfig(_win, width=e.width))

        tk.Label(self._det_inner, text="No detections yet",
                 font=F_SMALL, fg=SUBTEXT, bg=PANEL, pady=24).pack()

    def _build_canvas(self, parent):
        cv_fr = tk.Frame(parent, bg="#000000",
                         highlightbackground=BORDER, highlightthickness=1)
        cv_fr.pack(side="left", fill="both", expand=True)

        self._canvas = tk.Canvas(cv_fr, bg="#000000", bd=0, highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)

        self._canvas.create_text(
            400, 270,
            text="Select an image, video, or webcam to begin",
            fill=SUBTEXT, font=F_BODY,
        )

    def _build_statusbar(self):
        sb = tk.Frame(self, bg=PANEL, height=24)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)
        self._status_var = tk.StringVar(value="Ready")
        tk.Label(sb, textvariable=self._status_var,
                 font=F_SMALL, fg=SUBTEXT, bg=PANEL).pack(side="left", padx=10)
        tk.Label(sb, text="Group 4 MET5  ·  YOLOv8 + EasyOCR",
                 font=F_SMALL, fg=BORDER, bg=PANEL).pack(side="right", padx=10)

    # ── Widget helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _section(parent, title: str):
        tk.Label(parent, text=title, font=F_SMALL,
                 fg=SUBTEXT, bg=PANEL).pack(anchor="w", padx=10, pady=(10, 2))

    @staticmethod
    def _mk_btn(parent, text: str, cmd, bg: str, fg: str = BG) -> tk.Button:
        return tk.Button(
            parent, text=text, command=cmd,
            bg=bg, fg=fg, activebackground=bg, activeforeground=fg,
            font=F_BODY, bd=0, padx=6, pady=7,
            cursor="hand2", relief="flat", anchor="w",
        )

    @staticmethod
    def _detect_cameras(max_test: int = 5) -> list:
        found = []
        # Redirect stderr at the OS level to suppress VIDEOIO(DSHOW) warnings
        devnull = open(os.devnull, 'w')
        old_fd = os.dup(2)
        os.dup2(devnull.fileno(), 2)
        try:
            for i in range(max_test):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    found.append(str(i))
                    cap.release()
        finally:
            os.dup2(old_fd, 2)
            os.close(old_fd)
            devnull.close()
        return found or ["0"]

    # ── Button handlers ───────────────────────────────────────────────────────
    def on_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"),
                       ("All", "*.*")],
        )
        if path:
            self._status(f"Processing: {path}")
            self._start(self._image_worker, path)

    def on_video(self):
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Video", "*.mp4;*.avi;*.mkv;*.mov"), ("All", "*.*")],
        )
        if path:
            self._status(f"Playing: {path}")
            self._start(self._stream_worker, path)

    def on_webcam(self):
        try:
            idx = int(self._cam_var.get())
        except ValueError:
            idx = 0
        self._status(f"Webcam {idx} — live")
        self._start(self._stream_worker, idx)

    def on_stop(self):
        self._running = False
        self._status("Stopped")

    def _clear_history(self):
        self._hist_box.delete(0, "end")

    # ── Worker management ─────────────────────────────────────────────────────
    def _start(self, fn, *args):
        if self._running:
            self._running = False
            if self._worker and self._worker.is_alive():
                self._worker.join(timeout=1.5)

        self._running = True
        self._live_dot.config(fg=DANGER)
        self._btn_stop.config(state="normal")
        for b in (self._btn_image, self._btn_video, self._btn_webcam):
            b.config(state="disabled")

        self._worker = Thread(target=fn, args=args, daemon=True)
        self._worker.start()

    # ── Workers (run in background threads) ───────────────────────────────────
    def _image_worker(self, path: str):
        out = None
        det_infos = []
        try:
            from ANPR_Yolo.DetectNP import (detect_fn, filter_text,
                                             lookup_owner, ocr_it)

            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(path)

            out = img.copy()
            try:
                dets = detect_fn(img)
            except Exception:
                dets = []

            for d in dets:
                try:
                    raw_txt, ocr_info = ocr_it(d["crop"])
                except Exception:
                    raw_txt, ocr_info = "", {}
                plate = filter_text(raw_txt)
                yolo_conf = d["conf"]
                ocr_conf = (ocr_info or {}).get("best_conf", 0.0)

                rec = lookup_owner(plate) if plate else None
                owner = rec["owner_name"] if rec else ""

                x1, y1, x2, y2 = d["bbox"]
                color = (0, 220, 100)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                if plate and owner:
                    label = f"{plate} | {owner}"
                elif plate:
                    label = f"{plate} | Owner : Unknown"
                else:
                    label = f"{d.get('cls_name', 'plate')} {yolo_conf:.2f}"
                cv2.putText(out, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                det_infos.append({
                    "plate": plate, "yolo_conf": yolo_conf,
                    "ocr_conf": ocr_conf, "owner": owner,
                })
                if plate:
                    self._q.put(("hist", plate, yolo_conf, time.time()))

        except Exception as exc:
            self._q.put(("error", str(exc)))
        finally:
            if out is not None:
                self._q.put(("frame", out.copy(), det_infos, None))
            self._q.put(("done",))

    # ANPR_IMGSZ=1280 (env.py default) was tuned for single-photo analysis,
    # where a ~450ms/frame detection cost doesn't matter. Live video runs
    # detect_fn on every frame with no skipping, so that same imgsz caps
    # FPS around 1-2 regardless of OCR throttling. Live video needs
    # responsiveness more than the crowded-scene recall that imgsz=1280
    # buys, so the stream loop temporarily lowers it for its own duration.
    STREAM_IMGSZ = 640

    # No GPU on this machine: detect_fn costs ~120-200ms/frame even at
    # imgsz=640, and the LP character detector ~380ms/plate - together
    # far over the ~35-50ms budget a 20-30 FPS target allows. Detection
    # and OCR only run every Nth frame; cached boxes/text are redrawn on
    # the frames in between (a redraw-only frame costs ~8ms). Modeled
    # against measured per-stage costs: detect_every_n=8, ocr_every_n=24
    # averages to ~43ms/frame (~23 FPS) - box positions refresh ~3x/sec,
    # plate text ~1x/sec. Raise these for fresher updates at lower FPS;
    # lower them for higher FPS with staler updates.
    DETECT_EVERY_N = 8
    OCR_EVERY_N = 24

    def _stream_worker(self, source):
        original_imgsz = None
        try:
            import ANPR_Yolo.DetectNP as DetectNP
            from ANPR_Yolo.DetectNP import (detect_fn, filter_text,
                                             iou as iou_fn, lookup_owner,
                                             ocr_it)

            original_imgsz = DetectNP.ANPR_IMGSZ
            DetectNP.ANPR_IMGSZ = self.STREAM_IMGSZ

            cap = (cv2.VideoCapture(source, cv2.CAP_DSHOW)
                   if isinstance(source, int)
                   else cv2.VideoCapture(source))

            if not cap.isOpened():
                raise RuntimeError(f"Cannot open: {source}")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 450)
            cap.set(cv2.CAP_PROP_FPS, 30)

            last_boxes: list = []    # [(bbox, plate, ocr_conf)]
            last_det_infos: list = []
            last_ocr_idx = -999
            last_detect_idx = -999
            detect_every_n = self.DETECT_EVERY_N
            ocr_every_n = self.OCR_EVERY_N
            iou_keep = 0.45
            seen: dict = {}          # plate -> frame_idx (debounce)
            debounce = 30
            idx = 0
            fps_ema = None
            alpha = 0.2

            while self._running:
                ok, frame = cap.read()
                if not ok:
                    break

                tic = time.perf_counter()
                do_detect = (idx - last_detect_idx) >= detect_every_n

                if do_detect:
                    dets = []
                    try:
                        dets = detect_fn(frame)
                    except Exception:
                        pass

                    do_ocr = (idx - last_ocr_idx) >= ocr_every_n
                    new_boxes: list = []
                    det_infos: list = []

                    for d in dets:
                        bb = d.get("bbox")
                        if not bb:
                            continue
                        yolo_conf = d.get("conf", 0.0)
                        ocr_conf = 0.0

                        reused = None
                        if not do_ocr and last_boxes:
                            for bb_old, txt_old, oc_old in last_boxes:
                                if iou_fn(bb, bb_old) >= iou_keep:
                                    reused = (txt_old, oc_old)
                                    break

                        if reused:
                            plate, ocr_conf = reused
                        else:
                            try:
                                raw_txt, oi = ocr_it(d.get("crop"))
                                plate = filter_text(raw_txt)
                                ocr_conf = (oi or {}).get("best_conf", 0.0)
                            except Exception:
                                plate, ocr_conf = "", 0.0

                        owner = ""
                        try:
                            rec = lookup_owner(plate) if plate else None
                            owner = rec["owner_name"] if rec else ""
                        except Exception:
                            pass

                        new_boxes.append((bb, plate, ocr_conf))
                        det_infos.append({
                            "plate": plate, "yolo_conf": yolo_conf,
                            "ocr_conf": ocr_conf, "owner": owner,
                            "bbox": bb, "cls_name": d.get("cls_name", "plate"),
                        })

                        # debounced history log
                        if plate and (idx - seen.get(plate, -debounce * 2)) > debounce:
                            seen[plate] = idx
                            self._q.put(("hist", plate, yolo_conf, time.time()))

                    if do_ocr:
                        last_ocr_idx = idx
                    last_boxes = new_boxes
                    last_det_infos = det_infos
                    last_detect_idx = idx
                else:
                    # Skip detect_fn/ocr_it entirely this frame (the
                    # dominant costs on this CPU) - redraw the last known
                    # boxes/labels on the current frame instead, so the
                    # displayed video stays smooth between real detection
                    # cycles rather than flickering or stalling.
                    det_infos = last_det_infos

                # draw current box/label set (fresh this cycle, or reused)
                for info in det_infos:
                    bb = info.get("bbox")
                    if not bb:
                        continue
                    x1, y1, x2, y2 = bb
                    plate = info["plate"]
                    owner = info["owner"]
                    color = (0, 220, 100)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    if plate and owner:
                        label = f"{plate} | {owner}"
                    elif plate:
                        label = f"{plate} | Owner : Unknown"
                    else:
                        label = f"{info.get('cls_name', 'plate')} {info['yolo_conf']:.2f}"
                    cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # FPS overlay
                inst = 1.0 / max(1e-6, time.perf_counter() - tic)
                fps_ema = (inst if fps_ema is None
                           else alpha * inst + (1 - alpha) * fps_ema)
                cv2.putText(frame, f"FPS {fps_ema:.1f}", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 100), 2)

                self._q.put(("frame", frame.copy(), det_infos, fps_ema))
                idx += 1

            cap.release()

        except Exception as exc:
            self._q.put(("error", str(exc)))
        finally:
            if original_imgsz is not None:
                import ANPR_Yolo.DetectNP as DetectNP
                DetectNP.ANPR_IMGSZ = original_imgsz
            self._q.put(("done",))

    # ── Queue → UI (runs on main thread) ─────────────────────────────────────
    def _poll(self):
        try:
            while True:
                msg = self._q.get_nowait()
                kind = msg[0]
                if kind == "frame":
                    _, frame, det_infos, fps = msg
                    self._show_frame(frame)
                    self._refresh_det_panel(det_infos)
                    if fps is not None:
                        self._lbl_fps.config(text=f"FPS: {fps:.1f}")
                elif kind == "hist":
                    _, plate, conf, ts = msg
                    self._add_history(plate, conf, ts)
                elif kind == "error":
                    messagebox.showerror("Error", msg[1])
                    self._on_done()
                elif kind == "done":
                    self._on_done()
        except queue.Empty:
            pass
        self.after(16, self._poll)   # ~60 Hz

    def _show_frame(self, frame_bgr: np.ndarray):
        if not _PIL_OK:
            return
        cw = max(self._canvas.winfo_width(), 640)
        ch = max(self._canvas.winfo_height(), 360)
        fh, fw = frame_bgr.shape[:2]
        scale = min(cw / fw, ch / fh)
        nw, nh = int(fw * scale), int(fh * scale)

        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))

        self._canvas.delete("all")
        self._canvas.create_image(
            (cw - nw) // 2, (ch - nh) // 2, anchor="nw", image=photo)
        self._photo = photo   # prevent GC

    def _refresh_det_panel(self, det_infos: list):
        # Throttle rebuilds to ~4 Hz; skip if nothing changed
        now = time.time()
        new_hash = "|".join(
            f"{d['plate']}{d['yolo_conf']:.1f}{d['ocr_conf']:.1f}"
            for d in det_infos
        )
        if new_hash == self._last_det_hash and (now - self._last_panel_upd) < 0.25:
            return
        self._last_det_hash = new_hash
        self._last_panel_upd = now

        for w in self._det_inner.winfo_children():
            w.destroy()

        if not det_infos:
            tk.Label(self._det_inner, text="No plates detected",
                     font=F_SMALL, fg=SUBTEXT, bg=PANEL, pady=24).pack()
            return

        for info in det_infos:
            DetCard(
                self._det_inner,
                plate=info["plate"],
                yolo_conf=info["yolo_conf"],
                ocr_conf=info["ocr_conf"],
                owner=info["owner"],
            ).pack(fill="x", padx=4, pady=3)

    def _add_history(self, plate: str, conf: float, ts: float):
        t_str = time.strftime("%H:%M:%S", time.localtime(ts))
        entry = f" {t_str}  {plate:<12}  {conf:.0%}"
        # avoid consecutive duplicate plate entries
        size = self._hist_box.size()
        if size > 0 and plate in self._hist_box.get(size - 1):
            return
        self._hist_box.insert("end", entry)
        self._hist_box.see("end")
        self._total_plates += 1
        self._lbl_plates.config(text=f"Plates: {self._total_plates}")

    def _on_done(self):
        self._running = False
        self._live_dot.config(fg=BORDER)
        self._btn_stop.config(state="disabled")
        for b in (self._btn_image, self._btn_video, self._btn_webcam):
            b.config(state="normal")
        self._status("Done")

    def _status(self, msg: str):
        self._status_var.set(msg)

    # ── Registry dialog ───────────────────────────────────────────────────────
    def _on_hist_dbl_click(self, _event):
        sel = self._hist_box.curselection()
        if not sel:
            return
        parts = self._hist_box.get(sel[0]).strip().split()
        if len(parts) < 2:
            return
        self._open_registry_dialog(parts[1])

    def _open_registry_dialog(self, plate: str):
        import shutil as _shutil
        import pandas as pd
        try:
            from ANPR_Yolo.DetectNP import (load_registry, save_registry,
                                             canonicalize_plate)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        try:
            df = load_registry()
            key = canonicalize_plate(plate)
            hit = df[df["plate_norm"] == key]
            pre = hit.iloc[0].to_dict() if not hit.empty else {}
        except Exception:
            df = pd.DataFrame(columns=["plate", "owner_name", "phone", "notes",
                                        "photo", "plate_norm"])
            pre = {}

        # ── Dialog window ────────────────────────────────────────────────────
        dlg = tk.Toplevel(self)
        dlg.title(f"Register Owner — {plate}")
        dlg.configure(bg=PANEL)
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.geometry("440x320")

        # Pack button bar at bottom first so it always stays visible
        btn_fr = tk.Frame(dlg, bg=PANEL, padx=20, pady=12)
        btn_fr.pack(side="bottom", fill="x")

        fr = tk.Frame(dlg, bg=PANEL, padx=20, pady=14)
        fr.pack(fill="both", expand=True)

        tk.Label(fr, text=plate, font=F_MONO, fg=ACCENT,
                 bg=PANEL).grid(row=0, column=0, columnspan=2,
                                sticky="w", pady=(0, 10))

        field_specs = [
            ("Owner Name", "owner_name"),
            ("Phone",      "phone"),
            ("Notes",      "notes"),
        ]
        entry_vars = {}
        for i, (lbl, key_) in enumerate(field_specs, start=1):
            tk.Label(fr, text=lbl, font=F_SMALL, fg=SUBTEXT, bg=PANEL,
                     anchor="e", width=11).grid(row=i, column=0,
                                                sticky="e", pady=5, padx=(0, 8))
            var = tk.StringVar(value=str(pre.get(key_, "") or ""))
            tk.Entry(fr, textvariable=var, font=F_BODY,
                     bg=CARD, fg=TEXT, insertbackground=TEXT,
                     relief="flat", bd=4, width=30).grid(row=i, column=1,
                                                          sticky="ew", pady=5)
            entry_vars[key_] = var

        # Photo row
        photo_row = len(field_specs) + 1
        tk.Label(fr, text="Photo", font=F_SMALL, fg=SUBTEXT, bg=PANEL,
                 anchor="e", width=11).grid(row=photo_row, column=0,
                                             sticky="e", pady=5, padx=(0, 8))

        photo_fr = tk.Frame(fr, bg=PANEL)
        photo_fr.grid(row=photo_row, column=1, sticky="ew", pady=5)

        _initial_photo = str(pre.get("photo", "") or "")
        photo_display = tk.StringVar(
            value=os.path.basename(_initial_photo) if _initial_photo else "No photo selected")
        tk.Label(photo_fr, textvariable=photo_display, font=F_SMALL,
                 fg=SUBTEXT, bg=PANEL, anchor="w").pack(side="left",
                                                         fill="x", expand=True)

        _chosen = {"full": _initial_photo}

        def browse_photo():
            p = filedialog.askopenfilename(
                parent=dlg, title="Select owner photo",
                filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")])
            if p:
                _chosen["full"] = p
                photo_display.set(os.path.basename(p))

        tk.Button(photo_fr, text="Browse…", command=browse_photo,
                  bg=CARD, fg=TEXT, font=F_SMALL, relief="flat",
                  padx=6, pady=3, cursor="hand2").pack(side="right")

        fr.columnconfigure(1, weight=1)

        # ── Action buttons (btn_fr already packed at bottom) ─────────────────
        def on_save():
            try:
                df2 = load_registry()
            except Exception:
                df2 = pd.DataFrame(columns=["plate", "owner_name", "phone",
                                             "notes", "photo", "plate_norm"])

            if "photo" not in df2.columns:
                df2["photo"] = ""

            # Copy photo to owners dir if a new file was chosen
            dest_photo = _initial_photo
            src = _chosen["full"]
            if src and os.path.isfile(src) and src != _initial_photo:
                owners_dir = os.path.join("runs", "anpr_yolo", "owners")
                os.makedirs(owners_dir, exist_ok=True)
                canon = canonicalize_plate(plate).replace("-", "_").replace(".", "_")
                ext = os.path.splitext(src)[1] or ".jpg"
                dest = os.path.join(owners_dir, f"{canon}{ext}")
                _shutil.copy2(src, dest)
                dest_photo = dest

            new_row = {
                "plate":      plate,
                "owner_name": entry_vars["owner_name"].get().strip(),
                "phone":      entry_vars["phone"].get().strip(),
                "notes":      entry_vars["notes"].get().strip(),
                "photo":      dest_photo,
            }

            key2 = canonicalize_plate(plate)
            mask = df2["plate_norm"] == key2 if "plate_norm" in df2.columns \
                else pd.Series([False] * len(df2))

            if mask.any():
                for col, val in new_row.items():
                    if col in df2.columns:
                        df2.loc[mask, col] = val
            else:
                df2 = pd.concat([df2, pd.DataFrame([new_row])],
                                 ignore_index=True)

            save_registry(df2)
            dlg.destroy()
            messagebox.showinfo("Saved",
                                f"Owner registered for {plate}:\n"
                                f"{new_row['owner_name'] or '(no name)'}")

        tk.Button(btn_fr, text="Cancel", command=dlg.destroy,
                  bg=BORDER, fg=TEXT, font=F_BODY,
                  relief="flat", padx=10, pady=6,
                  cursor="hand2").pack(side="right", padx=(6, 0))
        tk.Button(btn_fr, text="Save", command=on_save,
                  bg=ACCENT, fg=BG, font=("Segoe UI", 10, "bold"),
                  relief="flat", padx=10, pady=6,
                  cursor="hand2").pack(side="right")


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    if not _PIL_OK:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Missing dependency",
            "Pillow is required for the GUI.\n\nInstall it with:\n  pip install Pillow",
        )
        root.destroy()
        return
    App().mainloop()

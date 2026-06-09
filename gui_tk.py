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
        try:
            from ANPR_Yolo.DetectNP import (detect_fn, filter_text,
                                             lookup_owner, ocr_it)

            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(path)

            dets = detect_fn(img)
            out = img.copy()
            det_infos = []

            for d in dets:
                raw_txt, ocr_info = ocr_it(d["crop"])
                plate = filter_text(raw_txt)
                yolo_conf = d["conf"]
                ocr_conf = (ocr_info or {}).get("best_conf", 0.0)

                rec = lookup_owner(plate) if plate else None
                owner = rec["owner_name"] if rec else ""

                x1, y1, x2, y2 = d["bbox"]
                color = (0, 220, 100)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                main_lbl = (f"{plate}  {yolo_conf:.0%}" if plate
                            else f"plate  {yolo_conf:.0%}")
                sub_lbl = owner or "Unknown owner"
                cv2.putText(out, main_lbl, (x1, max(20, y1 - 22)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
                cv2.putText(out, sub_lbl, (x1, max(6, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1)

                det_infos.append({
                    "plate": plate, "yolo_conf": yolo_conf,
                    "ocr_conf": ocr_conf, "owner": owner,
                })
                if plate:
                    self._q.put(("hist", plate, yolo_conf, time.time()))

            self._q.put(("frame", out.copy(), det_infos, None))

        except Exception as exc:
            self._q.put(("error", str(exc)))
        finally:
            self._q.put(("done",))

    def _stream_worker(self, source):
        try:
            from ANPR_Yolo.DetectNP import (detect_fn, filter_text,
                                             iou as iou_fn, lookup_owner,
                                             ocr_it)

            cap = (cv2.VideoCapture(source, cv2.CAP_DSHOW)
                   if isinstance(source, int)
                   else cv2.VideoCapture(source))

            if not cap.isOpened():
                raise RuntimeError(f"Cannot open: {source}")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 450)
            cap.set(cv2.CAP_PROP_FPS, 30)

            last_boxes: list = []    # [(bbox, plate, ocr_conf)]
            last_ocr_idx = -999
            ocr_every_n = 8
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
                    })

                    # draw on frame
                    x1, y1, x2, y2 = bb
                    color = (0, 220, 100)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    main_lbl = (f"{plate}  {yolo_conf:.0%}" if plate
                                else f"plate  {yolo_conf:.0%}")
                    sub_lbl = owner or "Unknown"
                    cv2.putText(frame, main_lbl, (x1, max(20, y1 - 22)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2)
                    cv2.putText(frame, sub_lbl, (x1, max(6, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

                    # debounced history log
                    if plate and (idx - seen.get(plate, -debounce * 2)) > debounce:
                        seen[plate] = idx
                        self._q.put(("hist", plate, yolo_conf, time.time()))

                if do_ocr:
                    last_ocr_idx = idx
                last_boxes = new_boxes

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

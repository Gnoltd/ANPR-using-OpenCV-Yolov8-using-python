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

try:
    import ANPR_Yolo.DetectNP as DetectNP
    from ANPR_Yolo.DetectNP import detect_fn, ocr_it, filter_text, lookup_owner, iou
except ImportError:
    import DetectNP as DetectNP
    from DetectNP import detect_fn, ocr_it, filter_text, lookup_owner, iou

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("ANPR Vision — License Plate Recognition")
        self.geometry("1150x680")
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
        self.sidebar.pack_propagate(False)
        
        # SVG camera logo container
        logo_canvas = tk.Canvas(self.sidebar, width=32, height=32, bg="#030408", highlightthickness=0, bd=0)
        logo_canvas.pack(pady=(25, 10))
        # Draw camera outline representation on canvas
        logo_canvas.create_rectangle(2, 8, 30, 28, outline="#38bdf8", width=2)
        logo_canvas.create_circle = lambda x, y, r, **kwargs: logo_canvas.create_oval(x-r, y-r, x+r, y+r, **kwargs)
        logo_canvas.create_circle(16, 18, 5, outline="#38bdf8", width=2)
        logo_canvas.create_polygon(10, 8, 13, 3, 19, 3, 22, 8, outline="#38bdf8", width=2, fill="#030408")

        lbl = ctk.CTkLabel(self.sidebar, text="ANPR VISION", font=("Inter", 16, "bold"), text_color="#f8fafc")
        lbl.pack(padx=20, pady=(0, 25))

        self.btn_dash = ctk.CTkButton(self.sidebar, text="📺 Detection Dashboard", command=lambda: self.select_tab("dash"), fg_color="transparent", text_color="#94a3b8", anchor="w", height=40)
        self.btn_dash.pack(fill="x", padx=15, pady=5)
        
        self.btn_reg = ctk.CTkButton(self.sidebar, text="📇 Vehicle Database", command=lambda: self.select_tab("reg"), fg_color="transparent", text_color="#94a3b8", anchor="w", height=40)
        self.btn_reg.pack(fill="x", padx=15, pady=5)
        
        self.btn_sett = ctk.CTkButton(self.sidebar, text="⚙️ Config & Sliders", command=lambda: self.select_tab("sett"), fg_color="transparent", text_color="#94a3b8", anchor="w", height=40)
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
            self.btn_dash.configure(fg_color="#0f172a", text_color="#38bdf8")
        elif name == "reg":
            self.registry_tab.pack(fill="both", expand=True)
            self.registry_tab.refresh_list()
            self.btn_reg.configure(fg_color="#0f172a", text_color="#38bdf8")
        elif name == "sett":
            self.settings_tab.pack(fill="both", expand=True)
            self.btn_sett.configure(fg_color="#0f172a", text_color="#38bdf8")

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
        try:
            idx = int(self.dashboard_tab.cam_var.get())
        except ValueError:
            idx = 0
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
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                return

            dets = detect_fn(img)
            det_infos = []

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
            self._q.put(("frame", pil_img, det_infos, f"{fw}x{fh}", None))
        except Exception as e:
            self._q.put(("error", str(e)))
        finally:
            self._q.put(("done",))

    def _stream_worker(self, source):
        original_imgsz = None
        try:
            original_imgsz = DetectNP.ANPR_IMGSZ
            DetectNP.ANPR_IMGSZ = 640  # Temporarily lower for live stream responsiveness

            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) else cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError("Cannot open stream source")

            idx = 0
            last_boxes = []
            last_det_infos = []
            fps_ema = None
            alpha = 0.2
            prev_frame_tic = time.perf_counter()

            while self._running:
                ok, frame = cap.read()
                if not ok:
                    break

                now = time.perf_counter()
                inst = 1.0 / max(1e-6, now - prev_frame_tic)
                prev_frame_tic = now

                cfg = self.get_config()
                do_detect = (idx % cfg["detect_every_n"]) == 0
                do_ocr = (idx % cfg["ocr_every_n"]) == 0

                det_infos = []
                if do_detect:
                    try:
                        dets = detect_fn(frame)
                    except Exception:
                        dets = []
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
                            try:
                                raw_txt, ocr_info = ocr_it(d["crop"])
                                plate = filter_text(raw_txt)
                                ocr_conf = (ocr_info or {}).get("best_conf", 0.0)
                            except Exception:
                                plate, ocr_conf = "", 0.0

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

                # FPS overlay (inst computed from true frame-to-frame time above)
                fps_ema = (inst if fps_ema is None else alpha * inst + (1 - alpha) * fps_ema)
                cv2.putText(frame, f"FPS {fps_ema:.1f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 185, 129), 2)

                # Process frame shape & color space in thread
                fh, fw = frame.shape[:2]
                cw = max(self.dashboard_tab.canvas.winfo_width(), 640)
                ch = max(self.dashboard_tab.canvas.winfo_height(), 360)
                scale = min(cw / fw, ch / fh)
                nw, nh = int(fw * scale), int(fh * scale)

                resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                self._q.put(("frame", pil_img, det_infos, f"{fw}x{fh}", fps_ema))
                idx += 1
                
                # Backpressure throttle: sleep if UI is slower than stream
                if self._q.qsize() > 2:
                    time.sleep(0.01)
            cap.release()
        except Exception as e:
            self._q.put(("error", str(e)))
        finally:
            if original_imgsz is not None:
                DetectNP.ANPR_IMGSZ = original_imgsz
            self._q.put(("done",))

    def _poll(self):
        try:
            while True:
                msg = self._q.get_nowait()
                kind = msg[0]
                if kind == "frame":
                    _, pil_img, det_infos, resolution, fps = msg
                    self._photo = ImageTk.PhotoImage(pil_img)
                    
                    cw = max(self.dashboard_tab.canvas.winfo_width(), 640)
                    ch = max(self.dashboard_tab.canvas.winfo_height(), 360)
                    self.dashboard_tab.canvas.delete("all")
                    self.dashboard_tab.canvas.create_image((cw - pil_img.width) // 2, (ch - pil_img.height) // 2, anchor="nw", image=self._photo)
                    
                    if fps is not None:
                        self.dashboard_tab.lbl_dims.configure(text=f"{resolution} | {fps:.1f} FPS")
                    else:
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

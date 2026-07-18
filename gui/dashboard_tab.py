import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import cv2

class DashboardTab(ctk.CTkFrame):
    def __init__(self, parent, on_image, on_video, on_webcam, on_stop, **kwargs):
        super().__init__(parent, fg_color="#090a0f", **kwargs)
        
        # Grid configuration
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
        hdr.grid(row=0, column=0, sticky="ew", padx=15, pady=10)
        
        ctk.CTkLabel(hdr, text="Camera Stream Viewport", font=("Inter", 13, "bold"), text_color="#94a3b8").pack(side="left")
        self.lbl_dims = ctk.CTkLabel(hdr, text="No input source", font=("JetBrains Mono", 11), text_color="#38bdf8")
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

import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import cv2

class VehicleCard(ctk.CTkFrame):
    def __init__(self, parent, plate, yolo_conf, ocr_conf, owner, phone, is_authorized=False, **kwargs):
        super().__init__(parent, fg_color="#0c0e17", border_width=1, border_color="#1e293b", corner_radius=8, **kwargs)
        
        self.pack(fill="x", padx=5, pady=6)
        
        # Header layout
        hdr_fr = ctk.CTkFrame(self, fg_color="transparent")
        hdr_fr.pack(fill="x", padx=12, pady=(10, 4))
        
        disp_plate = plate if plate else "PLATE DETECTED"
        lbl_plate = ctk.CTkLabel(hdr_fr, text=disp_plate, font=("JetBrains Mono", 15, "bold"), text_color="#38bdf8" if plate else "#94a3b8")
        lbl_plate.pack(side="left")
        
        status_text = "Authorized" if is_authorized else ("Unknown" if plate else "Locating...")
        status_color = "#052e16" if is_authorized else ("#450a0a" if plate else "#1e293b")
        text_color = "#4ade80" if is_authorized else ("#f87171" if plate else "#94a3b8")
        
        badge = ctk.CTkLabel(hdr_fr, text=status_text, font=("Inter", 9, "bold"), 
                             fg_color=status_color, text_color=text_color, 
                             corner_radius=4, height=18, width=65)
        badge.pack(side="right")
        
        # Owner details box
        det_fr = ctk.CTkFrame(self, fg_color="#05070d", border_width=1, border_color="#1e293b", corner_radius=6)
        det_fr.pack(fill="x", padx=12, pady=6)
        
        lbl_owner = ctk.CTkLabel(det_fr, text=f"Owner: {owner if owner else 'Unknown'}", font=("Inter", 11), text_color="#e2e8f0", anchor="w")
        lbl_owner.pack(fill="x", padx=10, pady=3)
        
        lbl_phone = ctk.CTkLabel(det_fr, text=f"Phone: {phone if phone else '—'}", font=("Inter", 11), text_color="#e2e8f0", anchor="w")
        lbl_phone.pack(fill="x", padx=10, pady=(0, 3))
        
        # Confidence bars
        self._build_progress(self, "YOLO Detection Conf", yolo_conf)
        self._build_progress(self, "EasyOCR Fallback Conf", ocr_conf)

    def _build_progress(self, parent, title, val):
        prog_fr = ctk.CTkFrame(parent, fg_color="transparent")
        prog_fr.pack(fill="x", padx=12, pady=(2, 6))
        
        lbl = ctk.CTkLabel(prog_fr, text=f"{title}: {val:.0%}", font=("Inter", 10), text_color="#64748b", anchor="w")
        lbl.pack(fill="x")
        
        bar = ctk.CTkProgressBar(prog_fr, height=4, fg_color="#05070d", progress_color="#10b981")
        bar.pack(fill="x", pady=(2, 2))
        bar.set(val)

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
        self.mon_fr = ctk.CTkFrame(self, fg_color="#030408", border_width=1, border_color="#1e293b")
        self.mon_fr.grid(row=0, column=1, sticky="nsew", padx=(10, 20), pady=20)
        
        ctk.CTkLabel(self.mon_fr, text="Live Plate Monitor", font=("Inter", 13, "bold"), text_color="#64748b").pack(anchor="w", padx=15, pady=10)

        # Scrollable container for vehicle cards
        self.cards_scroll = ctk.CTkScrollableFrame(self.mon_fr, fg_color="transparent")
        self.cards_scroll.pack(fill="both", expand=True, padx=5, pady=(0, 10))

        self.clear_plate_data()

    def set_plates_data(self, det_infos):
        # Clear existing cards
        for child in self.cards_scroll.winfo_children():
            child.destroy()
            
        if not det_infos:
            lbl_placeholder = ctk.CTkLabel(self.cards_scroll, text="No plates detected", font=("Inter", 12), text_color="#64748b")
            lbl_placeholder.pack(pady=40)
            return

        for info in det_infos:
            VehicleCard(
                self.cards_scroll,
                plate=info.get("plate", ""),
                yolo_conf=info.get("yolo_conf", 0.0),
                ocr_conf=info.get("ocr_conf", 0.0),
                owner=info.get("owner", ""),
                phone=info.get("phone", ""),
                is_authorized=info.get("is_auth", False)
            )

    def set_plate_data(self, plate, yolo_conf, ocr_conf, owner, phone, is_authorized=False):
        # Backward compatibility fallback
        info = {
            "plate": plate,
            "yolo_conf": yolo_conf,
            "ocr_conf": ocr_conf,
            "owner": owner,
            "phone": phone,
            "is_auth": is_authorized
        }
        self.set_plates_data([info])

    def clear_plate_data(self):
        for child in self.cards_scroll.winfo_children():
            child.destroy()
        lbl_placeholder = ctk.CTkLabel(self.cards_scroll, text="System Standby", font=("Inter", 12), text_color="#64748b")
        lbl_placeholder.pack(pady=40)

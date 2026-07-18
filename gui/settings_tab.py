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

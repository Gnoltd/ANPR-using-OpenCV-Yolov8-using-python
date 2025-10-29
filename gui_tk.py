from tkinter import Tk, Label, Button, OptionMenu, StringVar, filedialog, messagebox
from threading import Thread
import cv2

try:
    from ANPR_Yolo.ANPR import run_image, run_video, run_webcam
except ImportError:
    messagebox.showerror("Import Error")

def _run_in_thread(fn, *args, **kwargs):
    t = Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
    t.start()
    return t

class App(Tk):
    def __init__(self):
        super().__init__()
        self.title("form")
        self.geometry("600x420")

        Label(self, text="Group 4 MET5", font=("Segoe UI", 12)).pack(pady=10)
        Button(self, text="Detect IMAGE", command=self.on_image).pack(pady=6)
        Button(self, text="Detect VIDEO", command=self.on_video).pack(pady=6)

        Label(self, text="Select Webcam:").pack(pady=4)

        self.webcam_var = StringVar(value="0")
        self.webcam_list = self._detect_cameras()

        OptionMenu(self, self.webcam_var, *self.webcam_list).pack(pady=4)

        Button(self, text="Start WEBCAM", command=self.on_webcam).pack(pady=6)
        Button(self, text="Quit", command=self.destroy).pack(pady=12)

    def _detect_cameras(self, max_test=5):
        cams = []
        for i in range(max_test):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW giúp mở nhanh trên Windows
            if cap.isOpened():
                cams.append(str(i))
                cap.release()
        if not cams:
            cams = ["0"]  # fallback
        return cams

    def on_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("All files", "*.*")]
        )
        if not path:
            return
        _run_in_thread(run_image, path)

    def on_video(self):
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Video", "*.mp4;*.avi;*.mkv;*.mov"), ("All files", "*.*")]
        )
        if not path:
            return
        _run_in_thread(run_video, path)

    def on_webcam(self):
        try:
            index = int(self.webcam_var.get())
            _run_in_thread(run_webcam, index)
        except Exception as e:
            messagebox.showerror("Error", str(e))

def main():
    App().mainloop()

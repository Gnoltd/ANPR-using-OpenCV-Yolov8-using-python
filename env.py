import os, sys
from pathlib import Path
PLATE_CLASS_NAMES = set()
BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
WEIGHTS = str(BASE_DIR / "best.pt")
CONF_THRES = 0.25
IOU_THRES  = 0.45
ANPR_DEVICE = "cpu"
ANPR_IMGSZ = 640
ANPR_USE_HALF = 0
SAVE_DIR = "runs/anpr_yolo"
MIN_AREA = int(os.environ.get("ANPR_MIN_AREA", "300"))
MAX_AR   = float(os.environ.get("ANPR_MAX_AR", "8.0"))
MIN_AR   = float(os.environ.get("ANPR_MIN_AR", "1.0"))

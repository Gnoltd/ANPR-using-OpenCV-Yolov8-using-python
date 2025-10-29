
from ultralytics import YOLO
import easyocr
from ANPR_Yolo.env import *
_yolo_model = None
_easy_reader = None
_PRINTED_MODEL_INFO = False
def _load_model(weights=WEIGHTS):

    global _yolo_model
    if _yolo_model is None:
        if YOLO is None:
            raise RuntimeError()
        _yolo_model =YOLO(weights)

        try:
            _yolo_model.fuse()
        except Exception:
            pass
    return _yolo_model

def _load_ocr():

    global _easy_reader
    if _easy_reader is None:
        if easyocr is None:
            raise RuntimeError()
        _easy_reader = easyocr.Reader(['en', 'vi'], gpu=False)
    return _easy_reader
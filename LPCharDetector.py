import json
import os
import sys

DEFAULT_STATE_DICT_PATH = "lp_char_detector_state_dict.pt"
DEFAULT_ARCH_PATH = "lp_char_detector_arch.json"

ROW_GAP_FRAC = 0.18


def order_detected_chars(detections, crop_h, row_gap_frac=ROW_GAP_FRAC):
    """detections: list of (x1, y1, x2, y2, char, conf) tuples, any order.
    Groups into 1+ rows by vertical center (top-to-bottom), sorts each row
    left-to-right by horizontal center, and concatenates. Returns
    (text, min_confidence, row_count), or ("", 0.0, 0) for no detections.
    Row grouping threshold matches ocr_it's existing EasyOCR row-grouping
    constant (0.18 * crop height) for consistency with the rest of this
    codebase. row_count is physical evidence (how many distinct text rows
    were actually found) usable to disambiguate car-vs-moto formatting for
    a compact string that's ambiguous by character count alone - see
    filter_text's row_hint parameter in DetectNP.py."""
    if not detections:
        return "", 0.0, 0

    items = []
    confs = []
    for x1, y1, x2, y2, ch, conf in detections:
        items.append(((y1 + y2) / 2.0, (x1 + x2) / 2.0, ch))
        confs.append(conf)

    items.sort(key=lambda t: t[0])
    row_thresh = max(8, row_gap_frac * crop_h)
    rows = []
    for y, x, ch in items:
        if not rows or abs(y - rows[-1][-1][0]) > row_thresh:
            rows.append([(y, x, ch)])
        else:
            rows[-1].append((y, x, ch))

    text = ""
    for row in rows:
        row_sorted = sorted(row, key=lambda t: t[1])
        text += "".join(ch for _y, _x, ch in row_sorted)

    return text, min(confs), len(rows)


def load_lp_char_detector(state_dict_path=DEFAULT_STATE_DICT_PATH, arch_path=DEFAULT_ARCH_PATH):
    """Safely loads the pretrained VN-plate character detector: pure-code
    architecture reconstruction from a JSON config plus a weights_only=True
    state_dict load - no pickle deserialization of arbitrary objects.

    (The original checkpoint, sourced from a third-party GitHub repo, was a
    full pickled YOLOv5 model object requiring torch.load(weights_only=False)
    to open - explicitly authorized once, offline, to extract just the
    tensor weights and the plain-data architecture config below. This
    function never does that unsafe load; it only consumes the two safe
    artifacts produced by that one-time, explicitly-authorized conversion.)
    """
    if not os.path.exists(state_dict_path) or not os.path.exists(arch_path):
        raise FileNotFoundError(
            f"LP character detector artifacts not found: {state_dict_path}, {arch_path}\n"
            "These must be generated once via the safe extraction procedure "
            "documented in docs/superpowers/notes/ (not regenerated automatically, "
            "since it requires an explicitly-trusted third-party checkpoint)."
        )

    import yolov5  # noqa: F401 - import needed so `models`/`utils` subpackages resolve below
    yolov5_dir = os.path.dirname(yolov5.__file__)
    if yolov5_dir not in sys.path:
        sys.path.insert(0, yolov5_dir)

    from models.yolo import DetectionModel
    from models.common import AutoShape
    import torch

    with open(arch_path) as f:
        arch = json.load(f)
    names = {int(k): v for k, v in arch["names"].items()}

    core = DetectionModel(cfg=arch["yaml"])
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    core.load_state_dict(state_dict)
    core.eval()
    core.names = names

    model = AutoShape(core)
    model.conf = 0.25
    return model


def detect_plate_text(crop_bgr, model):
    """Runs the character detector on a plate crop and returns (text,
    min_confidence, row_count), or ("", 0.0, 0) if nothing was detected."""
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0, 0

    results = model(crop_bgr)
    preds = results.pred[0]
    names = model.model.names if hasattr(model, "model") else model.names

    detections = []
    for *box, conf, cls in preds.tolist():
        x1, y1, x2, y2 = box
        detections.append((x1, y1, x2, y2, names[int(cls)], conf))

    return order_detected_chars(detections, crop_h=crop_bgr.shape[0])

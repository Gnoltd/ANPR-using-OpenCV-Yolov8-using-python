import json
import os
import sys

DEFAULT_STATE_DICT_PATH = "lp_char_detector_state_dict.pt"
DEFAULT_ARCH_PATH = "lp_char_detector_arch.json"

ROW_GAP_FRAC = 0.18

# Real 2-line motorbike plates always split as exactly this shape: line 1
# is province (2 digits) + series (letter+digit) = 4 characters; line 2 is
# the body, 4 digits (compact) or 5 digits (dotted). Matches
# _MOTO_ROW1_COUNT / _MOTO_ROW2_COUNTS in DetectNP.py / PlateOCR.py.
_MOTO_ROW1_COUNT = 4
_MOTO_ROW2_COUNTS = (4, 5)


def order_detected_chars(detections, crop_h, row_gap_frac=ROW_GAP_FRAC):
    """detections: list of (x1, y1, x2, y2, char, conf) tuples, any order.
    Groups into 1+ visual clusters (top-to-bottom) by vertical center,
    sorts each cluster left-to-right by horizontal center, and
    concatenates. Returns (text, min_confidence, row_count), or
    ("", 0.0, 0) for no detections.

    Clustering (for text assembly) always uses the loose crop-height-
    fraction threshold (matches ocr_it's existing EasyOCR row-grouping
    constant). A photographed plate is often slightly skewed, so even a
    genuinely single physical row can split into two visual clusters
    (e.g. the province+letter group sitting a few pixels higher than the
    digit group) - the loose threshold correctly separates these clusters
    so characters are read in the right left-to-right order, rather than
    interleaving two groups that don't actually share an x-range.

    The returned row_count (used as physical evidence for car/moto
    disambiguation - see filter_text's row_hint parameter in DetectNP.py)
    only trusts a 2-cluster split as a genuine second physical text row
    if the cluster sizes match a real moto plate's fixed shape (exactly
    4 characters in the first cluster, 4 or 5 in the second). A pixel-
    geometry heuristic (gap vs. average character height) was tried first
    and found unreliable on real data: a skewed single-row car plate's
    cluster gap-to-height ratio (1.067) was actually LARGER than a
    genuine 2-row moto plate's ratio (0.994) elsewhere in this eval set -
    the two cases are not geometrically separable, only content-
    separable, since real moto plates have a fixed, known layout shape
    that skew artifacts don't reproduce."""
    if not detections:
        return "", 0.0, 0

    items = []
    confs = []
    for x1, y1, x2, y2, ch, conf in detections:
        items.append(((y1 + y2) / 2.0, (x1 + x2) / 2.0, ch))
        confs.append(conf)

    items.sort(key=lambda t: t[0])

    text_row_thresh = max(8, row_gap_frac * crop_h)
    rows = []
    for y, x, ch in items:
        if not rows or (y - rows[-1][-1][0]) > text_row_thresh:
            rows.append([(y, x, ch)])
        else:
            rows[-1].append((y, x, ch))

    text = ""
    for row in rows:
        row_sorted = sorted(row, key=lambda t: t[1])
        text += "".join(ch for _y, _x, ch in row_sorted)

    if (len(rows) == 2 and len(rows[0]) == _MOTO_ROW1_COUNT
            and len(rows[1]) in _MOTO_ROW2_COUNTS):
        row_count = 2
    else:
        row_count = 1

    return text, min(confs), row_count


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
    # Class-agnostic NMS: YOLOv5's default NMS suppresses overlapping boxes
    # only within the same predicted class, so two near-identical boxes
    # classified as *different* characters (e.g. a spurious low-confidence
    # "1" at virtually the same position as a correct high-confidence "7")
    # both survive. Found via a real duplicate-character misread
    # (75H1357192 for GT 75H135792 - a genuine extra "1"); class-agnostic
    # NMS correctly keeps only the higher-confidence box regardless of
    # its predicted class, since two different characters can never
    # legitimately occupy the same physical position on a plate.
    model.agnostic = True
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

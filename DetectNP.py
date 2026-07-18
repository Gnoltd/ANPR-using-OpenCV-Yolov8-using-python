import os
import re
import time
import unicodedata
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ANPR_Yolo.env import *
from ANPR_Yolo.Load import _load_model, _load_ocr
from ANPR_Yolo.LPCharDetector import load_lp_char_detector, detect_plate_text

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

_PRINTED_MODEL_INFO = False

_lp_char_detector_model = None


def _get_lp_char_detector():
    global _lp_char_detector_model
    if _lp_char_detector_model is None:
        try:
            _lp_char_detector_model = load_lp_char_detector()
        except FileNotFoundError:
            _lp_char_detector_model = False  # sentinel: tried and unavailable
    return _lp_char_detector_model or None

# Compiled plate-format patterns (reused on every frame in video mode)
_RE_PLATE_FULL  = re.compile(r"^([0-9]{2})([A-Z])-?([0-9]{3})\.?([0-9]{2})$")
_RE_PLATE_COMP  = re.compile(r"^([0-9]{2})([A-Z])([0-9]{5})$")
# Motorbike-style series (letter+digit, e.g. B1/V7/L1/U2/S1/V5), dotted or 4-digit-no-dot body
_RE_PLATE_MOTO_DOT     = re.compile(r"^([0-9]{2})([A-Z][0-9])-?([0-9]{3})\.?([0-9]{2})$")
_RE_PLATE_MOTO_COMPACT = re.compile(r"^([0-9]{2})([A-Z][0-9])-?([0-9]{4})$")
# Special 2-letter series (e.g. LD = joint-venture vehicles), same body shape as car format
_RE_PLATE_SPECIAL_DOT     = re.compile(r"^([0-9]{2})([A-Z]{2})-?([0-9]{3})\.?([0-9]{2})$")
_RE_PLATE_SPECIAL_COMPACT = re.compile(r"^([0-9]{2})([A-Z]{2})([0-9]{5})$")
_RE_PLATE_CLEAN = re.compile(r"[^A-Z0-9\-\.]")
_RE_NON_ALNUM   = re.compile(r"[^A-Z0-9]")


def _format_full(m):
    return f"{m.group(1)}{m.group(2)}-{m.group(3)}.{m.group(4)}"


def _format_comp(m):
    s = m.group(3)
    return f"{m.group(1)}{m.group(2)}-{s[:3]}.{s[3:]}"


def _format_moto_compact(m):
    return f"{m.group(1)}{m.group(2)}-{m.group(3)}"


# Real car plates are physically always 1 row; real moto plates using the
# letter+digit series shape are physically always 2 rows. Grouped so a
# row_hint (from physical evidence - how many text rows a detector actually
# found) can reorder which group is tried first for a genuinely ambiguous
# compact string, without ever rejecting a match neither group can produce.
_CAR_LIKE_PATTERNS = (
    (_RE_PLATE_FULL, _format_full),
    (_RE_PLATE_COMP, _format_comp),
    (_RE_PLATE_SPECIAL_DOT, _format_full),
    (_RE_PLATE_SPECIAL_COMPACT, _format_comp),
)
_MOTO_LIKE_PATTERNS = (
    (_RE_PLATE_MOTO_DOT, _format_full),
    (_RE_PLATE_MOTO_COMPACT, _format_moto_compact),
)
# Default trial order (no row_hint): car-dotted, car-compact, moto-dotted,
# moto-compact, special-dotted, special-compact. Car patterns first so an
# ambiguous fully-compact string (no dash, e.g. could be read as
# 1-letter+5-digit OR 2-char-series+4-digit) resolves to the car
# interpretation, which is the more common format.
_PLATE_PATTERNS = (
    (_RE_PLATE_FULL, _format_full),
    (_RE_PLATE_COMP, _format_comp),
    (_RE_PLATE_MOTO_DOT, _format_full),
    (_RE_PLATE_MOTO_COMPACT, _format_moto_compact),
    (_RE_PLATE_SPECIAL_DOT, _format_full),
    (_RE_PLATE_SPECIAL_COMPACT, _format_comp),
)


def _match_plate_format(t, row_hint=None):
    """Try each known plate pattern against t; return the formatted plate
    string (e.g. "18A-123.45" or "29B1-256.62") or None if none match.

    row_hint, when given (1 or 2), is physical evidence of how many text
    rows the plate was actually read as - not a filter, just a trial-order
    preference: a 2-row hint tries moto-shaped patterns before car-shaped
    ones (real car plates are never 2 physical rows), which only changes
    the outcome for a string ambiguous between both shapes."""
    if row_hint == 2:
        order = _MOTO_LIKE_PATTERNS + _CAR_LIKE_PATTERNS
    else:
        order = _PLATE_PATTERNS
    for pat, formatter in order:
        m = pat.match(t)
        if m:
            return formatter(m)
    return None

def _valid_plate_bbox(x1, y1, x2, y2):
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    if w * h < MIN_AREA:
        return False
    if h == 0:
        return False
    ar = w / h
    if ar < MIN_AR or ar > MAX_AR:
        return False
    return True

def detect_fn(image_bgr):

    global _PRINTED_MODEL_INFO
    model = _load_model()
    device_arg = ANPR_DEVICE
    use_half = False
    try:
        import torch  # type: ignore
        if str(device_arg).lower() not in ("cpu", "mps") and torch.cuda.is_available():
            use_half = bool(ANPR_USE_HALF)
    except Exception:
        use_half = False

    results = model.predict(
        source=image_bgr, device=device_arg, imgsz=ANPR_IMGSZ, conf=CONF_THRES,iou=IOU_THRES,half=use_half, verbose=False
    )

    dets = []
    if not results:
        return dets

    res = results[0]
    names = getattr(res, "names", None) or getattr(model, "names", {}) or {}


    if not _PRINTED_MODEL_INFO:
        try:
            if isinstance(names, dict):
                cls_list = [names[i] for i in sorted(names.keys())]
            else:
                cls_list = list(names)
            print("[MODEL] Classes:", cls_list)
        except Exception:
            pass
        _PRINTED_MODEL_INFO = True

    h, w = image_bgr.shape[:2]
    for box in res.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0].detach().cpu().tolist())
        conf = float(box.conf[0]) if box.conf is not None else 0.0
        cls_id = int(box.cls[0]) if box.cls is not None else -1
        cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(x2, w - 1), min(y2, h - 1)


        if PLATE_CLASS_NAMES and (cls_name not in PLATE_CLASS_NAMES):
            continue


        if not _valid_plate_bbox(x1, y1, x2, y2):
            continue

        crop = image_bgr[y1:y2, x1:x2].copy()
        dets.append({
            "bbox": (x1, y1, x2, y2),
            "conf": conf,
            "cls": cls_id,
            "cls_name": cls_name,
            "crop": crop
        })
    return dets

def filter_text(text, strict=False, row_hint=None):
    if not text:
        return ""

    t = text.upper().strip()
    t = t.replace("—", "-").replace("–", "-").replace("_", "-").replace(" ", "")
    t = _RE_PLATE_CLEAN.sub("", t)

    matched = _match_plate_format(t, row_hint=row_hint)
    if matched:
        return matched

    if len(t) >= 3 and t[0:2].isdigit() and t[2].isdigit():
        digit_to_letter = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "4": "A"}
        c_fix = digit_to_letter.get(t[2])
        if c_fix:
            t2 = t[:2] + c_fix + t[3:]
            matched = _match_plate_format(t2, row_hint=row_hint)
            if matched:
                return matched

    if strict:
        return ""

    compact = _RE_NON_ALNUM.sub("", t)
    return compact if len(compact) >= 5 else ""


def select_plate_text(candidates):
    """Given OCR candidate strings in preference order, return the first
    that strictly matches a known plate format; if none do, fall back to
    the first that satisfies the loose alphanumeric-length heuristic;
    else "" if nothing qualifies."""
    for cand in candidates:
        ft = filter_text(cand, strict=True)
        if ft:
            return ft
    for cand in candidates:
        ft = filter_text(cand)
        if ft:
            return ft
    return ""

def load_registry(path: str = REGISTRY_CSV) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(columns=["plate","owner_name","phone","notes","plate_norm"])
        df.to_csv(path, index=False, encoding="utf-8-sig")

    for c in ["plate","owner_name","phone","notes"]:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].astype(str).fillna("").str.strip()

    df["plate_norm"] = df["plate"].apply(canonicalize_plate)
    return df

def save_registry(df: pd.DataFrame, path: str = REGISTRY_CSV):
    for c in ["plate","owner_name","phone","notes"]:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].astype(str).fillna("").str.strip()
    df["plate_norm"] = df["plate"].apply(canonicalize_plate)
    df.to_csv(path, index=False, encoding="utf-8-sig")

_CONFUSABLE_PAIRS = (("3", "8"), ("I", "V"))


def _confusable_variants(text: str) -> set:
    variants = set()
    chars = list(text)
    for i, ch in enumerate(chars):
        for a, b in _CONFUSABLE_PAIRS:
            replacement = None
            if ch == a:
                replacement = b
            elif ch == b:
                replacement = a
            if replacement is not None:
                swapped = chars.copy()
                swapped[i] = replacement
                variants.add("".join(swapped))
    variants.discard(text)
    return variants


def correct_against_registry(plate: str, registry_df: pd.DataFrame) -> str:
    """If `plate` has no registry match, try single-character confusable
    substitutions; if exactly one substitution matches a registered plate,
    return that corrected plate. Otherwise return `plate` unchanged."""
    key = canonicalize_plate(plate)
    if not key:
        return plate

    known_norms = set(registry_df["plate_norm"])
    if key in known_norms:
        return plate

    matched_norms = {v for v in _confusable_variants(key) if v in known_norms}
    if len(matched_norms) != 1:
        return plate

    matched_norm = next(iter(matched_norms))
    row = registry_df[registry_df["plate_norm"] == matched_norm].iloc[0]
    return row["plate"]


def lookup_owner(plate: str, path: str = REGISTRY_CSV):
    if not plate:
        return None
    df = load_registry(path)
    key = canonicalize_plate(plate)
    row = df[df["plate_norm"] == key]
    if row.empty:
        corrected = correct_against_registry(plate, df)
        if corrected != plate:
            key = canonicalize_plate(corrected)
            row = df[df["plate_norm"] == key]
    if not row.empty and row.iloc[0].get("owner_name","").strip():
        r = row.iloc[0].to_dict()
        return {
            "plate": r.get("plate",""),
            "owner_name": r.get("owner_name",""),
            "phone": r.get("phone",""),
            "notes": r.get("notes","")
        }
    return None
def _scale_to_target_height(crop_bgr, target_h=120):
    h, w = crop_bgr.shape[:2]
    if h == target_h:
        return crop_bgr
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    interp = cv2.INTER_CUBIC if h < target_h else cv2.INTER_LINEAR
    return cv2.resize(crop_bgr, (new_w, target_h), interpolation=interp)


def ocr_it(crop_bgr, joiner='-'):
    if crop_bgr is None or crop_bgr.size == 0:
        return "", {"all": [], "best_conf": 0.0}

    lp_detector_model = _get_lp_char_detector()
    if lp_detector_model is not None:
        detected_text, detector_conf, row_count = detect_plate_text(crop_bgr, lp_detector_model)
        if detected_text:
            formatted = filter_text(detected_text, strict=True, row_hint=row_count)
            if formatted:
                return formatted, {"all": [], "best_conf": detector_conf}

    # scale crop cho chiều cao ~120px để tăng độ nét (scale both up and down)
    crop_bgr = _scale_to_target_height(crop_bgr, target_h=120)

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    # EasyOCR trả về [(bbox(points4), text, conf), ...]
    reader = _load_ocr()
    result = reader.readtext(
        rgb,
        detail=1,
        paragraph=False,  # để tự sắp xếp thủ công theo y,x
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-."
    )

    # Thu thập & sắp xếp theo y->x
    items = []
    for (bbox, text, conf) in result:
        if not text or len(text.strip()) < 1:
            continue
        # y/x để sort (lấy min của 4 điểm)
        xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
        x_min, y_min = min(xs), min(ys)
        items.append((y_min, x_min, text.strip(), float(conf)))

    if not items:
        return "", {"all": [], "best_conf": 0.0}

    items.sort(key=lambda t: (t[0], t[1]))  # sort theo hàng rồi theo cột

    # Gom thành các "hàng" bằng ngưỡng theo chiều cao ảnh
    row_thresh = max(8, 0.18 * crop_bgr.shape[0])  # ngưỡng tách hàng
    rows = []  # mỗi row: {"y":..., "parts":[(x,text),...]}
    for y, x, text, conf in items:
        if not rows or abs(y - rows[-1]["y"]) > row_thresh:
            rows.append({"y": y, "parts": [(x, text)]})
        else:
            rows[-1]["parts"].append((x, text))

    # Nối các phần trong từng hàng (trái→phải), rồi ghép các hàng (trên→dưới)
    row_texts = []
    for row in rows:
        parts = [t for _, t in sorted(row["parts"])]
        row_texts.append("".join(parts))

    # Thử ghép kiểu TOP-BOTTOM trước (phù hợp biển VN 2 dòng)
    candidates = []
    if len(row_texts) >= 2:
        candidates.append(joiner.join([row_texts[0], row_texts[1]]))  # "36A-666.86"
    candidates.append(" ".join(row_texts))                             # "36A 666.86"
    # nếu chỉ có 1 dòng
    if len(row_texts) == 1:
        candidates.append(row_texts[0])

    # Chọn ứng viên khớp pattern VN (ưu tiên khớp định dạng chuẩn trước khi rơi vào fallback)
    ft = select_plate_text(candidates)
    if ft:
        all_items = [{"text": t, "conf": 0.0} for t in row_texts]
        return ft, {"all": all_items, "best_conf": max((c for *_ , c in items), default=0.0)}

    # Fallback: lấy dòng có conf cao nhất rồi lọc
    best = max(items, key=lambda z: z[3])[2]
    return filter_text(best), {"all": [{"text": t[2], "conf": t[3]} for t in items], "best_conf": max(t[3] for t in items)}


def norm_punct(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKC", s.strip())
    s = s.upper()

    # Gộp các biến thể dấu gạch/dấu chấm về 1 ký tự chuẩn
    s = s.replace("—","-").replace("–","-").replace("_","-")
    s = s.replace("•",".").replace("·",".").replace("∙",".")
    s = re.sub(r"\s+", " ", s)           # gộp khoảng trắng thừa
    s = s.replace(" -", "-").replace("- ", "-").replace(" .", ".")
    return s

def iou(boxA, boxB):
    (x1, y1, x2, y2) = boxA
    (X1, Y1, X2, Y2) = boxB
    interX1, interY1 = max(x1, X1), max(y1, Y1)
    interX2, interY2 = min(x2, X2), min(y2, Y2)
    iw, ih = max(0, interX2 - interX1), max(0, interY2 - interY1)
    inter = iw * ih
    areaA = max(0, x2 - x1) * max(0, y2 - y1)
    areaB = max(0, X2 - X1) * max(0, Y2 - Y1)
    union = areaA + areaB - inter + 1e-6
    return inter / union
def draw_contour_debug(image_bgr):

    dbg = image_bgr.copy()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(gray, 80, 180, apertureSize=3)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("ANPR Debug: edges", edges)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 300:
            continue
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 255), 1)
    cv2.putText(dbg, "NO YOLO DETECTION - showing contours", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("ANPR Debug: contours", dbg)


def show_contour_crops(image_path, min_area=300, save_dir=None):
    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray  = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(gray, 80, 180, apertureSize=3)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        crop = img[y:y+h, x:x+w].copy()
        rois.append((x, y, w, h, crop))
    rois.sort(key=lambda r: r[2]*r[3], reverse=True)

    if not rois:
        print("Cannot find contour crop.")
        cv2.imshow("Edges", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return []

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i, (x, y, w, h, crop) in enumerate(rois, 1):
        title = f"crop {i} ({w}x{h})"
        cv2.imshow(title, crop)
        if save_dir:
            cv2.imwrite(os.path.join(save_dir, f"crop_{i}_{x}_{y}.jpg"), crop)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,255), 2)

        k = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(title)
        if k in (27, ord('q')):
            break

    cv2.imshow("Contours on image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return [c for *_, c in rois]
def save_results(image_bgr, detections, ocr_texts, save_dir=SAVE_DIR, file_stem="frame", source_tag=""):
    os.makedirs(save_dir, exist_ok=True)
    out = image_bgr.copy()
    rows = []
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    for det, plate in zip(detections, ocr_texts):
        (x1, y1, x2, y2) = det["bbox"]
        conf = det["conf"]
        owner = lookup_owner(plate) if plate else None
        owner_name = owner["owner_name"] if owner else ""
        known = bool(owner and owner_name)
        conf_tag = f" [{conf:.0%}]"
        label = (f"{plate}{conf_tag} | {owner_name}" if (plate and known)
                 else (f"{plate}{conf_tag} | Owner : Unknown" if plate
                       else f"{det['cls_name']} {conf:.0%}"))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        rows.append({
            "timestamp": ts,
            "source": source_tag,
            "plate": plate,
            "owner_name": owner_name,
            "known_owner": int(known),
            "conf": conf,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })

    img_path = os.path.join(save_dir, f"{file_stem}_annotated.jpg")
    cv2.imwrite(img_path, out)

    csv_path = os.path.join(save_dir, "anpr_results.csv")
    df_new = pd.DataFrame(rows)
    if os.path.exists(csv_path) and len(rows) > 0:
        df_old = pd.read_csv(csv_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    if len(df) > 0:
        df.to_csv(csv_path, index=False)
    return img_path, csv_path
def canonicalize_plate(s: str) -> str:
    if not s:
        return ""
    t = unicodedata.normalize("NFKC", str(s)).upper().strip()

    t = t.replace("—","-").replace("–","-").replace("_","-")

    t = re.sub(r"[^A-Z0-9\-\.]", "", t)

    trans = str.maketrans({"O":"0","Q":"0","I":"1","L":"1","Z":"2","S":"5","B":"8"})
    t = t.translate(trans)


    if len(t) >= 3 and t[:2].isdigit() and t[2].isdigit():
        fix_map = {"0":"O","1":"I","2":"Z","5":"S","8":"B","4":"A"}
        t = t[:2] + fix_map.get(t[2], t[2]) + t[3:]


    matched = _match_plate_format(t)
    if matched:
        return matched

    compact = _RE_NON_ALNUM.sub("", t)
    return compact if len(compact) >= 5 else ""

def debug_owner_lookup(plate_raw, path: str = REGISTRY_CSV, topn=5):
    print("RAW       :", repr(plate_raw))
    key = canonicalize_plate(plate_raw)
    print("CANON     :", key)

    df = load_registry(path)
    print("CSV rows  :", len(df))

    hit = df[df["plate_norm"] == key]
    if not hit.empty:
        print("MATCH  :", hit.iloc[0][["plate","plate_norm","owner_name","phone","notes"]].to_dict())
        return

    prefix = key[:3]
    cand = df[df["plate_norm"].str.startswith(prefix, na=False)].copy()
    print(f"No exact match. Candidates with prefix '{prefix}':", len(cand))
    print(cand[["plate", "plate_norm", "owner_name"]].head(topn).to_string(index=False))

def repair_registry(path: str = REGISTRY_CSV):
    df = load_registry(path)       # đã canonicalize lại bên trong
    save_registry(df, path)        # ghi lại utf-8-sig
    print("Rewrote CSV with fresh plate_norm:", path)
    print(df.head(5).to_string(index=False))

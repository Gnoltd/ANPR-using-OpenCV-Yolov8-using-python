import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHAR_ALPHABET = "0123456789ABCDEFGHKLMNPSTUVXYZ"

MIN_CHAR_CONFIDENCE = 0.5

_CAR_ROW1_COUNT = 8
_MOTO_ROW1_COUNT = 4
_MOTO_ROW2_COUNTS = (4, 5)

# Connected-component filter thresholds (fractions of crop area/height/width).
# Tuned empirically against both synthetic rendered plates and 4 real VN
# plate crops (car x3, moto x1) from eval_images_vn/ - see Task 1 Step 6
# in docs/superpowers/plans/2026-07-18-character-classifier-ocr.md for the
# measured pass/fail results that produced these values.
_MIN_AREA_FRAC = 0.003
_MAX_AREA_FRAC = 0.35
_MIN_H_FRAC = 0.12
_MAX_H_FRAC = 0.98
_MAX_W_FRAC = 0.6
_ROW_GAP_FRAC = 0.15


def _binarize(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _character_components(gray):
    """Connected components on the binarized crop, filtered to plausible
    single-character shapes. Rejects components touching >=3 of the crop's
    4 edges (border-frame remnants) rather than relying on a fixed border
    trim, since real plate borders vary in thickness relative to crop size."""
    h, w = gray.shape[:2]
    binary = _binarize(gray)
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    crop_area = h * w

    comps = []
    for i in range(1, n):  # label 0 is background
        x, y, cw, ch, area = stats[i]
        if area < _MIN_AREA_FRAC * crop_area or area > _MAX_AREA_FRAC * crop_area:
            continue
        if ch < _MIN_H_FRAC * h or ch > _MAX_H_FRAC * h:
            continue
        if cw > _MAX_W_FRAC * w:
            continue
        touches = sum([x <= 1, y <= 1, (x + cw) >= w - 1, (y + ch) >= h - 1])
        if touches >= 3:
            continue
        comps.append((x, y, cw, ch, y + ch / 2.0))
    return comps


def _group_into_rows(comps, row_gap, crop_gray):
    comps = sorted(comps, key=lambda c: c[4])  # sort by vertical center
    rows = []
    for c in comps:
        if not rows or (c[4] - rows[-1][-1][4]) > row_gap:
            rows.append([c])
        else:
            rows[-1].append(c)

    result = []
    for row in rows:
        row_sorted = sorted(row, key=lambda c: c[0])  # left-to-right
        chars = [crop_gray[y:y + ch, x:x + cw] for (x, y, cw, ch, _cy) in row_sorted]
        result.append(chars)
    return result


def segment_plate_characters(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return []

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]

    comps = _character_components(gray)
    if not comps:
        return []

    rows = _group_into_rows(comps, row_gap=_ROW_GAP_FRAC * h, crop_gray=gray)

    counts = [len(r) for r in rows]
    if len(rows) == 1:
        if counts[0] != _CAR_ROW1_COUNT:
            return []
    elif len(rows) == 2:
        if counts[0] != _MOTO_ROW1_COUNT or counts[1] not in _MOTO_ROW2_COUNTS:
            return []
    else:
        return []

    return rows


def pad_to_square(img, pad_value=255):
    """Pads a grayscale image to a square (longer side wins) by adding
    equal border on the shorter dimension, centering the content. Real
    character crops from segment_plate_characters are tightly cropped to
    their ink (often narrow, e.g. a "1"), while synthetic training
    characters are rendered onto a padded square canvas - resizing a tight
    real crop straight to a square target distorts its aspect ratio in a
    way the classifier never saw during training. Padding to square first
    keeps real and synthetic preprocessing consistent."""
    h, w = img.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return img
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)


class CharClassifierCNN(nn.Module):
    def __init__(self, num_classes=len(CHAR_ALPHABET)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32 -> 16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16 -> 8
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def load_char_classifier(path="char_classifier.pt"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Character classifier weights not found: {path}\n"
            "Run train_char_classifier.py to generate them."
        )
    model = CharClassifierCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def classify_character(char_img, model):
    squared = pad_to_square(char_img)
    resized = cv2.resize(squared, (32, 32)) if squared.shape[:2] != (32, 32) else squared
    x = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(probs).item())
    return CHAR_ALPHABET[idx], float(probs[idx].item())


def classify_plate(crop_bgr, model, joiner='-'):
    rows = segment_plate_characters(crop_bgr)
    if not rows:
        return "", 0.0

    row_texts = []
    confidences = []
    for row in rows:
        chars = []
        for char_img in row:
            ch, conf = classify_character(char_img, model)
            chars.append(ch)
            confidences.append(conf)
        row_texts.append("".join(chars))

    if min(confidences) < MIN_CHAR_CONFIDENCE:
        return "", 0.0

    text = joiner.join(row_texts) if len(row_texts) > 1 else row_texts[0]
    return text, min(confidences)

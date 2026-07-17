import csv
import os
import random
import sys
import types

import cv2 as _cv2
import cv2
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from PIL import Image, ImageDraw, ImageFont

from PlateOCR import CHAR_ALPHABET, CharClassifierCNN, segment_plate_characters


def _register_anpr_yolo_from_here():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    if "ANPR_Yolo" not in sys.modules:
        pkg = types.ModuleType("ANPR_Yolo")
        pkg.__path__ = [here]
        pkg.__file__ = os.path.join(here, "__init__.py")
        pkg.__package__ = "ANPR_Yolo"
        sys.modules["ANPR_Yolo"] = pkg

# Bold sans-serif approximation of the FE-Schrift plate font (TCVN 4888:2019).
# Windows ships "arialbd.ttf" (Arial Bold) by default; if unavailable at
# runtime, PIL's built-in bitmap font is used as a last-resort fallback
# (still produces a valid, if lower-fidelity, training image).
_FALLBACK_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\arial.ttf",
]


def _load_font(size, font_path=None):
    candidates = [font_path] if font_path else _FALLBACK_FONT_CANDIDATES
    for path in candidates:
        if path:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _render_char_pil(char, size, font_path=None):
    render_size = size * 3
    img = Image.new("L", (render_size, render_size), color=255)
    draw = ImageDraw.Draw(img)
    font = _load_font(int(render_size * 0.8), font_path)
    bbox = draw.textbbox((0, 0), char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (render_size - tw) / 2 - bbox[0]
    y = (render_size - th) / 2 - bbox[1]
    draw.text((x, y), char, fill=0, font=font)
    return np.array(img)


def _augment(img, size):
    h, w = img.shape[:2]
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=255)

    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    jitter = w * 0.06
    dst = src + np.random.uniform(-jitter, jitter, src.shape).astype(np.float32)
    P = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, P, (w, h), borderValue=255)

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    if random.random() < 0.7:
        k = random.choice([1, 3])
        if k > 1:
            img = cv2.GaussianBlur(img, (k, k), 0)

    noise = np.random.normal(0, random.uniform(2, 12), img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    contrast = random.uniform(0.7, 1.3)
    brightness = random.uniform(-20, 20)
    img = np.clip(img.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)

    return img


def generate_synthetic_char(char, size=32, font_path=None):
    rendered = _render_char_pil(char, size, font_path)
    return _augment(rendered, size)


def generate_synthetic_dataset(n_per_class=300, size=32, font_path=None):
    images = []
    labels = []
    for label, char in enumerate(CHAR_ALPHABET):
        for _ in range(n_per_class):
            images.append(generate_synthetic_char(char, size=size, font_path=font_path))
            labels.append(label)
    return np.stack(images), np.array(labels, dtype=np.int64)


def _to_tensor_dataset(images, labels):
    x = torch.from_numpy(images).float().unsqueeze(1) / 255.0  # (N, 1, H, W), normalized
    y = torch.from_numpy(labels).long()
    return TensorDataset(x, y)


def train_on_synthetic(n_per_class=300, epochs=15, batch_size=64, size=32, out_path="char_classifier.pt"):
    images, labels = generate_synthetic_dataset(n_per_class=n_per_class, size=size)
    dataset = _to_tensor_dataset(images, labels)

    val_size = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = CharClassifierCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
        val_acc = correct / len(val_ds)
        print(f"epoch {epoch + 1}/{epochs}  train_loss={total_loss / train_size:.4f}  val_acc={val_acc:.1%}")

    torch.save(model.state_dict(), out_path)
    print(f"Saved {out_path}")
    return model


def _plate_to_chars(plate_gt):
    # Strip formatting punctuation to get the raw character sequence,
    # matching how anpr_eval.normalize_for_compare treats plates, but
    # keeping case/character identity rather than lowercasing.
    return [c for c in plate_gt.upper() if c not in "-. "]


def extract_real_char_dataset(gt_rows, load_crops_fn):
    """gt_rows: list of (image_filename, plate_gt) tuples, possibly several
    rows per image (multi-plate images).
    load_crops_fn: image_filename -> list of that image's detected plate
    crops (np.ndarray, BGR) - ALL detections, not just one, since an image
    with N ground-truth plates needs its N crops matched individually.
    Returns (images, labels) for every (crop, plate) pairing where the
    crop's segmentation result's total character count matches
    len(_plate_to_chars(plate_gt)). Each crop is consumed by at most one
    GT row (first match wins) so a multi-plate image's several GT rows
    don't all get paired against the same crop."""
    images = []
    labels = []
    crops_by_image = {}
    for image_filename, plate_gt in gt_rows:
        if image_filename not in crops_by_image:
            crops_by_image[image_filename] = list(load_crops_fn(image_filename) or [])
        remaining_crops = crops_by_image[image_filename]

        expected = _plate_to_chars(plate_gt)
        matched_index = None
        matched_flat_chars = None
        for idx, crop in enumerate(remaining_crops):
            rows = segment_plate_characters(crop)
            if not rows:
                continue
            flat_chars = [ch for row in rows for ch in row]
            if len(flat_chars) == len(expected):
                matched_index = idx
                matched_flat_chars = flat_chars
                break

        if matched_index is None:
            continue
        del remaining_crops[matched_index]

        for char_img, expected_char in zip(matched_flat_chars, expected):
            if expected_char not in CHAR_ALPHABET:
                continue
            images.append(_cv2.resize(char_img, (32, 32)))
            labels.append(CHAR_ALPHABET.index(expected_char))
    return images, labels


def load_gt_rows(gt_csv="gt_vn.csv"):
    with open(gt_csv, newline="", encoding="utf-8") as f:
        return [(row["image"], row["plate"]) for row in csv.DictReader(f)]


def fine_tune_on_real(model, images, labels, epochs=10, batch_size=16, out_path="char_classifier.pt"):
    if not images:
        print("No real character examples extracted; skipping fine-tune.")
        return model

    x = torch.from_numpy(np.stack(images)).float().unsqueeze(1) / 255.0
    y = torch.from_numpy(np.array(labels, dtype=np.int64))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # lower LR than pretraining
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f"fine-tune epoch {epoch + 1}/{epochs}  loss={total_loss / len(dataset):.4f}")

    model.eval()
    torch.save(model.state_dict(), out_path)
    print(f"Saved fine-tuned {out_path} ({len(images)} real character examples used)")
    return model


if __name__ == "__main__":
    model = train_on_synthetic()

    def _load_crops(image_filename):
        _register_anpr_yolo_from_here()
        from ANPR_Yolo.DetectNP import detect_fn
        img = _cv2.imread(os.path.join("eval_images_vn", image_filename))
        dets = sorted(detect_fn(img), key=lambda d: -d["conf"])
        return [d["crop"] for d in dets]

    gt_rows = load_gt_rows("gt_vn.csv")
    images, labels = extract_real_char_dataset(gt_rows, _load_crops)
    print(f"Extracted {len(images)} real labeled characters from eval_images_vn/")
    fine_tune_on_real(model, images, labels)

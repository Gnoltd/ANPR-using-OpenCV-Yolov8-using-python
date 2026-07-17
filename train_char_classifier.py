import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from PlateOCR import CHAR_ALPHABET

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

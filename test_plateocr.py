import os
import sys
import types


def _register_anpr_yolo_package():
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    if "ANPR_Yolo" not in sys.modules:
        pkg = types.ModuleType("ANPR_Yolo")
        pkg.__path__ = [here]
        pkg.__file__ = os.path.join(here, "__init__.py")
        pkg.__package__ = "ANPR_Yolo"
        sys.modules["ANPR_Yolo"] = pkg


_register_anpr_yolo_package()

import unittest
import numpy as np
import cv2

from PlateOCR import segment_plate_characters


def _render_row(text, char_w=40, char_h=60, gap=14, margin=15, font_scale=1.6, thickness=3):
    width = margin * 2 + len(text) * (char_w + gap) - gap
    height = margin * 2 + char_h
    img = np.full((height, width), 255, dtype=np.uint8)
    x = margin
    for ch in text:
        cv2.putText(img, ch, (x, margin + char_h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0,), thickness)
        x += char_w + gap
    return img


def _render_plate(rows_text, **kw):
    row_imgs = [_render_row(r, **kw) for r in rows_text]
    width = max(im.shape[1] for im in row_imgs)
    row_gap = 25
    total_h = sum(im.shape[0] for im in row_imgs) + row_gap * (len(row_imgs) - 1)
    canvas = np.full((total_h, width), 255, dtype=np.uint8)
    y = 0
    for im in row_imgs:
        canvas[y:y + im.shape[0], 0:im.shape[1]] = im
        y += im.shape[0] + row_gap
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


class SegmentPlateCharactersTests(unittest.TestCase):
    def test_car_plate_single_row_eight_chars(self):
        img = _render_plate(["18A12345"])
        rows = segment_plate_characters(img)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]), 8)

    def test_moto_plate_two_rows_dotted_body(self):
        img = _render_plate(["29B1", "25662"])
        rows = segment_plate_characters(img)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]), 4)
        self.assertEqual(len(rows[1]), 5)

    def test_moto_plate_two_rows_compact_body(self):
        img = _render_plate(["29U2", "7914"])
        rows = segment_plate_characters(img)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]), 4)
        self.assertEqual(len(rows[1]), 4)

    def test_implausible_char_count_returns_empty(self):
        img = _render_plate(["ABC"])
        self.assertEqual(segment_plate_characters(img), [])

    def test_empty_crop_returns_empty(self):
        self.assertEqual(segment_plate_characters(np.zeros((0, 0, 3), dtype=np.uint8)), [])

    def test_none_crop_returns_empty(self):
        self.assertEqual(segment_plate_characters(None), [])


from unittest.mock import MagicMock, patch
import torch

from PlateOCR import classify_character, classify_plate, CharClassifierCNN


class ClassifyCharacterTests(unittest.TestCase):
    def test_returns_highest_probability_character_and_confidence(self):
        model = MagicMock()
        logits = torch.zeros((1, 30))
        logits[0, 5] = 10.0  # index 5 -> CHAR_ALPHABET[5]
        model.return_value = logits
        char_img = np.zeros((32, 32), dtype=np.uint8)
        ch, conf = classify_character(char_img, model)
        from PlateOCR import CHAR_ALPHABET
        self.assertEqual(ch, CHAR_ALPHABET[5])
        self.assertGreater(conf, 0.9)


class ClassifyPlateTests(unittest.TestCase):
    def test_assembles_single_row_car_plate(self):
        fake_rows = [[np.zeros((32, 32), dtype=np.uint8) for _ in range(8)]]
        fake_chars = list("18A12345")
        with patch("PlateOCR.segment_plate_characters", return_value=fake_rows), \
             patch("PlateOCR.classify_character", side_effect=[(c, 0.95) for c in fake_chars]):
            text, conf = classify_plate(np.zeros((10, 10, 3), dtype=np.uint8), model=MagicMock())
        self.assertEqual(text, "18A12345")
        self.assertGreater(conf, 0.9)

    def test_assembles_two_row_moto_plate_with_joiner(self):
        fake_rows = [
            [np.zeros((32, 32), dtype=np.uint8) for _ in range(4)],
            [np.zeros((32, 32), dtype=np.uint8) for _ in range(5)],
        ]
        fake_chars = list("29B1") + list("25662")
        with patch("PlateOCR.segment_plate_characters", return_value=fake_rows), \
             patch("PlateOCR.classify_character", side_effect=[(c, 0.95) for c in fake_chars]):
            text, conf = classify_plate(np.zeros((10, 10, 3), dtype=np.uint8), model=MagicMock(), joiner="-")
        self.assertEqual(text, "29B1-25662")

    def test_returns_empty_on_segmentation_failure(self):
        with patch("PlateOCR.segment_plate_characters", return_value=[]):
            text, conf = classify_plate(np.zeros((10, 10, 3), dtype=np.uint8), model=MagicMock())
        self.assertEqual(text, "")
        self.assertEqual(conf, 0.0)

    def test_returns_empty_when_any_character_confidence_is_low(self):
        fake_rows = [[np.zeros((32, 32), dtype=np.uint8) for _ in range(8)]]
        confidences = [0.95] * 7 + [0.2]  # one low-confidence character
        with patch("PlateOCR.segment_plate_characters", return_value=fake_rows), \
             patch("PlateOCR.classify_character", side_effect=list(zip("18A12345", confidences))):
            text, conf = classify_plate(np.zeros((10, 10, 3), dtype=np.uint8), model=MagicMock())
        self.assertEqual(text, "")


if __name__ == "__main__":
    unittest.main()

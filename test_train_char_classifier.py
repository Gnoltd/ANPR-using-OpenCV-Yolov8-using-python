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

from PlateOCR import CHAR_ALPHABET, CharClassifierCNN
from train_char_classifier import generate_synthetic_char, generate_synthetic_dataset


class GenerateSyntheticCharTests(unittest.TestCase):
    def test_returns_requested_size(self):
        img = generate_synthetic_char("A", size=32)
        self.assertEqual(img.shape, (32, 32))

    def test_different_calls_are_not_identical(self):
        # Augmentation (rotation/noise/etc.) should vary between calls.
        img1 = generate_synthetic_char("5", size=32)
        img2 = generate_synthetic_char("5", size=32)
        self.assertFalse(np.array_equal(img1, img2))

    def test_produces_non_blank_image(self):
        img = generate_synthetic_char("B", size=32)
        self.assertGreater(img.std(), 1.0)


class GenerateSyntheticDatasetTests(unittest.TestCase):
    def test_dataset_shape_matches_alphabet_and_count(self):
        images, labels = generate_synthetic_dataset(n_per_class=5, size=32)
        expected_n = 5 * len(CHAR_ALPHABET)
        self.assertEqual(images.shape, (expected_n, 32, 32))
        self.assertEqual(labels.shape, (expected_n,))

    def test_labels_cover_full_alphabet_range(self):
        images, labels = generate_synthetic_dataset(n_per_class=5, size=32)
        self.assertEqual(set(labels.tolist()), set(range(len(CHAR_ALPHABET))))


import torch


class CharClassifierCNNTests(unittest.TestCase):
    def test_forward_pass_output_shape(self):
        model = CharClassifierCNN()
        model.eval()
        x = torch.zeros((4, 1, 32, 32), dtype=torch.float32)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(tuple(out.shape), (4, len(CHAR_ALPHABET)))


from unittest.mock import patch

from train_char_classifier import extract_real_char_dataset


class ExtractRealCharDatasetTests(unittest.TestCase):
    def test_matches_segmentation_to_gt_string_by_length(self):
        # A car-format GT plate "18A12345" (8 chars, ignoring formatting
        # punctuation) should pair with an 8-character single-row
        # segmentation result, one label per character in order.
        fake_rows = [[np.zeros((32, 32), dtype=np.uint8) for _ in range(8)]]
        with patch("train_char_classifier.segment_plate_characters", return_value=fake_rows):
            images, labels = extract_real_char_dataset(
                [("fake.jpg", "18A-123.45")],
                lambda path: [np.zeros((10, 10, 3), dtype=np.uint8)],
            )
        self.assertEqual(len(images), 8)
        expected = [CHAR_ALPHABET.index(c) for c in "18A12345"]
        self.assertEqual(labels, expected)

    def test_skips_plates_where_segmentation_length_mismatches_gt(self):
        fake_rows = [[np.zeros((32, 32), dtype=np.uint8) for _ in range(3)]]  # wrong count
        with patch("train_char_classifier.segment_plate_characters", return_value=fake_rows):
            images, labels = extract_real_char_dataset(
                [("fake.jpg", "18A-123.45")],
                lambda path: [np.zeros((10, 10, 3), dtype=np.uint8)],
            )
        self.assertEqual(images, [])
        self.assertEqual(labels, [])

    def test_multi_plate_image_matches_each_gt_row_to_a_different_crop(self):
        # Two GT plates for the same image, both 8 chars: each of the
        # image's two crops must be paired with a distinct GT row, not
        # both matched against the same first crop.
        crop_a = np.full((10, 10, 3), 1, dtype=np.uint8)
        crop_b = np.full((10, 10, 3), 2, dtype=np.uint8)
        fake_rows_a = [[np.full((32, 32), 1, dtype=np.uint8) for _ in range(8)]]
        fake_rows_b = [[np.full((32, 32), 2, dtype=np.uint8) for _ in range(8)]]

        def fake_segment(crop):
            if np.array_equal(crop, crop_a):
                return fake_rows_a
            if np.array_equal(crop, crop_b):
                return fake_rows_b
            return []

        with patch("train_char_classifier.segment_plate_characters", side_effect=fake_segment):
            images, labels = extract_real_char_dataset(
                [("multi.jpg", "18A-123.45"), ("multi.jpg", "30E-999.99")],
                lambda path: [crop_a, crop_b],
            )
        # 16 characters total (8 per plate), and each plate's characters
        # come from its own matched crop (all pixel value 1s then all 2s,
        # not an interleaved or duplicated single source).
        self.assertEqual(len(images), 16)
        expected_labels = [CHAR_ALPHABET.index(c) for c in "18A12345"] + \
                           [CHAR_ALPHABET.index(c) for c in "30E99999"]
        self.assertEqual(labels, expected_labels)


if __name__ == "__main__":
    unittest.main()

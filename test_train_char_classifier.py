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


if __name__ == "__main__":
    unittest.main()

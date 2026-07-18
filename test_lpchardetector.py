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

from LPCharDetector import order_detected_chars


class OrderDetectedCharsTests(unittest.TestCase):
    def test_single_row_sorted_left_to_right(self):
        # boxes given out of order; y-centers all close together (1 row)
        detections = [
            (70, 10, 90, 40, "A", 0.9),
            (10, 10, 30, 40, "1", 0.9),
            (40, 10, 60, 40, "8", 0.9),
        ]
        text, conf, row_count = order_detected_chars(detections, crop_h=50)
        self.assertEqual(text, "18A")
        self.assertAlmostEqual(conf, 0.9)
        self.assertEqual(row_count, 1)

    def test_two_rows_grouped_and_ordered_top_to_bottom(self):
        # crop_h=100, row_gap_frac=0.18 -> threshold=18. Row1 y-center=20,
        # row2 y-center=70 (gap=50, well over threshold - correctly split).
        # Boxes given in scrambled order to exercise the sort/group logic,
        # not just a pass-through of already-ordered input.
        detections = [
            (65, 60, 85, 80, "6", 0.96),   # row2, x_c=75
            (30, 10, 50, 30, "9", 0.95),   # row1, x_c=40
            (25, 60, 45, 80, "5", 0.95),   # row2, x_c=35
            (50, 10, 70, 30, "B", 0.93),   # row1, x_c=60
            (5, 60, 25, 80, "2", 0.95),    # row2, x_c=15
            (85, 60, 105, 80, "6", 0.95),  # row2, x_c=95
            (105, 60, 125, 80, "2", 0.94),  # row2, x_c=115
            (70, 10, 90, 30, "1", 0.91),   # row1, x_c=80
            (10, 10, 30, 30, "2", 0.95),   # row1, x_c=20
        ]
        text, conf, row_count = order_detected_chars(detections, crop_h=100)
        self.assertEqual(text, "29B125662")
        self.assertEqual(row_count, 2)

    def test_min_confidence_returned(self):
        detections = [
            (10, 10, 30, 40, "1", 0.9),
            (40, 10, 60, 40, "8", 0.3),
        ]
        text, conf, row_count = order_detected_chars(detections, crop_h=50)
        self.assertAlmostEqual(conf, 0.3)

    def test_empty_detections_returns_empty(self):
        text, conf, row_count = order_detected_chars([], crop_h=50)
        self.assertEqual(text, "")
        self.assertEqual(conf, 0.0)
        self.assertEqual(row_count, 0)


if __name__ == "__main__":
    unittest.main()

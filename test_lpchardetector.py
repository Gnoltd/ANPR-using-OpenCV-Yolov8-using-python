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

    def test_skewed_single_row_car_plate_not_split_into_two_rows(self):
        # Real detector output from eval_images_vn/vn-crowd1.png's
        # "75A-182.83" plate (car format, genuinely 1 physical row) that
        # was incorrectly reported as row_count=2 before this fix: the
        # crop is photographed at a slight angle, so "75A" sits a bit
        # higher than "18283" - a real vertical offset, but the resulting
        # clusters (3 chars, then 5 chars) don't match the fixed shape of
        # a real moto plate's two physical lines (always exactly 4 then
        # 4-5 chars: 2-digit province + 2-char series on line 1). A pixel-
        # geometry threshold (gap vs. character height) was tried first
        # and found unreliable: on real data this exact skewed-row case
        # (gap/height=1.067) had a LARGER ratio than a genuine 2-row moto
        # plate elsewhere in this eval set (gap/height=0.994) - the two
        # cases are not geometrically separable, only content-separable.
        detections = [
            # (x1, y1, x2, y2, char, conf) - x1/x2 centered on the real
            # detector's x-centers with an arbitrary uniform width; y1/y2
            # reconstructed from the real detector's y-center and height.
            (31.9, 9.6, 56.9, 34.6, "7", 0.9),
            (47.0, 11.2, 72.0, 35.4, "5", 0.9),
            (60.8, 11.9, 85.8, 37.4, "A", 0.9),
            (13.4, 35.8, 38.4, 59.2, "1", 0.9),
            (28.4, 36.9, 53.4, 60.2, "8", 0.9),
            (42.6, 38.1, 67.6, 61.4, "2", 0.9),
            (60.6, 39.4, 85.6, 62.8, "8", 0.9),
            (75.0, 40.5, 100.0, 63.6, "3", 0.9),
        ]
        text, conf, row_count = order_detected_chars(detections, crop_h=69)
        self.assertEqual(row_count, 1)
        self.assertEqual(text, "75A18283")

    def test_two_clusters_with_wrong_char_counts_not_trusted_as_moto_rows(self):
        # Real detector output from eval_images_vn/vn-bienso.jpg's
        # "30A-999.99" plate (car format, genuinely 1 row): clusters as
        # 3 chars ("30A") then 5 chars ("99999") - real moto plates are
        # always exactly 4 then 4-5, so a 3+5 split is content-invalid
        # and must not be trusted as row_count=2, even though the pixel
        # gap (12.7px on a 32px crop) exceeds the loose text-assembly
        # threshold.
        detections = [
            (10.2, 4.1, 22.2, 16.1, "3", 0.9),
            (17.3, 4.2, 29.3, 16.3, "0", 0.9),
            (23.8, 4.3, 35.8, 16.9, "A", 0.9),
            (1.7, 17.6, 13.7, 29.0, "9", 0.9),
            (8.6, 17.6, 20.6, 29.2, "9", 0.9),
            (15.2, 17.7, 27.2, 29.5, "9", 0.9),
            (23.8, 18.1, 35.8, 30.1, "9", 0.9),
            (30.8, 18.5, 42.8, 30.1, "9", 0.9),
        ]
        text, conf, row_count = order_detected_chars(detections, crop_h=32)
        self.assertEqual(row_count, 1)
        self.assertEqual(text, "30A99999")

    def test_genuine_moto_four_plus_four_split_still_trusted(self):
        # Real detector output from eval_images_vn/vn-119.jpg's
        # "29V5-2108" plate (genuine 2-row moto plate): clusters as
        # exactly 4 then 4 chars, matching the real moto layout shape -
        # must still be trusted as row_count=2 even though its pixel
        # gap/height ratio (0.994) is actually SMALLER than the false-
        # positive skew case above (1.067), which is why content shape,
        # not pixel geometry, is the deciding signal.
        detections = [
            (107.9, 8.9, 132.9, 56.9, "5", 0.9),
            (82.4, 9.8, 107.4, 57.8, "V", 0.9),
            (37.5, 13.7, 62.5, 60.9, "9", 0.9),
            (12.9, 15.2, 37.9, 62.3, "2", 0.9),
            (116.0, 62.4, 141.0, 109.7, "8", 0.9),
            (82.6, 63.3, 107.6, 110.6, "0", 0.9),
            (53.6, 64.4, 78.6, 112.2, "1", 0.9),
            (19.2, 65.9, 44.2, 114.1, "2", 0.9),
        ]
        text, conf, row_count = order_detected_chars(detections, crop_h=121)
        self.assertEqual(row_count, 2)
        self.assertEqual(text, "29V52108")

    def test_genuine_moto_narrow_gap_still_splits(self):
        # Real detector output from eval_images_vn/vn-117.jpg's
        # "29V7-376.94" plate (genuine 2-row moto plate): row1 y-centers
        # 27.3-44.2, row2 y-centers 62.4-84.7, gap=18.2px on a 116px crop
        # - just under the old 0.18*crop_h=20.9 threshold, so it was
        # never split into clusters at all and got flattened into one
        # scrambled x-sorted "row" (right characters, wrong order:
        # "32796V974" instead of "29V737694"). The clustering threshold
        # must be loose enough to catch this real case.
        detections = [
            (87.2, 8.6, 124.7, 45.9, "7", 0.9),
            (70.4, 13.0, 107.9, 48.7, "V", 0.9),
            (40.0, 22.9, 77.5, 56.9, "9", 0.9),
            (21.5, 26.7, 59.0, 61.8, "2", 0.9),
            (106.2, 44.7, 143.7, 80.1, "4", 0.9),
            (85.1, 50.3, 122.6, 84.8, "9", 0.9),
            (58.2, 57.4, 95.7, 91.7, "6", 0.9),
            (38.8, 62.2, 76.3, 97.9, "7", 0.9),
            (20.2, 68.2, 57.7, 101.3, "3", 0.9),
        ]
        text, conf, row_count = order_detected_chars(detections, crop_h=116)
        self.assertEqual(row_count, 2)
        self.assertEqual(text, "29V737694")


if __name__ == "__main__":
    unittest.main()

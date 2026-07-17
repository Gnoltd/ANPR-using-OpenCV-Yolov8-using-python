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
from DetectNP import filter_text, canonicalize_plate


class FilterTextCarFormatTests(unittest.TestCase):
    def test_dotted_car_plate_passthrough(self):
        self.assertEqual(filter_text("18A-123.45"), "18A-123.45")

    def test_compact_car_plate_gets_dash_and_dot(self):
        self.assertEqual(filter_text("18A12345"), "18A-123.45")


class FilterTextMotorbikeFormatTests(unittest.TestCase):
    def test_dotted_motorbike_plate_passthrough(self):
        self.assertEqual(filter_text("29B1-256.62"), "29B1-256.62")

    def test_compact_dotted_motorbike_plate_gets_formatted(self):
        self.assertEqual(filter_text("29B125662"), "29B1-256.62")

    def test_motorbike_plate_no_dot_four_digit_body(self):
        self.assertEqual(filter_text("29U2-7914"), "29U2-7914")

    def test_real_eval_sample_v7(self):
        self.assertEqual(filter_text("29V7-376.94"), "29V7-376.94")


class CanonicalizePlateMotorbikeFormatTests(unittest.TestCase):
    def test_dotted_motorbike_plate(self):
        self.assertEqual(canonicalize_plate("29B1-256.62"), "29B1-256.62")

    def test_compact_motorbike_plate_no_dot(self):
        self.assertEqual(canonicalize_plate("29U2-7914"), "29U2-7914")

    def test_car_plate_regression(self):
        self.assertEqual(canonicalize_plate("18A-123.45"), "18A-123.45")


if __name__ == "__main__":
    unittest.main()

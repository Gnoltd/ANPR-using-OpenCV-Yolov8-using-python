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
from DetectNP import filter_text, canonicalize_plate, select_plate_text


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

    def test_ambiguous_compact_prefers_car_format_over_moto(self):
        # "29B12345" is genuinely ambiguous: it matches both car-compact
        # (letter "B" + 5-digit body "12345") and moto-compact (series "B1"
        # + 4-digit body "2345"). Car format wins by pattern order in
        # _PLATE_PATTERNS (see DetectNP.py) since it's the more common
        # format — this is an accepted false-negative for real moto plates
        # in this exact shape (2-char series + exactly 4 digits, no dash).
        self.assertEqual(filter_text("29B12345"), "29B-123.45")


class CanonicalizePlateMotorbikeFormatTests(unittest.TestCase):
    def test_dotted_motorbike_plate(self):
        self.assertEqual(canonicalize_plate("29B1-256.62"), "29B1-256.62")

    def test_compact_motorbike_plate_no_dot(self):
        self.assertEqual(canonicalize_plate("29U2-7914"), "29U2-7914")

    def test_car_plate_regression(self):
        self.assertEqual(canonicalize_plate("18A-123.45"), "18A-123.45")


class FilterTextStrictModeTests(unittest.TestCase):
    def test_strict_true_returns_empty_for_unformatted_text(self):
        self.assertEqual(filter_text("RANDOMTEXT123", strict=True), "")

    def test_strict_true_still_matches_valid_car_format(self):
        self.assertEqual(filter_text("18A-123.45", strict=True), "18A-123.45")

    def test_strict_false_keeps_loose_fallback_behavior(self):
        self.assertEqual(filter_text("RANDOMTEXT123", strict=False), "RANDOMTEXT123")


class SelectPlateTextTests(unittest.TestCase):
    def test_strict_match_on_first_candidate_wins(self):
        candidates = ["18A-123.45", "18A 123.45 EXTRA"]
        self.assertEqual(select_plate_text(candidates), "18A-123.45")

    def test_strict_match_on_later_candidate_preferred_over_earlier_loose_match(self):
        candidates = ["10DTM5INWJRS6VEHICVFHP", "18A-123.45"]
        self.assertEqual(select_plate_text(candidates), "18A-123.45")

    def test_falls_back_to_loose_match_when_no_strict_match_exists(self):
        candidates = ["10DTM5INWJRS6VEHICVFHP", "SHORT"]
        self.assertEqual(select_plate_text(candidates), "10DTM5INWJRS6VEHICVFHP")

    def test_empty_string_when_nothing_qualifies(self):
        candidates = ["AB", "1"]
        self.assertEqual(select_plate_text(candidates), "")


if __name__ == "__main__":
    unittest.main()

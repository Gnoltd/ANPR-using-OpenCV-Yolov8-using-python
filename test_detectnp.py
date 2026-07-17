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
import csv
import tempfile
from unittest.mock import patch
import numpy as np
import cv2
import pandas as pd

from DetectNP import (
    filter_text, canonicalize_plate, select_plate_text,
    correct_against_registry, lookup_owner, _valid_plate_bbox,
    _scale_to_target_height, save_results,
)


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


class ValidPlateBboxTests(unittest.TestCase):
    def test_plate_shaped_box_is_valid(self):
        self.assertTrue(_valid_plate_bbox(0, 0, 100, 30))

    def test_tiny_box_below_min_area_is_invalid(self):
        self.assertFalse(_valid_plate_bbox(0, 0, 10, 10))

    def test_zero_height_box_is_invalid(self):
        self.assertFalse(_valid_plate_bbox(0, 0, 100, 0))

    def test_too_narrow_aspect_ratio_is_invalid(self):
        self.assertFalse(_valid_plate_bbox(0, 0, 20, 200))

    def test_too_wide_aspect_ratio_is_invalid(self):
        self.assertFalse(_valid_plate_bbox(0, 0, 500, 20))


class SaveResultsCropFileTests(unittest.TestCase):
    def test_does_not_write_individual_crop_files(self):
        with tempfile.TemporaryDirectory() as d:
            img = np.zeros((100, 200, 3), dtype=np.uint8)
            crop = np.zeros((30, 60, 3), dtype=np.uint8)
            detections = [{
                "bbox": (10, 10, 70, 40),
                "conf": 0.9,
                "cls": 0,
                "cls_name": "LicensePlate",
                "crop": crop,
            }]
            save_results(img, detections, [""], save_dir=d, file_stem="test", source_tag="test")
            files = os.listdir(d)
            self.assertNotIn("test_10_10.jpg", files)

    def test_still_writes_annotated_image_and_csv_log(self):
        with tempfile.TemporaryDirectory() as d:
            img = np.zeros((100, 200, 3), dtype=np.uint8)
            crop = np.zeros((30, 60, 3), dtype=np.uint8)
            detections = [{
                "bbox": (10, 10, 70, 40),
                "conf": 0.9,
                "cls": 0,
                "cls_name": "LicensePlate",
                "crop": crop,
            }]
            save_results(img, detections, [""], save_dir=d, file_stem="test", source_tag="test")
            files = os.listdir(d)
            self.assertIn("test_annotated.jpg", files)
            self.assertIn("anpr_results.csv", files)


class ScaleToTargetHeightTests(unittest.TestCase):
    def test_small_crop_is_upscaled_to_target_height(self):
        crop = np.zeros((28, 54, 3), dtype=np.uint8)
        out = _scale_to_target_height(crop, target_h=120)
        self.assertEqual(out.shape[0], 120)
        self.assertEqual(out.shape[1], round(54 * 120 / 28))

    def test_small_crop_upscale_uses_cubic_interpolation(self):
        crop = np.zeros((28, 54, 3), dtype=np.uint8)
        with patch("DetectNP.cv2.resize", wraps=cv2.resize) as spy:
            _scale_to_target_height(crop, target_h=120)
            _, kwargs = spy.call_args
            self.assertEqual(kwargs.get("interpolation"), cv2.INTER_CUBIC)

    def test_large_crop_is_downscaled_to_target_height(self):
        crop = np.zeros((200, 400, 3), dtype=np.uint8)
        out = _scale_to_target_height(crop, target_h=120)
        self.assertEqual(out.shape[0], 120)
        self.assertEqual(out.shape[1], round(400 * 120 / 200))

    def test_large_crop_downscale_uses_linear_interpolation(self):
        crop = np.zeros((200, 400, 3), dtype=np.uint8)
        with patch("DetectNP.cv2.resize", wraps=cv2.resize) as spy:
            _scale_to_target_height(crop, target_h=120)
            _, kwargs = spy.call_args
            self.assertEqual(kwargs.get("interpolation"), cv2.INTER_LINEAR)

    def test_crop_already_at_target_height_is_unchanged(self):
        crop = np.zeros((120, 300, 3), dtype=np.uint8)
        out = _scale_to_target_height(crop, target_h=120)
        self.assertIs(out, crop)


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


def _make_registry(rows):
    df = pd.DataFrame(rows, columns=["plate", "owner_name", "phone", "notes"])
    df["plate_norm"] = df["plate"].apply(canonicalize_plate)
    return df


class CorrectAgainstRegistryTests(unittest.TestCase):
    def test_exact_match_returns_plate_unchanged(self):
        registry = _make_registry([
            {"plate": "30A-339.18", "owner_name": "A", "phone": "", "notes": ""},
        ])
        self.assertEqual(correct_against_registry("30A-339.18", registry), "30A-339.18")

    def test_single_unambiguous_confusable_correction(self):
        registry = _make_registry([
            {"plate": "30A-339.18", "owner_name": "A", "phone": "", "notes": ""},
        ])
        self.assertEqual(correct_against_registry("80A-339.18", registry), "30A-339.18")

    def test_no_correction_when_no_registry_match_found(self):
        registry = _make_registry([
            {"plate": "30A-339.18", "owner_name": "A", "phone": "", "notes": ""},
        ])
        self.assertEqual(correct_against_registry("99Z-999.99", registry), "99Z-999.99")

    def test_no_correction_when_ambiguous(self):
        registry = _make_registry([
            {"plate": "30A-339.18", "owner_name": "A", "phone": "", "notes": ""},
            {"plate": "80A-339.13", "owner_name": "B", "phone": "", "notes": ""},
        ])
        self.assertEqual(correct_against_registry("80A-339.18", registry), "80A-339.18")

    def test_single_unambiguous_iv_correction(self):
        # I/V only does real work at the series-letter position: canonicalize_plate's
        # digit-at-series-position repair defaults an OCR'd "1" there to "I", so a
        # true "V" series plate needs this pair to resolve against the registry.
        registry = _make_registry([
            {"plate": "29V5-2108", "owner_name": "A", "phone": "", "notes": ""},
        ])
        self.assertEqual(correct_against_registry("29I5-2108", registry), "29V5-2108")


class LookupOwnerRegistryCorrectionTests(unittest.TestCase):
    def test_lookup_owner_applies_registry_correction(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "registry.csv")
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["plate", "owner_name", "phone", "notes"])
                writer.writerow(["30A-339.18", "Nguyen Van A", "0900000000", ""])
            result = lookup_owner("80A-339.18", path=path)
            self.assertIsNotNone(result)
            self.assertEqual(result["owner_name"], "Nguyen Van A")


if __name__ == "__main__":
    unittest.main()

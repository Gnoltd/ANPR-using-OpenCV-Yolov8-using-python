import csv
import os
import tempfile
import unittest

from anpr_eval import (
    normalize_for_compare,
    levenshtein,
    char_error_rate,
    plate_from_filename,
    build_ground_truth,
    evaluate,
    format_report,
    read_csv_rows,
    write_csv_rows,
)


class NormalizeForCompareTests(unittest.TestCase):
    def test_strips_punctuation_and_uppercases(self):
        self.assertEqual(normalize_for_compare("18a-123.45"), "18A12345")

    def test_empty_string(self):
        self.assertEqual(normalize_for_compare(""), "")

    def test_already_normalized(self):
        self.assertEqual(normalize_for_compare("36A12345"), "36A12345")


class LevenshteinTests(unittest.TestCase):
    def test_identical_strings(self):
        self.assertEqual(levenshtein("ABC", "ABC"), 0)

    def test_empty_vs_string(self):
        self.assertEqual(levenshtein("", "ABC"), 3)

    def test_single_substitution(self):
        self.assertEqual(levenshtein("ABC", "ABD"), 1)

    def test_classic_kitten_sitting(self):
        self.assertEqual(levenshtein("kitten", "sitting"), 3)


class CharErrorRateTests(unittest.TestCase):
    def test_exact_match_is_zero(self):
        self.assertEqual(char_error_rate("18A12345", "18A12345"), 0.0)

    def test_completely_wrong(self):
        self.assertAlmostEqual(char_error_rate("ABC", "XYZ"), 1.0)

    def test_empty_ref_and_hyp_is_zero(self):
        self.assertEqual(char_error_rate("", ""), 0.0)

    def test_empty_ref_nonempty_hyp_is_one(self):
        self.assertEqual(char_error_rate("", "ABC"), 1.0)

    def test_partial_error(self):
        # 1 substitution out of 8 reference chars
        self.assertAlmostEqual(char_error_rate("18A12345", "18A12341"), 1 / 8)


class PlateFromFilenameTests(unittest.TestCase):
    def test_plain_stem_is_plate(self):
        self.assertEqual(plate_from_filename("18A-123.45.jpg"), "18A-123.45")

    def test_disambiguation_suffix_stripped(self):
        self.assertEqual(plate_from_filename("18A-123.45__front.jpg"), "18A-123.45")

    def test_noplate_keyword_is_empty(self):
        self.assertEqual(plate_from_filename("noplate__01.jpg"), "")

    def test_background_keyword_is_empty(self):
        self.assertEqual(plate_from_filename("background.png"), "")

    def test_negative_keyword_is_case_insensitive(self):
        self.assertEqual(plate_from_filename("Negative__3.jpeg"), "")


class BuildGroundTruthTests(unittest.TestCase):
    def test_scans_image_folder(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("18A-123.45.jpg", "noplate__1.png", "not_an_image.txt"):
                open(os.path.join(d, name), "w").close()
            rows = build_ground_truth(d)
            by_image = {r["image"]: r["plate"] for r in rows}
            self.assertEqual(by_image["18A-123.45.jpg"], "18A-123.45")
            self.assertEqual(by_image["noplate__1.png"], "")
            self.assertNotIn("not_an_image.txt", by_image)


class EvaluateTests(unittest.TestCase):
    def test_perfect_predictions(self):
        gt = [{"image": "a.jpg", "plate": "18A-123.45"}]
        pred = [{"image": "a.jpg", "plate": "18A-123.45", "detected": "1"}]
        m = evaluate(gt, pred)
        self.assertEqual(m["detection_rate"], 1.0)
        self.assertEqual(m["recognition_accuracy"], 1.0)
        self.assertEqual(m["mean_cer"], 0.0)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["f1"], 1.0)

    def test_missed_detection_counts_as_fn(self):
        gt = [{"image": "a.jpg", "plate": "18A-123.45"}]
        pred = [{"image": "a.jpg", "plate": "", "detected": "0"}]
        m = evaluate(gt, pred)
        self.assertEqual(m["detection_rate"], 0.0)
        self.assertEqual(m["recognition_accuracy"], 0.0)
        self.assertEqual(m["mean_cer"], 1.0)
        self.assertEqual(m["recall"], 0.0)
        self.assertIsNone(m["precision"])  # no positive predictions at all

    def test_false_positive_on_negative_image(self):
        gt = [{"image": "bg.jpg", "plate": ""}]
        pred = [{"image": "bg.jpg", "plate": "99Z-999.99", "detected": "1"}]
        m = evaluate(gt, pred)
        self.assertEqual(m["precision"], 0.0)
        self.assertIsNone(m["recall"])  # no GT-positive images at all

    def test_true_negative_on_negative_image(self):
        gt = [{"image": "bg.jpg", "plate": ""}]
        pred = [{"image": "bg.jpg", "plate": "", "detected": "0"}]
        m = evaluate(gt, pred)
        self.assertIsNone(m["precision"])
        self.assertIsNone(m["recall"])

    def test_mixed_batch_precision_recall_f1(self):
        gt = [
            {"image": "a.jpg", "plate": "18A-123.45"},
            {"image": "b.jpg", "plate": "36A-999.99"},
            {"image": "bg.jpg", "plate": ""},
        ]
        pred = [
            {"image": "a.jpg", "plate": "18A-123.45", "detected": "1"},   # TP
            {"image": "b.jpg", "plate": "36A-111.11", "detected": "1"},   # wrong -> FN + FP
            {"image": "bg.jpg", "plate": "12B-345.67", "detected": "1"},  # FP
        ]
        m = evaluate(gt, pred)
        # TP=1 (a), FN=1 (b's real plate unmatched), FP=2 (b's wrong output + bg's spurious output)
        self.assertEqual(m["recognition_accuracy"], 1 / 2)
        self.assertAlmostEqual(m["precision"], 1 / 3)
        self.assertAlmostEqual(m["recall"], 1 / 2)
        self.assertAlmostEqual(m["f1"], 2 * (1 / 3) * (1 / 2) / ((1 / 3) + (1 / 2)))

    def test_missing_prediction_row_treated_as_no_detection(self):
        gt = [{"image": "a.jpg", "plate": "18A-123.45"}]
        m = evaluate(gt, [])
        self.assertEqual(m["detection_rate"], 0.0)
        self.assertEqual(m["recognition_accuracy"], 0.0)

    def test_multi_plate_image_partial_match(self):
        gt = [
            {"image": "multi.jpg", "plate": "29B1-256.62"},
            {"image": "multi.jpg", "plate": "18B2-547.79"},
        ]
        pred = [
            {"image": "multi.jpg", "plate": "29B1-256.62", "detected": "1"},  # matches plate 1
            {"image": "multi.jpg", "plate": "18B2-000.00", "detected": "1"},  # wrong reading of plate 2
        ]
        m = evaluate(gt, pred)
        self.assertEqual(m["n_gt_plates"], 2)
        self.assertEqual(m["tp"], 1)
        self.assertEqual(m["fn"], 1)
        self.assertEqual(m["fp"], 1)
        self.assertEqual(m["detection_rate"], 1.0)  # 2 boxes found for 2 GT plates
        self.assertEqual(m["recognition_accuracy"], 0.5)

    def test_extra_spurious_detection_on_positive_image_is_fp(self):
        gt = [{"image": "multi2.jpg", "plate": "29B1-256.62"}]
        pred = [
            {"image": "multi2.jpg", "plate": "29B1-256.62", "detected": "1"},  # TP
            {"image": "multi2.jpg", "plate": "99Z-999.99", "detected": "1"},   # spurious extra box
        ]
        m = evaluate(gt, pred)
        self.assertEqual(m["tp"], 1)
        self.assertEqual(m["fp"], 1)
        self.assertEqual(m["fn"], 0)
        self.assertEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["precision"], 0.5)


class ReportFormattingTests(unittest.TestCase):
    def test_report_contains_key_metrics(self):
        gt = [{"image": "a.jpg", "plate": "18A-123.45"}]
        pred = [{"image": "a.jpg", "plate": "18A-123.45", "detected": "1"}]
        m = evaluate(gt, pred)
        report = format_report(m)
        self.assertIn("Detection Rate", report)
        self.assertIn("Recognition Accuracy", report)
        self.assertIn("Mean CER", report)
        self.assertIn("Precision", report)
        self.assertIn("Recall", report)
        self.assertIn("F1", report)


class CsvRoundTripTests(unittest.TestCase):
    def test_write_then_read_round_trip(self):
        rows = [
            {"image": "a.jpg", "plate": "18A-123.45", "detected": "1",
             "det_conf": "0.9", "ocr_conf": "0.8"},
        ]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "preds.csv")
            write_csv_rows(path, rows, fieldnames=["image", "plate", "detected", "det_conf", "ocr_conf"])
            with open(path, newline="", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("18A-123.45", content)
            read_back = read_csv_rows(path)
            self.assertEqual(read_back, rows)


if __name__ == "__main__":
    unittest.main()

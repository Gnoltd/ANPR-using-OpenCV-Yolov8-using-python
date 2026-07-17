"""Evaluate ANPR detection/recognition accuracy against labeled images.

See the "Evaluation" section of README.md for the ground-truth filename
convention, the preds.csv/gt.csv schema, and the metric definitions this
module implements.
"""
import argparse
import csv
import os
import re
import sys

_RE_NON_ALNUM = re.compile(r"[^A-Z0-9]")
_NEGATIVE_KEYWORDS = {"noplate", "background", "negative", "none"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_for_compare(s):
    if not s:
        return ""
    return _RE_NON_ALNUM.sub("", s.upper())


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur_row[j] = min(
                prev_row[j] + 1,       # deletion
                cur_row[j - 1] + 1,    # insertion
                prev_row[j - 1] + cost,  # substitution
            )
        prev_row = cur_row
    return prev_row[-1]


def char_error_rate(ref, hyp):
    ref = ref or ""
    hyp = hyp or ""
    if not ref:
        return 1.0 if hyp else 0.0
    return levenshtein(ref, hyp) / len(ref)


def plate_from_filename(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    token = stem.split("__", 1)[0]
    if token.lower() in _NEGATIVE_KEYWORDS:
        return ""
    return token


def build_ground_truth(image_dir):
    rows = []
    for name in sorted(os.listdir(image_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext not in _IMAGE_EXTS:
            continue
        rows.append({"image": name, "plate": plate_from_filename(name)})
    return rows


def _truthy(v):
    return str(v).strip() in ("1", "true", "True", "yes")


def evaluate(gt_rows, pred_rows):
    gt_by_image = {r["image"]: r.get("plate", "") for r in gt_rows}
    pred_by_image = {
        r["image"]: {
            "plate": r.get("plate", ""),
            "detected": _truthy(r.get("detected", "0")),
        }
        for r in pred_rows
    }

    n_gt_plates = 0
    n_negatives = 0
    n_detected_of_gt = 0
    n_matches = 0
    cer_sum = 0.0
    tp = fp = fn = tn = 0

    for image, gt_plate in gt_by_image.items():
        pred = pred_by_image.get(image, {"plate": "", "detected": False})
        gt_norm = normalize_for_compare(gt_plate)
        pred_norm = normalize_for_compare(pred["plate"])
        is_match = bool(gt_norm) and gt_norm == pred_norm

        if gt_norm:
            n_gt_plates += 1
            if pred["detected"]:
                n_detected_of_gt += 1
            if is_match:
                n_matches += 1
                tp += 1
            else:
                fn += 1
            cer_sum += char_error_rate(gt_norm, pred_norm)
        else:
            n_negatives += 1
            if pred_norm:
                fp += 1
            else:
                tn += 1

    detection_rate = (n_detected_of_gt / n_gt_plates) if n_gt_plates else None
    recognition_accuracy = (n_matches / n_gt_plates) if n_gt_plates else None
    mean_cer = (cer_sum / n_gt_plates) if n_gt_plates else None

    precision = (tp / (tp + fp)) if (tp + fp) else None
    recall = (tp / (tp + fn)) if (tp + fn) else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "n_images": len(gt_by_image),
        "n_gt_plates": n_gt_plates,
        "n_negatives": n_negatives,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "detection_rate": detection_rate,
        "recognition_accuracy": recognition_accuracy,
        "mean_cer": mean_cer,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _fmt(v, pct=True):
    if v is None:
        return "n/a"
    return f"{v:.1%}" if pct else f"{v:.3f}"


def format_report(m):
    lines = [
        "ANPR Evaluation Report",
        "=======================",
        f"Images total:          {m['n_images']}",
        f"Images with GT plate:  {m['n_gt_plates']}",
        f"Negative images:       {m['n_negatives']}",
        "",
        f"Detection Rate:        {_fmt(m['detection_rate'])}",
        f"Recognition Accuracy:  {_fmt(m['recognition_accuracy'])}",
        f"Mean CER:              {_fmt(m['mean_cer'], pct=False)}",
        f"Precision:             {_fmt(m['precision'])}",
        f"Recall:                {_fmt(m['recall'])}",
        f"F1:                    {_fmt(m['f1'])}",
        "",
        f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}",
    ]
    return "\n".join(lines)


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _register_anpr_yolo_package():
    """Match Run.py: register this directory as the ANPR_Yolo package."""
    import types

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


def run_predictions(image_dir, out_csv):
    _register_anpr_yolo_package()
    from ANPR_Yolo.ANPR import run_image

    fieldnames = ["image", "detected", "plate", "det_conf", "ocr_conf"]
    rows = []
    for name in sorted(os.listdir(image_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext not in _IMAGE_EXTS:
            continue
        path = os.path.join(image_dir, name)
        result = run_image(path, show_window=False)

        plate, det_conf, ocr_conf = "", 0.0, 0.0
        detected = 1 if result["plates"] else 0
        if result["plates"]:
            best_i = max(
                range(len(result["plates"])),
                key=lambda i: result["yolo_confs"][i],
            )
            plate = result["plates"][best_i]
            det_conf = result["yolo_confs"][best_i]
            ocr_conf = result["ocr_confs"][best_i]

        rows.append({
            "image": name,
            "detected": detected,
            "plate": plate,
            "det_conf": f"{det_conf:.4f}",
            "ocr_conf": f"{ocr_conf:.4f}",
        })
        print(f"[PRED] {name}: detected={detected} plate={plate!r}")

    write_csv_rows(out_csv, rows, fieldnames)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_gt = sub.add_parser("build-gt", help="Derive ground truth from image filenames")
    p_gt.add_argument("--images", required=True)
    p_gt.add_argument("--out", default="gt.csv")

    p_pred = sub.add_parser("predict", help="Run the ANPR pipeline over a folder of images")
    p_pred.add_argument("--images", required=True)
    p_pred.add_argument("--out", default="preds.csv")

    p_eval = sub.add_parser("eval", help="Score predictions against ground truth")
    p_eval.add_argument("--preds", required=True)
    p_eval.add_argument("--gt", required=True)
    p_eval.add_argument("--report", default=None)

    p_run = sub.add_parser("run", help="build-gt + predict + eval in one shot")
    p_run.add_argument("--images", required=True)
    p_run.add_argument("--gt-out", default="gt.csv")
    p_run.add_argument("--preds-out", default="preds.csv")
    p_run.add_argument("--report", default=None)

    args = parser.parse_args(argv)

    if args.command == "build-gt":
        rows = build_ground_truth(args.images)
        write_csv_rows(args.out, rows, fieldnames=["image", "plate"])
        print(f"Wrote {len(rows)} ground-truth rows to {args.out}")

    elif args.command == "predict":
        rows = run_predictions(args.images, args.out)
        print(f"Wrote {len(rows)} prediction rows to {args.out}")

    elif args.command == "eval":
        gt_rows = read_csv_rows(args.gt)
        pred_rows = read_csv_rows(args.preds)
        metrics = evaluate(gt_rows, pred_rows)
        report = format_report(metrics)
        print(report)
        if args.report:
            with open(args.report, "w", encoding="utf-8") as f:
                f.write(report + "\n")

    elif args.command == "run":
        gt_rows = build_ground_truth(args.images)
        write_csv_rows(args.gt_out, gt_rows, fieldnames=["image", "plate"])
        print(f"Wrote {len(gt_rows)} ground-truth rows to {args.gt_out}")

        pred_rows = run_predictions(args.images, args.preds_out)
        print(f"Wrote {len(pred_rows)} prediction rows to {args.preds_out}")

        metrics = evaluate(gt_rows, pred_rows)
        report = format_report(metrics)
        print(report)
        if args.report:
            with open(args.report, "w", encoding="utf-8") as f:
                f.write(report + "\n")


if __name__ == "__main__":
    main()

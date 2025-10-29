from ANPR_Yolo.env import *
import os
import re
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import easyocr
from ANPR_Yolo.Load import _load_model, _load_ocr
from ANPR_Yolo.DetectNP import (
    detect_fn, filter_text, load_registry, save_registry, lookup_owner,
    ocr_it, norm_punct, iou, draw_contour_debug, show_contour_crops,
    save_results, canonicalize_plate, debug_owner_lookup, repair_registry
)
def run_image(path, show_window=True):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)

    dets = detect_fn(img)


    if len(dets) == 0 and show_window:
        draw_contour_debug(img)


    texts = []
    for d in dets:
        raw_txt, _ = ocr_it(d["crop"])
        texts.append(filter_text(raw_txt))

    save_path, csv_path = save_results(
        img, dets, texts,
        file_stem=os.path.splitext(os.path.basename(path))[0],
        source_tag="image"
    )

    if show_window:
        disp = cv2.imread(save_path) if len(dets) > 0 else img
        if len(dets) > 0:
            cv2.imshow("ANPR YOLO (image)", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    for p in texts:
        if p:
            info = lookup_owner(p)
            print(f"[IMAGE] {p} -> {info['owner_name'] if info else 'Unknown'}")

    return {"plates": texts, "csv": csv_path, "annotated_image": save_path}

def _set_cam_props(cap, width=640, height=360, fps=30):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)

def run_webcam(
    source=0,
    show_window=True,
    target_width=640,
    target_height=360,
    target_fps=30,
    max_frames=None,
    ocr_every_n=10,
    iou_thres_keep=0.5,
    frame_skip=0,
    debounce_frames=30,
    draw_thickness=2,
):

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    _set_cam_props(cap, target_width, target_height, target_fps)

    # trạng thái cho OCR tái sử dụng kết quả
    last_boxes = []              # [(bbox, text)]
    last_ocr_frame = -10**9
    seen_recent = {}             # plate -> last frame idx

    idx = 0
    fps_ema = None
    alpha = 0.25
    color = (0, 255, 0)
    draw_t = int(max(1, draw_thickness))
    t0 = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # ép kích thước khung hình
            if frame.shape[1] != target_width or frame.shape[0] != target_height:
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

            if frame_skip and (idx % (frame_skip + 1) != 0):
                idx += 1
                continue

            tic = time.perf_counter()
            dets = []
            texts = []
            new_last_boxes = []

            # --- Detect nếu có detect_fn ---
            if 'detect_fn' in globals() and callable(detect_fn):
                try:
                    dets = detect_fn(frame)
                except Exception:
                    dets = []

            # --- OCR nếu có ocr_it + filter_text ---
            if dets and ('ocr_it' in globals()) and ('filter_text' in globals()) \
                     and callable(ocr_it) and callable(filter_text):
                do_ocr = (idx - last_ocr_frame) >= ocr_every_n
                # hàm iou (nếu có) để tái sử dụng text
                _iou = globals().get('iou', None)

                for d in dets:
                    bb = d.get("bbox")
                    if not bb or len(bb) != 4:
                        texts.append("")
                        new_last_boxes.append(((0,0,0,0), ""))
                        continue

                    reused = None
                    if (not do_ocr) and last_boxes and callable(_iou):
                        for (bb_old, txt_old) in last_boxes:
                            if _iou(bb, bb_old) >= iou_thres_keep:
                                reused = txt_old
                                break

                    if (reused is not None) and (not do_ocr):
                        ft = reused
                    else:
                        crop = d.get("crop", None)
                        try:
                            raw_txt, _ = ocr_it(crop) if crop is not None else ("", None)
                        except Exception:
                            raw_txt = ""
                        ft = filter_text(raw_txt)

                    texts.append(ft)
                    new_last_boxes.append((bb, ft))

                if do_ocr:
                    last_ocr_frame = idx
                last_boxes = new_last_boxes
            else:
                last_boxes = []

            # --- Vẽ & hiển thị ---
            if show_window:
                if dets:
                    for d, plate in zip(dets, texts if texts else [""]*len(dets)):
                        x1, y1, x2, y2 = d["bbox"]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, draw_t)
                        lbl = ""
                        if plate:
                            info = None
                            if 'lookup_owner' in globals() and callable(lookup_owner):
                                try:
                                    info = lookup_owner(plate)
                                except Exception:
                                    info = None
                            lbl = f"{plate} | {info['owner_name']}" if (info and info.get("owner_name")) \
                                  else f"{plate} | Owner: Not Defined"
                        else:
                            lbl = d.get("cls_name", "plate")
                        cv2.putText(frame, lbl, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, draw_t)
                # FPS
                inst_fps = 1.0 / max(1e-6, time.perf_counter() - tic)
                fps_ema = inst_fps if fps_ema is None else (alpha * inst_fps + (1 - alpha) * fps_ema)
                cv2.putText(frame, f"FPS: {fps_ema:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, draw_t)

                cv2.imshow("Webcam/Video", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):  # ESC hoặc q
                    break
                if key == ord('s'):
                    import os
                    os.makedirs("../runs/anpr_yolo", exist_ok=True)
                    fn = f"runs/anpr_yolo/snapshot_{idx:06d}.jpg"
                    cv2.imwrite(fn, frame)
                    print(f"[SNAPSHOT] saved {fn}")

            # --- Log biển số mới (nếu có) ---
            if texts:
                for plate in texts:
                    if plate:
                        last_seen = seen_recent.get(plate, -10**9)
                        if (idx - last_seen) > debounce_frames:
                            owner = "Unknown"
                            if 'lookup_owner' in globals() and callable(lookup_owner):
                                try:
                                    info = lookup_owner(plate)
                                    if info and info.get("owner_name"):
                                        owner = info["owner_name"]
                                except Exception:
                                    pass
                            print(f"[LIVE] {plate} -> {owner}")
                            seen_recent[plate] = idx

            idx += 1
            if (max_frames is not None) and (idx >= max_frames):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


    print(f"Processed {idx} frames in {time.perf_counter() - t0:.2f}s")
    return True

def run_video(
    source=0,
    show_window=True,
    target_width=640,
    target_height=360,
    target_fps=30,
    max_frames=None,
    ocr_every_n=10,
    iou_thres_keep=0.5,
    frame_skip=0,
    debounce_frames=30,
    draw_thickness=2,
):

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    _set_cam_props(cap, target_width, target_height, target_fps)

    # trạng thái cho OCR tái sử dụng kết quả
    last_boxes = []              # [(bbox, text)]
    last_ocr_frame = -10**9
    seen_recent = {}             # plate -> last frame idx

    idx = 0
    fps_ema = None
    alpha = 0.25
    color = (0, 255, 0)
    draw_t = int(max(1, draw_thickness))
    t0 = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # ép kích thước khung hình
            if frame.shape[1] != target_width or frame.shape[0] != target_height:
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

            if frame_skip and (idx % (frame_skip + 1) != 0):
                idx += 1
                continue

            tic = time.perf_counter()
            dets = []
            texts = []
            new_last_boxes = []

            # --- Detect nếu có detect_fn ---
            if 'detect_fn' in globals() and callable(detect_fn):
                try:
                    dets = detect_fn(frame)
                except Exception:
                    dets = []

            # --- OCR nếu có ocr_it + filter_text ---
            if dets and ('ocr_it' in globals()) and ('filter_text' in globals()) \
                     and callable(ocr_it) and callable(filter_text):
                do_ocr = (idx - last_ocr_frame) >= ocr_every_n
                # hàm iou (nếu có) để tái sử dụng text
                _iou = globals().get('iou', None)

                for d in dets:
                    bb = d.get("bbox")
                    if not bb or len(bb) != 4:
                        texts.append("")
                        new_last_boxes.append(((0,0,0,0), ""))
                        continue

                    reused = None
                    if (not do_ocr) and last_boxes and callable(_iou):
                        for (bb_old, txt_old) in last_boxes:
                            if _iou(bb, bb_old) >= iou_thres_keep:
                                reused = txt_old
                                break

                    if (reused is not None) and (not do_ocr):
                        ft = reused
                    else:
                        crop = d.get("crop", None)
                        try:
                            raw_txt, _ = ocr_it(crop) if crop is not None else ("", None)
                        except Exception:
                            raw_txt = ""
                        ft = filter_text(raw_txt)

                    texts.append(ft)
                    new_last_boxes.append((bb, ft))

                if do_ocr:
                    last_ocr_frame = idx
                last_boxes = new_last_boxes
            else:
                last_boxes = []

            # --- Vẽ & hiển thị ---
            if show_window:
                if dets:
                    for d, plate in zip(dets, texts if texts else [""]*len(dets)):
                        x1, y1, x2, y2 = d["bbox"]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, draw_t)
                        lbl = ""
                        if plate:
                            info = None
                            if 'lookup_owner' in globals() and callable(lookup_owner):
                                try:
                                    info = lookup_owner(plate)
                                except Exception:
                                    info = None
                            lbl = f"{plate} | {info['owner_name']}" if (info and info.get("owner_name")) \
                                  else f"{plate} | Owner: Not Defined"
                        else:
                            lbl = d.get("cls_name", "plate")
                        cv2.putText(frame, lbl, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, draw_t)
                # FPS
                inst_fps = 1.0 / max(1e-6, time.perf_counter() - tic)
                fps_ema = inst_fps if fps_ema is None else (alpha * inst_fps + (1 - alpha) * fps_ema)
                cv2.putText(frame, f"FPS: {fps_ema:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, draw_t)

                cv2.imshow("Webcam/Video", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):  # ESC hoặc q
                    break
                if key == ord('s'):
                    import os
                    os.makedirs("../runs/anpr_yolo", exist_ok=True)
                    fn = f"runs/anpr_yolo/snapshot_{idx:06d}.jpg"
                    cv2.imwrite(fn, frame)
                    print(f"[SNAPSHOT] saved {fn}")

            # --- Log biển số mới (nếu có) ---
            if texts:
                for plate in texts:
                    if plate:
                        last_seen = seen_recent.get(plate, -10**9)
                        if (idx - last_seen) > debounce_frames:
                            owner = "Unknown"
                            if 'lookup_owner' in globals() and callable(lookup_owner):
                                try:
                                    info = lookup_owner(plate)
                                    if info and info.get("owner_name"):
                                        owner = info["owner_name"]
                                except Exception:
                                    pass
                            print(f"[LIVE] {plate} -> {owner}")
                            seen_recent[plate] = idx

            idx += 1
            if (max_frames is not None) and (idx >= max_frames):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


    print(f"Processed {idx} frames in {time.perf_counter() - t0:.2f}s")
    return True
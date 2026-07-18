# LP character detector: source and safe-extraction procedure

## Source

`lp_char_detector_state_dict.pt` / `lp_char_detector_arch.json` (both
untracked, same convention as `best.pt`) are derived from `LP_ocr.pt`, a
pretrained YOLOv5 model found in `trungdinh22/License-Plate-Recognition`
(GitHub). Unlike our own from-scratch character classifier (trained on 85
real examples across 30 classes, which never produced a usable real-world
result), this model detects and classifies each plate character directly
in one pass — no separate segmentation step — and was verified against 5
real crops from `eval_images_vn/` with **5/5 exact full-plate matches**
after fixing a row-ordering bug (see commit history around this date).

## Why extraction was needed, not a direct load

`LP_ocr.pt` is a full pickled YOLOv5 model object (not just a state_dict).
Loading it via `torch.load` requires `weights_only=False`, which permits
arbitrary code execution during deserialization if the file were
malicious — acceptable for a one-time, explicitly-authorized inspection,
but not something the checked-in runtime code (`DetectNP.py`, `Run.py`,
etc.) should do on every process start indefinitely.

## The extraction (already done; not part of the repo's automated code)

Run once, interactively, with explicit trust in the source file:

```python
import sys, os, json
import yolov5
sys.path.insert(0, os.path.dirname(yolov5.__file__))
import torch

_orig_load = torch.load
def _patched_load(*a, **kw):
    kw["weights_only"] = False  # explicit, one-time, authorized
    return _orig_load(*a, **kw)
torch.load = _patched_load
wrapped = yolov5.load("LP_ocr.pt")
torch.load = _orig_load

core = wrapped.model.model if hasattr(wrapped.model, "model") else wrapped.model
torch.save(core.state_dict(), "lp_char_detector_state_dict.pt")
with open("lp_char_detector_arch.json", "w") as f:
    json.dump({"yaml": core.yaml, "names": core.names}, f)
```

This produces two artifacts that are both safe to load repeatedly:
- `lp_char_detector_state_dict.pt`: pure tensors, loadable with PyTorch's
  default `weights_only=True` (no object deserialization at all).
- `lp_char_detector_arch.json`: plain JSON (architecture config + class
  names), not a pickle at all.

`LPCharDetector.load_lp_char_detector()` (the function the app actually
calls) only ever consumes these two safe artifacts, reconstructing the
model architecture via pure Python code (`models.yolo.DetectionModel(cfg=...)`
from the trusted `yolov5` package) and loading only tensor weights into it.
It never performs the unsafe load itself.

## If these artifacts are lost or need regenerating

Re-run the extraction script above against a copy of `LP_ocr.pt` (get a
fresh copy from the source repo, or ask whoever has the original). This is
a deliberate one-time manual step, not something `Run.py` or any other
part of the app does automatically — regenerating it always requires a
human to re-confirm trust in the source file.

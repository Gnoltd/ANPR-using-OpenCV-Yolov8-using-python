# OCR decoding-parameter experiment results

Tested EasyOCR `readtext()` parameter overrides against `eval_images_vn/` (10 images, 22 GT plates) to see whether decoder/contrast/magnification tuning alone (no retraining, no new dependency) moves recognition accuracy meaningfully. No production code (`DetectNP.py`) was changed by this experiment - overrides were injected via monkeypatch only.

| Config | Overrides | Recognition Accuracy | Mean CER | FP |
|---|---|---|---|---|
| baseline | `{}` | 31.8% | 0.227 | 14 |
| beamsearch | `{'decoder': 'beamsearch'}` | 31.8% | 0.227 | 14 |
| more_contrast | `{'contrast_ths': 0.05, 'adjust_contrast': 0.7}` | 31.8% | 0.227 | 14 |
| more_mag | `{'mag_ratio': 1.5}` | 18.2% | 0.225 | 17 |
| combined | `{'decoder': 'beamsearch', 'contrast_ths': 0.05, 'adjust_contrast': 0.7, 'mag_ratio': 1.5}` | 22.7% | 0.215 | 16 |

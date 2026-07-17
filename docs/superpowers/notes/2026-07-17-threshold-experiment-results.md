# Detection-threshold experiment results

Run against `eval_images_vn/` (real VN photos, 11 plates, includes 3 crowded multi-plate scenes) and `eval_images/` (mixed US/VN sanity check) to see whether lowering CONF_THRES or raising ANPR_IMGSZ recovers missed detections in crowded scenes without a material false-positive cost. No env.py defaults were changed by this experiment.

| Config | CONF_THRES | ANPR_IMGSZ | Image set | Detection Rate | FP | Recognition Accuracy |
|---|---|---|---|---|---|---|
| baseline | 0.25 | 640 | eval_images_vn | 72.7% | 7 | 9.1% |
| baseline | 0.25 | 640 | eval_images | 88.9% | 16 | 25.9% |
| lower_conf | 0.15 | 640 | eval_images_vn | 81.8% | 8 | 9.1% |
| lower_conf | 0.15 | 640 | eval_images | 88.9% | 16 | 25.9% |
| larger_imgsz | 0.25 | 960 | eval_images_vn | 100.0% | 9 | 18.2% |
| larger_imgsz | 0.25 | 960 | eval_images | 92.6% | 18 | 25.9% |
| lower_conf_larger_imgsz | 0.15 | 960 | eval_images_vn | 100.0% | 9 | 18.2% |
| lower_conf_larger_imgsz | 0.15 | 960 | eval_images | 92.6% | 18 | 25.9% |

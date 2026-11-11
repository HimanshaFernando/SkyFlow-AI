# SkyFlow AI

Drone-based vehicle detection and traffic analysis project.

## Current status

Established a YOLO11m detection baseline using all 548 images in the
VisDrone2019-DET validation set. Tracking and traffic counting are
planned next.

## Detection baseline

- Model: YOLO11m
- Image size: 1280
- Confidence threshold: 0.25
- Device: CPU
- Images evaluated: 548

| Metric | Result |
|---|---:|
| Ground-truth vehicles | 17,040 |
| Correct detections | 9,740 |
| Missed vehicles | 7,300 |
| Unmatched detections | 1,333 |
| Precision | 88.0% |
| Recall | 57.2% |
| F1 score | 69.3% |
| Overall processing rate | Approximately 0.93 images/second |

### Evaluation scope

These are custom diagnostic results, not official VisDrone mAP.
Cars, vans, trucks and buses are grouped into one vehicle category.
Predictions are matched one-to-one in descending confidence order
at IoU >= 0.50.

Ground-truth entries marked ignored are excluded. Predictions with
more than 50% of their area inside an individual ignored region are
excluded before matching; 103 predictions were excluded.

Vehicle classification accuracy is not measured by this grouping.

### Limitations

Small-vehicle recall was 35.7%, compared with 75.3% for medium and
92.6% for large vehicles. Size groups use original-image box areas:
below 32 squared pixels, 32 squared to below 96 squared pixels,
and at least 96 squared pixels.

The run took approximately 589 seconds, including prediction-call
overhead and label saving. There was no separate warm-up.
This is an offline processing baseline, not demonstrated real-time
video performance.

## Next steps

- Preserve a reproducible evaluation script.
- Test tracking stability on a short drone traffic video.
- Evaluate vehicle counting after checking track quality.

## Data and model files

Datasets, videos, downloaded model weights and generated runs are
excluded from this repository. Third-party data and models remain
subject to their respective terms.

"""Run YOLO predictions individually on the VisDrone validation images."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="VisDrone2019-DET-val/images")
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    images = sorted(Path(args.images).glob("*.jpg"))
    if not images:
        raise SystemExit(f"No JPG images found in {args.images}")
    if args.limit < 0:
        raise SystemExit("--limit must be zero or positive")
    if args.limit:
        images = images[:args.limit]

    name = "visdrone_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = Path("runs") / name
    out.mkdir(parents=True, exist_ok=False)
    (out / "labels").mkdir()

    model = YOLO(args.model)
    started = perf_counter()

    with (out / "timing.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "width", "height", "detections",
            "inference_ms", "call_seconds"
        ])

        for i, path in enumerate(images, 1):
            tick = perf_counter()
            result = model.predict(
                source=str(path),
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                save=False,
                save_txt=True,
                save_conf=True,
                project="runs",
                name=name,
                exist_ok=True,
                verbose=False,
            )[0]

            elapsed = perf_counter() - tick
            height, width = result.orig_shape
            (out / "labels" / f"{path.stem}.txt").touch(exist_ok=True)
            writer.writerow([
                path.name, width, height, len(result.boxes),
                result.speed["inference"], elapsed
            ])
            f.flush()
            print(
                f"[{i}/{len(images)}] {path.name}: "
                f"{len(result.boxes)} detections, {elapsed:.2f}s",
                flush=True,
            )

    config = vars(args).copy()
    config["image_count"] = len(images)
    config["total_seconds"] = perf_counter() - started
    config["timing_note"] = (
        "No separate warm-up; first call includes startup overhead. "
        "Total excludes model loading."
    )
    (out / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"Finished. Results: {out.resolve()}")


if __name__ == "__main__":
    main()

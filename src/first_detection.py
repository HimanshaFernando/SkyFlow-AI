from pathlib import Path

from ultralytics import YOLO


def main():
    # Resolve paths from the project folder.
    project_root = Path(__file__).resolve().parents[1]
    video_path = project_root / "data" / "samples" / "traffic.mp4"
    model_path = project_root / "yolo26n.pt"

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))

    # Select vehicle classes by name from the loaded model.
    vehicle_names = {"car", "motorcycle", "bus", "truck"}
    vehicle_classes = [
        class_id
        for class_id, name in model.names.items()
        if name in vehicle_names
    ]

    if not vehicle_classes:
        raise ValueError("The model does not contain the expected vehicle classes.")

    frames = 0
    frame_detections = 0
    frames_with_detections = 0
    observed_track_ids = set()
    output_directory = None

    results = model.track(
        source=str(video_path),
        tracker="botsort.yaml",
        stream=True,
        save=True,
        project=str(project_root / "runs"),
        name="vehicle_test",
        exist_ok=False,
        conf=0.25,
        imgsz=1280,
        classes=vehicle_classes,
        show_labels=True,
        show_conf=True,
        line_width=2,
        verbose=False,
    )

    for result in results:
        frames += 1
        output_directory = result.save_dir

        if result.boxes is not None:
            detections = len(result.boxes)
            frame_detections += detections
            frames_with_detections += int(detections > 0)

            if result.boxes.id is not None:
                observed_track_ids.update(
                    result.boxes.id.int().cpu().tolist()
                )

        if frames % 30 == 0:
            print(
                f"Processed {frames} frames | "
                f"Vehicle detections: {frame_detections}"
            )

    if frames == 0:
        raise RuntimeError("No video frames were processed.")

    print("\nProcessing complete")
    print(f"Frames processed: {frames}")
    print(f"Frames containing detections: {frames_with_detections}")
    print(f"Total vehicle detections across frames: {frame_detections}")
    print(f"Distinct tracker IDs observed: {len(observed_track_ids)}")
    print(f"Output folder: {output_directory}")
    print("Tracker IDs are diagnostic; ID changes can inflate vehicle counts.")


if __name__ == "__main__":
    main()
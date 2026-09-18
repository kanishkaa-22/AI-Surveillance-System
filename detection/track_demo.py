"""
 Detection + Tracking core, GPU, full frame rate (no skipping).

For each video in data/detection_tracking_clips/:
  - runs YOLO + ByteTrack on GPU
  - writes an annotated output video (boxes + track IDs) to data/track_outputs/
    OURSELVES via cv2.VideoWriter — not via Ultralytics' internal save=True,
    which can silently skip writing when consumed through the stream=True
    generator depending on the installed version. Writing it explicitly
    guarantees the file exists and lands exactly where we expect.
  - saves a CSV of every track's position per frame to data/track_outputs/
    (this CSV is what Step 7's zone logic reads later — don't skip it)
"""

import os
os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] = "1"  # see detection/benchmark.py
import csv
import cv2
from ultralytics import YOLO

# ---- Config ----
MODEL_PATH = "yolo11s.pt"     # updated to match backend/config.py's final deployed model — this file has its own separate MODEL_PATH (not connected to backend/config.py) since it's a standalone Week 1 demo script, not part of the main pipeline     # swap to yolo11s.pt / yolo11m.pt / yolo26n.pt as you benchmark
CLIPS_DIR = "data/detection_tracking_clips"
TRACK_OUTPUT_DIR = "data/track_outputs"
DEVICE = 0                     # GPU 0. Use "cpu" to force CPU for comparison.

os.makedirs(TRACK_OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)


def run_tracking(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\nTracking: {video_name}")

    # --- Get source video properties so the output writer matches exactly ---
    src_cap = cv2.VideoCapture(video_path)
    fps = src_cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_cap.release()

    # --- Set up the annotated video writer ---
    video_out_path = os.path.join(TRACK_OUTPUT_DIR, f"{video_name}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

    # --- Set up the CSV writer ---
    csv_path = os.path.join(TRACK_OUTPUT_DIR, f"{video_name}_tracks.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "cx", "cy", "conf"])

    # stream=True lets us iterate frame-by-frame instead of loading everything at once
    results = model.track(
        source=video_path,
        classes=[0],                 # class 0 = person only
        tracker="bytetrack.yaml",
        persist=True,
        device=DEVICE,
        save=False,                  # we write the video ourselves below instead
        stream=True,
        verbose=False,
    )

    frame_idx = 0
    for r in results:
        # r.plot() returns the frame with boxes/IDs/labels drawn — this is the
        # exact same annotated image Ultralytics would have saved internally.
        annotated_frame = r.plot()
        video_writer.write(annotated_frame)

        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            for box, tid, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                writer.writerow([frame_idx, tid, x1, y1, x2, y2, cx, cy, conf])

        frame_idx += 1

    csv_file.close()
    video_writer.release()

    print(f"  {frame_idx} frames processed.")
    print(f"  Annotated video: {video_out_path}")
    print(f"  Tracks CSV:      {csv_path}")


def main():
    if not os.path.isdir(CLIPS_DIR):
        print(f"ERROR: '{CLIPS_DIR}' does not exist. "
              f"Run this script from the repo root (AI-Surveillance-System/).")
        return

    videos = [f for f in os.listdir(CLIPS_DIR) if f.lower().endswith((".mp4", ".mov"))]
    if not videos:
        print(f"No videos found in {CLIPS_DIR}")
        return

    for video_file in videos:
        run_tracking(os.path.join(CLIPS_DIR, video_file))

    print(f"\nDone. Annotated videos and CSVs are both under {TRACK_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
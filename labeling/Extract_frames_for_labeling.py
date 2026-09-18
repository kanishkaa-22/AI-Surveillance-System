"""
Extracts a handful of evenly-spaced still frames from a video, for manual
ground-truth labeling (Week 4 evaluation).

Run from repo root:
    python backend/utils/extract_frames_for_labeling.py data/detection_tracking_clips/crowd_01.mp4 10

The second argument is how many frames to pull (default 10). Frames are
saved as plain .jpg images — these are what you send to your teammate
(WhatsApp/email/drive), no video or code needed on their end.
"""

import os
import sys
import cv2

OUTPUT_DIR = "data/ground_truth/images"


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_frames_for_labeling.py <video_path> [num_frames]")
        return

    video_path = sys.argv[1]
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clip_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Could not read frame count from this video.")
        return

    step = max(total_frames // num_frames, 1)
    saved = 0

    for i in range(0, total_frames, step):
        if saved >= num_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        out_name = f"{clip_name}_frame{i}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), frame)
        print(f"Saved {out_name}")
        saved += 1

    cap.release()
    print(f"\n{saved} frames saved to {OUTPUT_DIR}/")
    print("Send this whole folder to whoever is labeling — images only, no code needed.")


if __name__ == "__main__":
    main()
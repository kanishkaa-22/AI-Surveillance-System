"""
Enrollment pipeline: video -> extracted face frames -> embeddings.

Folder expected (relative to repo root — run this script from
AI-Surveillance-System/, not from inside recognition/):
    data/enrollment_videos/<person_name>.mp4   (your 15-sec rotation clips)

Produces:
    recognition/enrolled_faces/<person_name>/frame_0.jpg ... (kept face crops, for manual review)
    recognition/embeddings/<person_name>.npy             (Nx512 array — one row per kept frame)

These .npy files are what Step 6 (enrollment_and_matching.py) loads into
SQLite as each person's stored embeddings.
"""

import os
import cv2
import numpy as np
import torch  # noqa: F401 — see recognition/face_test.py for why this must come
              # before the insightface import
from insightface.app import FaceAnalysis

# ---- Config ----
VIDEOS_DIR = "data/enrollment_videos"
FACES_OUT_DIR = "recognition/enrolled_faces"
EMBEDDINGS_OUT_DIR = "recognition/embeddings"

FRAME_SAMPLE_EVERY_N_SEC = 0.5   # how often to pull a candidate frame from the video
MIN_DET_SCORE = 0.65             # discard low-confidence face detections
MAX_KEPT_FRAMES = 8              # cap how many good frames we keep per person

os.makedirs(FACES_OUT_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_OUT_DIR, exist_ok=True)

# ---- Load InsightFace once (GPU) ----
face_app = FaceAnalysis(name="buffalo_l")  # detection + recognition bundle
face_app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 -> GPU 0; use -1 for CPU


def extract_candidate_frames(video_path, sample_every_sec=FRAME_SAMPLE_EVERY_N_SEC):
    """Pull evenly-spaced frames from the video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(int(fps * sample_every_sec), 1)

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frames.append(frame)
        frame_idx += 1
    cap.release()
    return frames


def process_person(video_path, person_name):
    print(f"\nProcessing {person_name} ...")
    candidate_frames = extract_candidate_frames(video_path)
    print(f"  {len(candidate_frames)} candidate frames extracted from video")

    good_faces = []  # list of (det_score, frame, embedding)

    for frame in candidate_frames:
        faces = face_app.get(frame)  # returns list of detected faces in this frame

        if len(faces) != 1:
            continue  # skip frames with 0 or >1 faces — enrollment needs exactly one

        face = faces[0]
        if face.det_score < MIN_DET_SCORE:
            continue  # skip low-confidence / blurry detections

        good_faces.append((face.det_score, frame, face.normed_embedding))

    if not good_faces:
        print(f"  WARNING: no usable frames found for {person_name} — reshoot needed.")
        return

    # Keep the highest-confidence frames, spread out rather than consecutive
    good_faces.sort(key=lambda x: x[0], reverse=True)
    kept = good_faces[:MAX_KEPT_FRAMES]
    print(f"  Kept {len(kept)} good frames (det_score range: "
          f"{kept[-1][0]:.2f}-{kept[0][0]:.2f})")

    # Save face crops for manual sanity-check, and collect embeddings
    person_face_dir = os.path.join(FACES_OUT_DIR, person_name)
    os.makedirs(person_face_dir, exist_ok=True)

    embeddings = []
    for i, (score, frame, emb) in enumerate(kept):
        cv2.imwrite(os.path.join(person_face_dir, f"frame_{i}.jpg"), frame)
        embeddings.append(emb)

    embeddings = np.array(embeddings)  # shape: (num_kept, 512)
    np.save(os.path.join(EMBEDDINGS_OUT_DIR, f"{person_name}.npy"), embeddings)
    print(f"  Saved embeddings: {embeddings.shape} -> "
          f"{EMBEDDINGS_OUT_DIR}/{person_name}.npy")


def main():
    if not os.path.isdir(VIDEOS_DIR):
        print(f"ERROR: '{VIDEOS_DIR}' does not exist.")
        print("Check two things:")
        print(f"  1. You're running this from the repo root "
              f"(AI-Surveillance-System/), not from inside recognition/.")
        print(f"  2. Your enrollment videos are actually in {VIDEOS_DIR}/ "
              f"— create the folder and drop the .mp4 files there if not.")
        return

    videos = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith((".mp4", ".mov"))]
    if not videos:
        print(f"'{VIDEOS_DIR}' exists but has no .mp4/.mov files in it.")
        return

    for video_file in videos:
        person_name = os.path.splitext(video_file)[0]  # filename = person's name
        process_person(os.path.join(VIDEOS_DIR, video_file), person_name)

    print("\nDone. Review kept frames in enrolled_faces/ before trusting the embeddings.")


if __name__ == "__main__":
    main()
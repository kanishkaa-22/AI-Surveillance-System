"""
Face detection + embedding extraction on GPU — standalone test,
not wired into tracking yet. Just confirms InsightFace works correctly and
fast on this machine before Step 6/7 depend on it.

Run on either a short video clip or a live webcam.
"""

import time
import cv2
import torch  # noqa: F401 — importing torch first loads PyTorch's bundled CUDA/cuDNN
              # DLLs so onnxruntime-gpu can find them (fixes cublasLt64_12.dll missing)
from insightface.app import FaceAnalysis

# ---- Config ----
SOURCE = "data/detection_tracking_clips/single_person_01.mp4"  # or 0 for webcam
MAX_FRAMES_TO_TEST = 30

face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 -> GPU 0, -1 -> CPU

cap = cv2.VideoCapture(SOURCE)

frame_count = 0
total_time = 0.0

print(f"Testing face detection + embedding on: {SOURCE}\n")

while frame_count < MAX_FRAMES_TO_TEST:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.perf_counter()
    faces = face_app.get(frame)
    elapsed = time.perf_counter() - start
    total_time += elapsed

    print(f"Frame {frame_count}: {len(faces)} face(s) detected, "
          f"{elapsed*1000:.1f} ms")

    for i, face in enumerate(faces):
        print(f"   face {i}: det_score={face.det_score:.2f}, "
              f"embedding_shape={face.normed_embedding.shape}")

    frame_count += 1

cap.release()

if frame_count > 0:
    avg_ms = (total_time / frame_count) * 1000
    print(f"\nAverage per-frame time: {avg_ms:.1f} ms "
          f"({1000/avg_ms:.1f} FPS if run every frame)")
    print("Embedding shape should read (512,) above — that confirms the "
          "detection + recognition bundle is working end-to-end on GPU.")
else:
    print("No frames read — check SOURCE path.")
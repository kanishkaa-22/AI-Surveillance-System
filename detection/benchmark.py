"""
Benchmarks inference-only FPS across:
  - device: CPU vs GPU
  - model size: nano / small / medium
  - format: PyTorch (.pt) vs ONNX

Writes detection/benchmark_results.csv — evaluation

Note: TensorRT (.engine) benchmarking is left out by default since it requires
a matching TensorRT install and export step that can vary by machine/driver
version. If your GPU machine has TensorRT set up, add "engine" as a format
below and export with: model.export(format="engine") before running.
"""

import os
# MUST be set before importing ultralytics. Ultralytics silently pip-installs
# packages it thinks are "missing" whenever it loads an ONNX model — on some
# setups this reinstalls a CPU-only onnxruntime over a working onnxruntime-gpu,
# every single time an ONNX model is loaded. This environment variable is
# Ultralytics' own documented switch to disable that behavior entirely.
os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] = "1"

import time
import csv
import cv2
from ultralytics import YOLO

# ---- Config ----
TEST_VIDEO = "data/detection_tracking_clips/single_person_01.mp4"  # pick one representative clip
MODEL_SIZES = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
DEVICES = ["cpu", 0]          # 0 = GPU 0
FORMATS = ["pt", "onnx"]      # pt = native PyTorch weights, onnx = exported ONNX
N_WARMUP_FRAMES = 5           # discard first few frames (model/CUDA warmup) from timing
RESULTS_CSV = "detection/benchmark_results.csv"


def get_frames(video_path, max_frames=100):
    """Load a fixed number of frames once, so every benchmark run uses identical input."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def export_onnx_if_needed(pt_path):
    onnx_path = pt_path.replace(".pt", ".onnx")
    import os
    if not os.path.exists(onnx_path):
        print(f"  ONNX file not found: {onnx_path}")
        print(f"  Export it ONCE, manually, before running this benchmark:")
        print(f"      python -c \"from ultralytics import YOLO; "
              f"YOLO('{pt_path}').export(format='onnx')\"")
        print(f"  (Exporting from inside this benchmark loop is what breaks your "
              f"onnxruntime-gpu install — Ultralytics silently reinstalls a CPU-only "
              f"onnxruntime during export on some setups. Export once, separately, "
              f"then re-run this benchmark without touching export again.)")
        raise SystemExit(1)
    return onnx_path


def benchmark_one(model_path, device, frames):
    model = YOLO(model_path)

    # Warmup
    for frame in frames[:N_WARMUP_FRAMES]:
        model.predict(source=frame, device=device, classes=[0], verbose=False)

    start = time.perf_counter()
    for frame in frames[N_WARMUP_FRAMES:]:
        model.predict(source=frame, device=device, classes=[0], verbose=False)
    elapsed = time.perf_counter() - start

    n_timed = len(frames) - N_WARMUP_FRAMES
    fps = n_timed / elapsed if elapsed > 0 else 0
    return fps


def main():
    frames = get_frames(TEST_VIDEO)
    print(f"Loaded {len(frames)} frames from {TEST_VIDEO} for benchmarking.\n")

    rows = []
    for model_size in MODEL_SIZES:
        for fmt in FORMATS:
            model_path = model_size if fmt == "pt" else export_onnx_if_needed(model_size)
            for device in DEVICES:
                # ONNX export above is CPU/GPU agnostic at export time; ORT picks
                # provider based on onnxruntime-gpu install + requested device.
                label = f"{model_size} | {fmt} | device={device}"
                print(f"Running: {label}")
                try:
                    fps = benchmark_one(model_path, device, frames)
                    print(f"  -> {fps:.2f} FPS")
                except Exception as e:
                    print(f"  -> FAILED: {e}")
                    fps = None
                rows.append({
                    "model": model_size,
                    "format": fmt,
                    "device": "GPU" if device == 0 else "CPU",
                    "fps": round(fps, 2) if fps else "FAILED",
                })

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "format", "device", "fps"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
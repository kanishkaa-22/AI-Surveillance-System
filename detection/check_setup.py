"""
Run this before doing anything else.
If any check fails, fix it before moving to next or starting — everything downstream
silently falls back to CPU without an error if GPU setup is wrong.
"""

import os
# See detection/benchmark.py for why this matters — prevents Ultralytics from
# silently reinstalling onnxruntime (CPU) over onnxruntime-gpu.
os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] = "1"

import sys

print("=" * 50)
print("Environment check")
print("=" * 50)

# --- Python version ---
print(f"Python version: {sys.version.split()[0]}")

# --- PyTorch + CUDA ---
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available: {cuda_ok}")
    if cuda_ok:
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("!! CUDA NOT available — torch is CPU-only. "
              "Reinstall torch with the correct --index-url for your CUDA version.")
except ImportError:
    print("!! torch not installed")

# --- Ultralytics ---
try:
    from ultralytics import YOLO
    print("Ultralytics: installed OK")
except ImportError:
    print("!! ultralytics not installed")

# --- OpenCV ---
try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except ImportError:
    print("!! opencv-python not installed")

# --- InsightFace ---
try:
    import insightface
    print("InsightFace: installed OK")
except ImportError:
    print("!! insightface not installed")

# --- ONNX Runtime GPU ---
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"ONNX Runtime providers: {providers}")
    if "CUDAExecutionProvider" not in providers:
        print("!! CUDAExecutionProvider not found — you likely installed "
              "'onnxruntime' instead of 'onnxruntime-gpu'.")
except ImportError:
    print("!! onnxruntime not installed")

print("=" * 50)
print("If all checks above look correct (no '!!' lines), you're ready for Step 2.")
print("=" * 50)
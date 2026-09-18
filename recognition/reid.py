"""
Re-Identification (ReID) — two methods, side by side.

METHOD 1 — Histogram (cheap, CPU-friendly baseline):
    HSV color histogram of the person's crop. Fast, no GPU needed, but
    fooled easily by similar clothing colors.

METHOD 2 — Deep embedding (GPU, lightweight):
    Uses a pretrained ResNet18 (ImageNet weights, final classification layer
    removed) as a general-purpose appearance feature extractor. This is NOT
    a dedicated person-ReID model (like OSNet) — it's a deliberately simple,
    easy-to-install stand-in that still captures more appearance detail than
    a color histogram. Good enough to demonstrate "does a deep embedding
    beat a cheap heuristic" without pulling in heavier ReID-specific
    dependencies.

Both methods share a LostTrackPool: when ByteTrack drops a track, it goes
into the pool for a short window. When a new track appears, we check it
against everything in the pool using BOTH methods (for comparison — see
id_switch_log.csv produced by main.py) but only ONE method's decision
actually re-links the ID, controlled by ACTIVE_REID_METHOD in config.py.
"""

import time
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
import config

_device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- ResNet18 backend (default) ----
_weights = ResNet18_Weights.IMAGENET1K_V1
_resnet_model = resnet18(weights=_weights)
_resnet_model.fc = torch.nn.Identity()   # strip classification head -> 512-d feature vector
_resnet_model.eval()
_resnet_model.to(_device)

_resnet_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 64)),   # tall aspect ratio, typical for person crops
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---- OSNet backend (opt-in, requires: pip install torchreid) ----
# OSNet is purpose-trained for person re-identification (unlike ResNet18,
# which was trained for general object classification and is just being
# repurposed here) — it should genuinely separate people better. It's kept
# OFF by default (config.DEEP_REID_BACKEND = "resnet18") because torchreid
# is a less common package and installing it this close to a deadline
# carries real risk of the same kind of dependency conflicts this project has already hit with onnxruntime-gpu and insightface.
_osnet_model = None
if config.DEEP_REID_BACKEND == "osnet":
    try:
        import torchreid
        _osnet_model = torchreid.utils.FeatureExtractor(
            model_name="osnet_x1_0",
            model_path="",  # empty = auto-download pretrained ImageNet+ReID weights
            device=_device,
        )
        print("reid.py: OSNet backend loaded successfully — deep ReID is using OSNet.")
    except ImportError:
        print("WARNING: config.DEEP_REID_BACKEND is 'osnet' but torchreid isn't "
              "installed (pip install torchreid). Falling back to resnet18 "
              "for this run.")
        config.DEEP_REID_BACKEND = "resnet18"
else:
    print("reid.py: deep ReID is using ResNet18 (default backend).")


def get_histogram_signature(frame, box):
    """Method 1: HSV color histogram of the crop."""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def get_deep_signature(frame, box):
    """Method 2: deep embedding of the crop — ResNet18 (default) or OSNet
    (opt-in, see DEEP_REID_BACKEND above)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    if config.DEEP_REID_BACKEND == "osnet" and _osnet_model is not None:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        embedding = _osnet_model(crop_rgb)  # torchreid returns a tensor
        return embedding.cpu().numpy().flatten()

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = _resnet_transform(crop_rgb).unsqueeze(0).to(_device)
    with torch.no_grad():
        embedding = _resnet_model(tensor).cpu().numpy().flatten()
    return embedding


def histogram_similarity(sig1, sig2):
    return float(cv2.compareHist(sig1, sig2, cv2.HISTCMP_CORREL))


def deep_similarity(sig1, sig2):
    return float(np.dot(sig1, sig2) / (np.linalg.norm(sig1) * np.linalg.norm(sig2) + 1e-8))


class LostTrackPool:
    """
    Holds recently-lost tracks for a short window, keyed by their original
    track_id, so a reappearing person can be matched back to their old ID
    instead of getting a brand-new one.
    """

    def __init__(self, max_age_seconds=30):
        self.max_age_seconds = max_age_seconds
        self.pool = {}  # track_id -> {"hist_sig": ..., "deep_sig": ..., "lost_at": time.time()}

    def add(self, track_id, hist_sig, deep_sig):
        self.pool[track_id] = {
            "hist_sig": hist_sig,
            "deep_sig": deep_sig,
            "lost_at": time.time(),
        }

    def expire_old(self):
        now = time.time()
        expired = [tid for tid, v in self.pool.items()
                   if now - v["lost_at"] > self.max_age_seconds]
        for tid in expired:
            del self.pool[tid]

    def find_best_match(self, hist_sig, deep_sig,
                         hist_threshold=0.6, deep_threshold=0.75):
        """
        Returns a dict with BOTH methods' best match (for comparison logging)
        plus which track_id each method would re-link to, or None.
        """
        self.expire_old()

        best_hist_id, best_hist_score = None, -1.0
        best_deep_id, best_deep_score = None, -1.0

        for tid, v in self.pool.items():
            if hist_sig is not None and v["hist_sig"] is not None:
                score = histogram_similarity(hist_sig, v["hist_sig"])
                if score > best_hist_score:
                    best_hist_score, best_hist_id = score, tid

            if deep_sig is not None and v["deep_sig"] is not None:
                score = deep_similarity(deep_sig, v["deep_sig"])
                if score > best_deep_score:
                    best_deep_score, best_deep_id = score, tid

        hist_match = best_hist_id if best_hist_score >= hist_threshold else None
        deep_match = best_deep_id if best_deep_score >= deep_threshold else None

        return {
            "hist_match_id": hist_match,
            "hist_score": best_hist_score,
            "deep_match_id": deep_match,
            "deep_score": best_deep_score,
        }

    def remove(self, track_id):
        self.pool.pop(track_id, None)
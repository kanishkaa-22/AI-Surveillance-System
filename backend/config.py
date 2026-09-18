"""
Central config — the team tunes these together instead of editing thresholds
scattered across files. Every value here has a short note on what raising
or lowering it does, so you can reason about trade-offs together.
"""

# ---- Detection / Tracking ----
MODEL_PATH = "yolo11s.pt"     # UPGRADED from yolo11n — benchmark showed yolo11s still runs ~135-148 FPS on GPU (RTX 5070), comfortably real-time, with meaningfully better accuracy on close/overlapping people (Issue B). First run will auto-download yolo11s.pt.
DEVICE = 0                     # GPU 0. Use "cpu" to force CPU.
TRACK_LOST_AFTER_FRAMES = 5    # ByteTrack can switch to a brand-new raw track ID within just 1-2 frames on a fast, abrupt occlusion/motion event. At 30 (the old default), those fast switches were never registered in the lost pool in time for ReID to catch them at all. Test at 5 first; only raise toward 15-20 if 5 causes its own problems (e.g. treating a brief real occlusion as a full loss too eagerly).

# ---- Face Recognition ----
MATCH_THRESHOLD = 0.28         # cosine similarity floor to accept a face match. Higher = fewer false positives (strangers misidentified as known people), but more false negatives (known people missed).
                                # Empirically calibrated via calibrate_threshold.py across 4 enrolled people — the original 0.45 was well above 3 of 4 people's genuine-match median, causing frequent
                                # under-recognition. 0.28 sits just above the highest measured
                                # impostor 95th-percentile score (person_1, 0.286) while being far more
                                # permissive for genuine matches. NOT changed to the suggested
                                # 0.30-0.32 yet — person_1's margin is already thin at 0.28, and a
                                # crowd-clip misidentification is more likely a rare outlier slipping
                                # past this margin than proof 0.28 itself is wrong. 
                                # re-run calibrate_threshold.py
                                # including the specific misidentified crowd frame first.
CONFIDENCE_FLOOR = 0.25        # below this score, treat it as "no reliable signal" rather than a meaningful non-match — a 0.02 score and a 0.43 score are both "Unknown" today, but they mean very different things. Scores are still stored as-is in the DB either way, so this doesn't hide data, it just tells you which numbers are noise.
RECOGNITION_LOOKBACK_FRAMES = 28  # raised from 8, per team suggestion — more chances to catch a good frame, at the cost of ~3x more face-detection work per trigger event, and more memory held per active track in crowd scenes.
REMEMBER_IDENTITY_PER_TRACK = True  # once a track_id is confidently recognized, reuse that identity for later events on the SAME track_id if a later trigger finds no usable face (e.g. exiting with back turned). Trade-off: if a track_id ever gets silently reassigned to a different real person, this could carry the wrong name forward — the ReID system reduces how often that happens, but doesn't eliminate it.
DET_SIZE = (640, 640)          # face detector input resolution. Raise (e.g. 800x800) only if you observe distant/small faces failing to detect — larger sizes cost more compute for no benefit if your camera distance doesn't need it.
NMS_IOU_THRESHOLD = 0.7        # how much two person boxes can overlap before YOLO's own duplicate-suppression treats them as "the same detection" and drops one. Standard default is 0.7. If close/overlapping people in a crowd are being merged into a single detection, try raising this (e.g. 0.8) — allows boxes to overlap more before being considered duplicates. Raising too far risks the opposite problem: one person's box getting detected twice.

# ---- Tracker tuning ----
# Points at a custom ByteTrack config with a longer track_buffer and slightly
# looser match_thresh than Ultralytics' default — reduces track fragmentation
# (one person getting split into multiple track IDs) before ReID even needs
# to step in. See backend/custom_bytetrack.yaml for the actual values.
TRACKER_CONFIG_PATH = "backend/custom_bytetrack.yaml"

# ---- GPU session tuning ----
GPU_MEM_LIMIT_BYTES = 5 * 1024**3   # raised from 4GB to 5GB, per team suggestion — cap for the face recognition session's CUDA arena, instead of ONNX Runtime's default "no limit." Your RTX 5070 has plenty of headroom either way.
USE_TENSORRT = True            # ENABLED — your check_setup.py confirms TensorrtExecutionProvider is available and working alongside CUDA. The first run will compile an engine (can take a minute or two) — this is normal, not a hang. If it fails, set back to False; nothing else in the pipeline depends on this succeeding.
TRT_ENGINE_CACHE_PATH = "backend/trt_cache"

# ---- ReID ----
ACTIVE_REID_METHOD = "deep"    # "histogram" or "deep" — which method's decision actually re-links a track_id. Both are still computed and logged either way, for comparison.
DEEP_REID_BACKEND = "resnet18"  # "resnet18" (default, proven working) or "osnet" (opt-in upgrade, requires: pip install torchreid). This is what actually powers the "deep" method above — ACTIVE_REID_METHOD="deep" alone does NOT tell you which backend is running; check this line too.
REID_HIST_THRESHOLD = 0.55     # LOWERED again from 0.6 — real id_switch_log.csv data showed the median genuine-relink score (0.582) sitting BELOW the 0.6 threshold, meaning most real reappearances were being rejected. This is a pragmatic estimate from raw score percentiles, not a true genuine-vs-impostor calibration — complete the manual reid_ground_truth_template.csv labeling (see evaluate_metrics.py) for a rigorous number later.
REID_DEEP_THRESHOLD = 0.68     # LOWERED again from 0.75 — same reasoning; median score (0.7215) sat below threshold. Re-evaluate once OSNet is compared, since this value is specific to the ResNet18 backend's score distribution.
REID_POOL_MAX_AGE_SECONDS = 42 # raised from 30, per team suggestion.

# ---- Zone / Trigger ----
COOLDOWN_SECONDS = 25          # raised from 15, per team suggestion.

# ---- Periodic recognition (evaluation/testing aid, OFF by default) ----
# By design, recognition normally only runs when someone crosses the zone
# line — this saves compute for a real deployment, but means anyone who
# never crosses the line is NEVER checked at all (not "Unknown" — simply
# never attempted). For accuracy evaluation on multi-person clips where you
# want EVERY visible person checked, not just line-crossers, turn this on.
# Leave it False for the actual live demo — it exists for testing, not
# production use.
PERIODIC_RECOGNITION_ENABLED = False
PERIODIC_RECOGNITION_INTERVAL_FRAMES = 30  # how often (in frames) to re-check each currently-visible person, independent of zone crossings

# ---- Movement analytics (novelty feature — dwell-time heatmap) ----
POSITION_LOG_INTERVAL_FRAMES = 10  # log each visible person's position roughly every N frames — dense enough for a useful heatmap, sparse enough to keep the table small

# ---- Zone coordinates: LINE (entry/exit crossing) ----
# Paste the output of backend/utils/get_zone_coords.py (line mode) here.
# This placeholder will NOT match your real camera setup.
ZONE_POINTS = [(320, 50), (320, 480)]
ZONE_NAME = "main_entry"

# ---- Zone coordinates: RECTANGLE (restricted area — presence, not crossing) ----
# Optional. Leave RESTRICTED_ZONE_ENABLED = False until you've actually
# clicked 4 real corners with get_zone_coords.py (rectangle mode).
RESTRICTED_ZONE_ENABLED = False
RESTRICTED_ZONE_POINTS = [(0, 0), (0, 0), (0, 0), (0, 0)]
RESTRICTED_ZONE_NAME = "restricted_area"
RESTRICTED_ZONE_COOLDOWN_SECONDS = 20  # separate cooldown from line-crossing events

# ---- Paths (relative to repo root — always run scripts from there) ----
DB_PATH = "backend/database/surveillance.db"
DATA_DIR = "data"
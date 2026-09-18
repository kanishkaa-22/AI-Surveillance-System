"""
detect -> track -> ReID (relink on reappearance) -> zone trigger ->
recognize -> log to DB -> write annotated frame for the dashboard

Run from repo root:
    python backend/main.py --source data/detection_tracking_clips/reid_reentry_01.mp4
    python backend/main.py --source 0        (webcam, live)

This is a LIVE frame loop (not reading a pre-saved tracking CSV like
zone_and_trigger.py did) — ReID needs to react the instant a track
disappears/reappears, which isn't possible from an offline CSV alone.

Outputs:
    - annotated output video: data/track_outputs/<name>_full_pipeline.mp4
    - dashboard/latest_frame.jpg   (overwritten every frame — dashboard reads this)
    - events + track_history rows written to the database
    - data/track_outputs/id_switch_log.csv — every ReID decision, both methods, for the comparison study
"""

import os
os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] = "1"  # see detection/benchmark.py
import sys
import csv
import json
import time
import argparse
import sqlite3
import cv2
from collections import deque
from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "recognition"))
from reid import get_histogram_signature, get_deep_signature, LostTrackPool
from enrollment_and_matching import load_all_enrolled_embeddings, match_face

sys.path.append(os.path.dirname(__file__))
import config
from utils.zone_geometry import side_of_line, point_in_polygon

from insightface.app import FaceAnalysis

ID_SWITCH_LOG_PATH = "data/track_outputs/id_switch_log.csv"
DASHBOARD_FRAME_PATH = "dashboard/latest_frame.jpg"


def get_or_create_zone(conn, name=None, zone_type="line", coords=None):
    name = name or config.ZONE_NAME
    coords = coords or config.ZONE_POINTS
    cur = conn.cursor()
    cur.execute("SELECT zone_id FROM zones WHERE name = ?", (name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute(
        "INSERT INTO zones (name, zone_type, coords) VALUES (?, ?, ?)",
        (name, zone_type, json.dumps(coords)),
    )
    conn.commit()
    return cur.lastrowid


def get_or_create_session(conn, source):
    cur = conn.cursor()
    cur.execute("INSERT INTO sessions (video_source) VALUES (?)", (str(source),))
    conn.commit()
    return cur.lastrowid


def log_event(conn, session_id, track_id, user_id, zone_id, event_type, confidence):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO events (session_id, track_id, user_id, zone_id, event_type, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, track_id, user_id, zone_id, event_type, confidence))
    conn.commit()


def update_track_history(conn, session_id, track_id, user_id=None):
    cur = conn.cursor()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("""
        SELECT track_history_id FROM track_history
        WHERE session_id = ? AND track_id = ?
    """, (session_id, track_id))
    row = cur.fetchone()
    if row:
        cur.execute("""
            UPDATE track_history SET last_seen = ?, user_id = COALESCE(?, user_id)
            WHERE track_history_id = ?
        """, (now, user_id, row[0]))
    else:
        cur.execute("""
            INSERT INTO track_history (session_id, track_id, user_id, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, track_id, user_id, now, now))
    conn.commit()


def log_position(conn, session_id, track_id, user_id, cx, cy):
    """..."""
    cx = float(cx)
    cy = float(cy)
    if not hasattr(log_position, "_confirmed"):
        print(f"[log_position] Confirming fix is active — cx={cx} (type {type(cx).__name__}), "
              f"cy={cy} (type {type(cy).__name__}). Both must say 'float', not 'float32'/'ndarray'.")
        log_position._confirmed = True
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO position_log (session_id, track_id, user_id, cx, cy)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, track_id, user_id, cx, cy))
    conn.commit()


def get_best_face_embedding(face_app, crop_buffer):
    """
    Runs face detection on every crop in a track's recent-frame buffer and
    returns the embedding from whichever crop had the highest detection
    confidence — instead of trusting whatever single frame happened to
    trigger the zone/restricted-area check. Returns (embedding, det_score),
    or (None, 0.0) if no crop in the buffer had a clean single-face detection.
    """
    best_score = -1.0
    best_embedding = None
    for crop in crop_buffer:
        if crop is None or crop.size == 0:
            continue
        faces = face_app.get(crop)
        if len(faces) == 1 and faces[0].det_score > best_score:
            best_score = faces[0].det_score
            best_embedding = faces[0].normed_embedding
    return best_embedding, (best_score if best_embedding is not None else 0.0)


def resolve_identity(conn, face_app, enrolled, stable_id, buffer, track_identity_cache):
    """
    Full recognition decision for one trigger event. Combines three fixes
    raised in teammate review:
      - CONFIDENCE_FLOOR: a near-zero score (no reliable signal) and a
        genuine near-threshold score are both "Unknown" in the end result,
        but this keeps the raw score itself honest in the DB either way, so
        the distinction is still queryable later.
      - REMEMBER_IDENTITY_PER_TRACK: if this exact track was already
        confidently recognized earlier (e.g. on entry, facing the camera)
        and this trigger finds no usable face at all (e.g. exiting, back
        turned), fall back to that track's last known identity instead of
        logging a fresh "Unknown".
      - Returns det_score too, for the "best face det_score=" logging.

    Returns: (name, score, user_id, det_score)
    """
    best_embedding, det_score = get_best_face_embedding(face_app, buffer)

    if best_embedding is None:
        # No face found in ANY buffered frame — genuinely no signal, not a
        # low-confidence non-match. Keep score as None so this is
        # distinguishable from a real (if low) match score in the DB.
        name, score = "Unknown", None
    else:
        name, score = match_face(best_embedding, enrolled, threshold=config.MATCH_THRESHOLD)
        if name != "Unknown" and score < config.CONFIDENCE_FLOOR:
            # Shouldn't normally happen (MATCH_THRESHOLD is usually higher
            # than CONFIDENCE_FLOOR), but guards against a misconfigured
            # threshold silently accepting near-zero-confidence matches.
            name = "Unknown"

    if name == "Unknown" and config.REMEMBER_IDENTITY_PER_TRACK and stable_id in track_identity_cache:
        name, score = track_identity_cache[stable_id]
    elif name != "Unknown":
        track_identity_cache[stable_id] = (name, score)

    user_id = None
    if name != "Unknown":
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM users WHERE name = ?", (name,))
        row = cur.fetchone()
        user_id = row[0] if row else None

    return name, score, user_id, det_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/detection_tracking_clips/reid_reentry_01.mp4",
                         help="Video file path or 0 for webcam")
    args = parser.parse_args()
    source = 0 if args.source == "0" else args.source

    os.makedirs("data/track_outputs", exist_ok=True)
    os.makedirs("dashboard", exist_ok=True)

    # ---- Models ----
    yolo = YOLO(config.MODEL_PATH)

    # Explicit provider order + memory limit, instead of ONNX Runtime's
    # defaults (unbounded GPU memory, TensorRT installed but never
    # requested)
    if config.USE_TENSORRT:
        providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        provider_options = [
            {"trt_engine_cache_enable": True, "trt_engine_cache_path": config.TRT_ENGINE_CACHE_PATH},
            {"gpu_mem_limit": config.GPU_MEM_LIMIT_BYTES},
            {},
        ]
        os.makedirs(config.TRT_ENGINE_CACHE_PATH, exist_ok=True)
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        provider_options = [{"gpu_mem_limit": config.GPU_MEM_LIMIT_BYTES}, {}]

    try:
        face_app = FaceAnalysis(name="buffalo_l", providers=providers, provider_options=provider_options)
    except TypeError:
        # Older insightface versions may not accept provider_options directly.
        print("NOTE: this insightface version didn't accept provider_options — "
              "falling back to providers only (GPU mem limit not applied).")
        face_app = FaceAnalysis(name="buffalo_l", providers=providers)
    face_app.prepare(ctx_id=0, det_size=config.DET_SIZE)
    enrolled = load_all_enrolled_embeddings()

    # ---- Database ----
    conn = sqlite3.connect(config.DB_PATH)
    zone_id = get_or_create_zone(conn)
    restricted_zone_id = None
    if config.RESTRICTED_ZONE_ENABLED:
        restricted_zone_id = get_or_create_zone(
            conn, name=config.RESTRICTED_ZONE_NAME, zone_type="polygon",
            coords=config.RESTRICTED_ZONE_POINTS,
        )
    session_id = get_or_create_session(conn, source)

    # ---- ReID state ----
    lost_pool = LostTrackPool(max_age_seconds=config.REID_POOL_MAX_AGE_SECONDS)
    if not os.path.exists(ID_SWITCH_LOG_PATH):
        open(ID_SWITCH_LOG_PATH, "w").close()
    id_switch_log = open(ID_SWITCH_LOG_PATH, "a", newline="")
    id_switch_writer = csv.writer(id_switch_log)
    if os.path.getsize(ID_SWITCH_LOG_PATH) == 0:
        id_switch_writer.writerow([
            "session_id", "frame", "new_raw_track_id", "resolved_track_id",
            "hist_match_id", "hist_score", "deep_match_id", "deep_score",
            "active_method_used", "deep_backend",
        ])

    raw_to_stable = {}       # ByteTrack's raw track_id -> our stable (possibly re-linked) ID
    seen_before = set()      # raw track_ids we've ever seen (to detect "brand new")
    last_side = {}           # stable_id -> which side of the zone line
    last_event_time = {}     # stable_id -> last logged event time (cooldown)
    last_in_restricted = {}  # stable_id -> currently inside the restricted area? (bool)
    last_restricted_event_time = {}  # stable_id -> last restricted-access event time (separate cooldown)
    track_cache = {}         # raw_id -> {"last_frame": idx, "hist_sig": ..., "deep_sig": ...}
    moved_to_pool = set()    # raw_ids already added to the lost pool, so we don't re-add every frame
    face_crop_buffers = {}   # stable_id -> deque of recent face crops, for best-frame recognition
    track_identity_cache = {}  # stable_id -> (name, score), last CONFIDENT recognition for this track
    last_periodic_check = {}  # stable_id -> frame_idx of last periodic recognition check (evaluation aid only)

    # ---- Video I/O ----
    cap_probe = cv2.VideoCapture(source)
    fps = cap_probe.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_probe.release()

    clip_name = "webcam" if source == 0 else os.path.splitext(os.path.basename(str(source)))[0]
    out_path = f"data/track_outputs/{clip_name}_full_pipeline.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    results = yolo.track(
        source=source, classes=[0], tracker=config.TRACKER_CONFIG_PATH,
        persist=True, device=config.DEVICE, stream=True, verbose=False,
        iou=config.NMS_IOU_THRESHOLD,
    )

    frame_idx = 0
    print(f"Processing: {source}  (press Ctrl+C to stop early)")
    print(f"  MATCH_THRESHOLD={config.MATCH_THRESHOLD}  "
          f"CONFIDENCE_FLOOR={config.CONFIDENCE_FLOOR}  "
          f"ACTIVE_REID_METHOD={config.ACTIVE_REID_METHOD}  "
          f"DEEP_REID_BACKEND={config.DEEP_REID_BACKEND}  "
          f"TRACKER_CONFIG_PATH={config.TRACKER_CONFIG_PATH}")

    for r in results:
        frame = r.orig_img
        current_frame_raw_ids = set()
        frame_labels = []  # (x1, y1, text) to draw on this frame's annotated output
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            raw_ids = r.boxes.id.cpu().numpy().astype(int)

            for box, raw_id in zip(boxes, raw_ids):
                # Normalize immediately to a native Python int — numpy's
                # int64 type looks like a normal integer everywhere in
                # Python, but sqlite3 sometimes silently stores it as raw
                # bytes instead of converting it to a real integer (this is
                # what caused corrupted track_id values in the database).
                # Converting here, once, at the earliest point, guarantees
                # every downstream use (dict keys, DB inserts) is a clean int.
                raw_id = int(raw_id)
                current_frame_raw_ids.add(raw_id)

                # ---- New raw track_id? Try ReID before accepting it as brand new ----
                if raw_id not in seen_before:
                    seen_before.add(raw_id)
                    hist_sig = get_histogram_signature(frame, box)
                    deep_sig = get_deep_signature(frame, box)
                    match = lost_pool.find_best_match(
                        hist_sig, deep_sig,
                        hist_threshold=config.REID_HIST_THRESHOLD,
                        deep_threshold=config.REID_DEEP_THRESHOLD,
                    )

                    chosen_match = (match["deep_match_id"] if config.ACTIVE_REID_METHOD == "deep"
                                     else match["hist_match_id"])

                    if chosen_match is not None:
                        stable_id = chosen_match
                        lost_pool.remove(chosen_match)
                        log_event(conn, session_id, stable_id, None, zone_id, "reid_relink", None)
                    else:
                        stable_id = raw_id  # genuinely new person

                    raw_to_stable[raw_id] = stable_id

                    id_switch_writer.writerow([
                        session_id, frame_idx, raw_id, stable_id,
                        match["hist_match_id"], round(match["hist_score"], 3),
                        match["deep_match_id"], round(match["deep_score"], 3),
                        config.ACTIVE_REID_METHOD, config.DEEP_REID_BACKEND,
                    ])

                stable_id = raw_to_stable[raw_id]
                update_track_history(conn, session_id, stable_id)

                # Cache this frame's signature — used later if this track goes missing
                track_cache[raw_id] = {
                    "last_frame": frame_idx,
                    "hist_sig": get_histogram_signature(frame, box),
                    "deep_sig": get_deep_signature(frame, box),
                }

                # ---- Zone crossing check ----
                x1, y1, x2, y2 = box
                cx, cy = float((x1 + x2) / 2), float((y1 + y2) / 2)  # explicit cast — box coords come from numpy (float32), and like the earlier track_id bug, an uncast numpy scalar can get silently stored as raw bytes by sqlite3 instead of a real number

                # Keep a rolling buffer of this person's recent crops, so a
                # trigger can pick the best-angled frame instead of relying
                # on whichever single frame happened to cross the line.
                x1c, y1c = max(int(x1), 0), max(int(y1), 0)
                crop_now = frame[y1c:int(y2), x1c:int(x2)]
                if stable_id not in face_crop_buffers:
                    face_crop_buffers[stable_id] = deque(maxlen=config.RECOGNITION_LOOKBACK_FRAMES)
                if crop_now.size > 0:
                    face_crop_buffers[stable_id].append(crop_now.copy())

                # Look up whatever identity this track was last confidently
                # recognized as (if any) and queue it to be drawn on the
                # output video/dashboard frame. Recognition is trigger-based
                # (only runs on a zone crossing, not every frame — see
                # earlier design notes on why), so a track shows "..." until
                # its first crossing gives it a name.
                cached_name = track_identity_cache.get(stable_id, (None, None))[0]
                label_text = f"#{stable_id} {cached_name}" if cached_name else f"#{stable_id} ..."
                frame_labels.append((int(x1), int(y2) + 5, label_text))  # below the box, not above (avoids overlapping YOLO's own top label)

                # ---- Movement analytics: log position on an interval ----
                if frame_idx % config.POSITION_LOG_INTERVAL_FRAMES == 0:
                    p_user_id_for_log = None
                    if cached_name:
                        cur = conn.cursor()
                        cur.execute("SELECT user_id FROM users WHERE name = ?", (cached_name,))
                        row = cur.fetchone()
                        p_user_id_for_log = row[0] if row else None
                    log_position(conn, session_id, stable_id, p_user_id_for_log, cx, cy)

                # ---- Periodic recognition (evaluation aid, off by default) ----
                # Checks EVERY visible person on an interval, independent of
                # whether they've crossed the zone line. Off by default so
                # the real demo/deployment stays trigger-only (see config.py
                # comment). Logged as event_type="periodic_check" (zone_id
                # None) so it never gets confused with a real entry/exit in
                # the events table, but still counts toward recognition
                # accuracy numbers in evaluate_metrics.py.
                if config.PERIODIC_RECOGNITION_ENABLED:
                    last_checked = last_periodic_check.get(stable_id, -config.PERIODIC_RECOGNITION_INTERVAL_FRAMES - 1)
                    if frame_idx - last_checked >= config.PERIODIC_RECOGNITION_INTERVAL_FRAMES:
                        last_periodic_check[stable_id] = frame_idx
                        buffer = face_crop_buffers.get(stable_id, [])
                        p_name, p_score, p_user_id, p_det_score = resolve_identity(
                            conn, face_app, enrolled, stable_id, buffer, track_identity_cache,
                        )
                        log_event(conn, session_id, stable_id, p_user_id, None, "periodic_check", p_score)
                        update_track_history(conn, session_id, stable_id, p_user_id)
                        p_score_str = f"{p_score:.2f}" if p_score is not None else "no face detected"
                        print(f"Frame {frame_idx}: track {stable_id} periodic check -> "
                              f"{p_name} ({p_score_str})")

                side = side_of_line((cx, cy), config.ZONE_POINTS[0], config.ZONE_POINTS[1])
                side_sign = 1 if side > 0 else -1
                crossed = stable_id in last_side and last_side[stable_id] != side_sign
                last_side[stable_id] = side_sign

                if crossed:
                    now = time.time()
                    if (stable_id not in last_event_time
                            or (now - last_event_time[stable_id]) >= config.COOLDOWN_SECONDS):
                        last_event_time[stable_id] = now
                        event_type = "entry" if side_sign > 0 else "exit"

                        # Check the last few frames of this person, not just
                        # this exact frame, with confidence-floor and
                        # identity-memory handling built in.
                        buffer = face_crop_buffers.get(stable_id, [])
                        name, score, user_id, det_score = resolve_identity(
                            conn, face_app, enrolled, stable_id, buffer, track_identity_cache,
                        )

                        log_event(conn, session_id, stable_id, user_id, zone_id, event_type, score)
                        update_track_history(conn, session_id, stable_id, user_id)
                        score_str = f"{score:.2f}" if score is not None else "no face detected"
                        print(f"Frame {frame_idx}: track {stable_id} -> {event_type}, "
                              f"recognized as {name} ({score_str}, best face det_score={det_score:.2f} "
                              f"from last {len(buffer)} frames)")

                # ---- Restricted-area check (presence, not crossing) ----
                if config.RESTRICTED_ZONE_ENABLED:
                    inside_now = point_in_polygon((cx, cy), config.RESTRICTED_ZONE_POINTS)
                    was_inside = last_in_restricted.get(stable_id, False)
                    last_in_restricted[stable_id] = inside_now

                    if inside_now and not was_inside:
                        now = time.time()
                        cooldown_ok = (
                            stable_id not in last_restricted_event_time
                            or (now - last_restricted_event_time[stable_id])
                            >= config.RESTRICTED_ZONE_COOLDOWN_SECONDS
                        )
                        if cooldown_ok:
                            last_restricted_event_time[stable_id] = now

                            buffer = face_crop_buffers.get(stable_id, [])
                            name, score, user_id, det_score = resolve_identity(
                                conn, face_app, enrolled, stable_id, buffer, track_identity_cache,
                            )

                            log_event(conn, session_id, stable_id, user_id,
                                      restricted_zone_id, "restricted_access", score)
                            update_track_history(conn, session_id, stable_id, user_id)
                            score_str = f"{score:.2f}" if score is not None else "no face detected"
                            print(f"Frame {frame_idx}: track {stable_id} entered RESTRICTED "
                                  f"area, recognized as {name} ({score_str})")

        # ---- Tracks not seen for TRACK_LOST_AFTER_FRAMES -> move to lost pool ----
        for raw_id, cached in list(track_cache.items()):
            frames_missing = frame_idx - cached["last_frame"]
            if frames_missing >= config.TRACK_LOST_AFTER_FRAMES and raw_id not in moved_to_pool:
                stable_id = raw_to_stable.get(raw_id, raw_id)
                lost_pool.add(stable_id, cached["hist_sig"], cached["deep_sig"])
                moved_to_pool.add(raw_id)

        annotated = r.plot()
        for x1, y1, text in frame_labels:
            # Drawn at the BOTTOM of the box, not the top — Ultralytics'
            # own r.plot() already draws its class+track-id label at the
            # top of the box, so placing ours there caused the two to
            # overlap. A filled background strip behind the text keeps it
            # readable over busy video content.
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = y1  # y1 here is actually the box's y2 (bottom) — see below
            cv2.rectangle(annotated, (x1, label_y), (x1 + text_w + 6, label_y + text_h + 10),
                          (0, 0, 0), -1)
            cv2.putText(annotated, text, (x1 + 3, label_y + text_h + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(annotated)
        cv2.imwrite(DASHBOARD_FRAME_PATH, annotated)  # dashboard reads this every refresh

        frame_idx += 1

    writer.release()
    id_switch_log.close()
    conn.close()
    print(f"\nDone. {frame_idx} frames processed.")
    print(f"Annotated video: {out_path}")
    print(f"ID-switch log:   {ID_SWITCH_LOG_PATH}")


if __name__ == "__main__":
    main()
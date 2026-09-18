"""
 Ties together tracking output (Step 2), zone logic, face recognition
(Step 6), and event logging (Step 5) into one chain.

Flow per tracked person, per frame:
  1. Check which side of the zone line they're on.
  2. If they just crossed sides since the last frame -> that's a trigger.
  3. On trigger (and if cooldown has passed for this track_id):
     - grab that frame from the source video
     - crop the person's bounding box
     - run face detection + embedding on the crop
     - match against enrolled users (Step 6)
     - log the event to the database

Run this on the GPU machine, since it calls InsightFace.
"""

import sqlite3
import time
import csv
import cv2
import torch  # noqa: F401 — see recognition/face_test.py for why this must come
              # before the insightface import
from insightface.app import FaceAnalysis
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "recognition"))
from enrollment_and_matching import load_all_enrolled_embeddings, match_face

# ---- Config ----
TRACK_CSV = "data/track_outputs/zone_crossing_01_tracks.csv"   # from Step 2
VIDEO_PATH = "data/detection_tracking_clips/zone_crossing_01.mp4"
DB_PATH = "backend/database/surveillance.db"

# Paste the two points from get_zone_coords.py here (line zone):
ZONE_POINTS = [(811, 203), (637, 979)]  
ZONE_NAME = "main_entry"

COOLDOWN_SECONDS = 15   # don't log a second event for the same track_id within this window


def side_of_line(point, line_p1, line_p2):
    """Returns positive/negative depending on which side of the line the point is on."""
    x, y = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def get_or_create_zone(conn):
    cur = conn.cursor()
    cur.execute("SELECT zone_id FROM zones WHERE name = ?", (ZONE_NAME,))
    row = cur.fetchone()
    if row:
        return row[0]
    import json
    cur.execute(
        "INSERT INTO zones (name, zone_type, coords) VALUES (?, 'line', ?)",
        (ZONE_NAME, json.dumps(ZONE_POINTS)),
    )
    conn.commit()
    return cur.lastrowid


def get_or_create_session(conn):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sessions (video_source) VALUES (?)", (VIDEO_PATH,)
    )
    conn.commit()
    return cur.lastrowid


def crop_and_get_embedding(video_capture, face_app, frame_idx, box):
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = video_capture.read()
    if not ret:
        return None

    # box values come from the CSV as strings like "594.93097" — convert via
    # float first, since int() can't parse a decimal string directly.
    x1, y1, x2, y2 = [int(float(v)) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    faces = face_app.get(crop)
    if len(faces) != 1:
        return None  # no face, or more than one — skip this trigger
    return faces[0].normed_embedding


def main():
    conn = sqlite3.connect(DB_PATH)
    zone_id = get_or_create_zone(conn)
    session_id = get_or_create_session(conn)

    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    enrolled = load_all_enrolled_embeddings()

    cap = cv2.VideoCapture(VIDEO_PATH)

    # Read all track rows, grouped by track_id, in frame order
    with open(TRACK_CSV) as f:
        rows = list(csv.DictReader(f))

    last_side = {}          # track_id -> last known side of line (+1 / -1)
    last_event_time = {}    # track_id -> time.time() of last logged event

    for row in rows:
        track_id = int(row["track_id"])
        frame_idx = int(row["frame"])
        cx, cy = float(row["cx"]), float(row["cy"])
        box = (row["x1"], row["y1"], row["x2"], row["y2"])

        side = side_of_line((cx, cy), ZONE_POINTS[0], ZONE_POINTS[1])
        side_sign = 1 if side > 0 else -1

        crossed = (
            track_id in last_side
            and last_side[track_id] != side_sign
        )
        last_side[track_id] = side_sign

        if not crossed:
            continue

        # Cooldown check
        now = time.time()
        if track_id in last_event_time and (now - last_event_time[track_id]) < COOLDOWN_SECONDS:
            continue
        last_event_time[track_id] = now

        event_type = "entry" if side_sign > 0 else "exit"
        print(f"Frame {frame_idx}: track {track_id} triggered zone crossing ({event_type})")

        embedding = crop_and_get_embedding(cap, face_app, frame_idx, box)

        if embedding is not None:
            name, score = match_face(embedding, enrolled)
        else:
            name, score = "Unknown", 0.0

        cur = conn.cursor()
        user_id = None
        if name != "Unknown":
            cur.execute("SELECT user_id FROM users WHERE name = ?", (name,))
            result = cur.fetchone()
            user_id = result[0] if result else None

        cur.execute("""
            INSERT INTO events (session_id, track_id, user_id, zone_id, event_type, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, track_id, user_id, zone_id, event_type, score))
        conn.commit()

        print(f"  -> Logged: {name} ({event_type}, confidence={score:.3f})")

    cap.release()
    conn.close()
    print("\nDone. Check the events table in surveillance.db.")


if __name__ == "__main__":
    main()
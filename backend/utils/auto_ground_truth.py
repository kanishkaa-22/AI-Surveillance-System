"""
Auto ground-truth helper for ReID and multi-person validation.

Finds occlusion/re-entry boundary frames automatically by running YOUR
ACTUAL tracker config (not Ultralytics' default) on a clip — no need to
have manually noted timestamps. Records every frame where a track ID first
appears (entry) or is last seen before disappearing (exit), then extracts
those frames for you to label.

Run from repo root:
    python backend/utils/auto_ground_truth.py data/detection_tracking_clips/reid_reentry_01.mp4

For a multi-person clip, also log who's really in it (for comparing against every name the pipeline actually reported):
    python backend/utils/auto_ground_truth.py data/detection_tracking_clips/multi_person_02.mp4 --people <name>,someone_else
"""

import os
import sys
import csv
import cv2
import argparse
from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config

OUTPUT_DIR = "data/ground_truth"


def find_track_events(video_path):
    """
    Runs YOLO + YOUR tuned tracker (same config.py / custom_bytetrack.yaml
    as main.py) and returns every frame number where a track ID first
    appears (entry) or is last seen before disappearing (exit). Using the
    same tracker config as the real pipeline matters — a different tracker
    config would find different (wrong) boundary frames.
    """
    model = YOLO(config.MODEL_PATH)
    results = model.track(
        source=video_path, stream=True, persist=True, classes=[0],
        tracker=config.TRACKER_CONFIG_PATH, device=config.DEVICE,
        verbose=False,
    )

    last_seen = {}
    first_seen = {}
    events = []

    frame_idx = 0
    for r in results:
        ids = r.boxes.id
        if ids is not None:
            for tid in ids.int().tolist():
                if tid not in first_seen:
                    first_seen[tid] = frame_idx
                    events.append({"frame": frame_idx, "track_id": tid, "event": "entry"})
                last_seen[tid] = frame_idx
        frame_idx += 1

    for tid, f in last_seen.items():
        events.append({"frame": f, "track_id": tid, "event": "exit"})

    events.sort(key=lambda e: e["frame"])
    return events


def extract_event_frames(video_path, events, clip_name):
    """Saves one image per entry/exit event so you can label each boundary."""
    output_dir = os.path.join(OUTPUT_DIR, f"events_{clip_name}")
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    rows = []
    for e in events:
        idx = e["frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        filename = f"frame_{idx:06d}_track{e['track_id']}_{e['event']}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), frame)
        rows.append({
            "frame_number": idx,
            "track_id": e["track_id"],
            "event": e["event"],
            "image_file": filename,
            "true_identity": "",  # fill by hand: name, or "unenrolled"
        })

    cap.release()
    print(f"Extracted {len(rows)} boundary-frame images to {output_dir}/")
    return rows


def build_reid_event_table(rows, clip_name):
    """
    Pairs up exit->entry across DIFFERENT track_ids to form candidate
    occlusion events for you to review. The pairing is a starting guess
    (nearest following entry with a different track_id) — always verify
    against the actual images before trusting a pairing, especially in
    busy multi-person clips where the naive nearest-match can occasionally
    pick the wrong person.
    """
    output_csv = os.path.join(OUTPUT_DIR, f"{clip_name}_reid_ground_truth.csv")
    exits = [r for r in rows if r["event"] == "exit"]
    entries = [r for r in rows if r["event"] == "entry"]

    fieldnames = [
        "exit_frame", "exit_track_id",
        "entry_frame", "entry_track_id",
        "true_identity",           # fill by hand: who is this really (same both sides)
        "pipeline_label_before",   # fill from main.py's console log / events table: name/Unknown before exit
        "pipeline_label_after",    # fill from main.py's console log / events table: name/Unknown after entry
        "reid_correct",            # fill: y/n, did it correctly re-link?
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ex in exits:
            candidates = [en for en in entries
                          if en["frame_number"] > ex["frame_number"]
                          and en["track_id"] != ex["track_id"]]
            if not candidates:
                continue
            en = min(candidates, key=lambda c: c["frame_number"])
            writer.writerow({
                "exit_frame": ex["frame_number"], "exit_track_id": ex["track_id"],
                "entry_frame": en["frame_number"], "entry_track_id": en["track_id"],
                "true_identity": "", "pipeline_label_before": "",
                "pipeline_label_after": "", "reid_correct": "",
            })

    print(f"ReID event table written to {output_csv} — fill in the blank columns by hand, "
          f"double-checking each pairing against the images in events_{clip_name}/ first.")


def build_multiperson_clip_table(clip_name, people_present):
    """
    Simplest ground truth for multi-person clips: one row listing who's
    really in the clip. Run main.py on this same clip, list every DISTINCT
    name it reported (from console output or the events/track_history
    tables), and fill 'pipeline_reported_names' + flag any name that
    ISN'T in true_people_present as a misrecognition.
    """
    output_csv = os.path.join(OUTPUT_DIR, "multiperson_ground_truth.csv")
    file_exists = os.path.exists(output_csv)
    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["clip_name", "true_people_present", "pipeline_reported_names", "misrecognition_flag"])
        writer.writerow([clip_name, ";".join(people_present), "", ""])
    print(f"Row added to {output_csv} for clip '{clip_name}'. "
          f"Now run main.py on this clip and fill in what it actually reported.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--people", default=None,
                         help="Comma-separated list of who's really in this clip, "
                              "e.g. --people sreesha,evana (for multi-person clips)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clip_name = os.path.splitext(os.path.basename(args.video_path))[0]

    print(f"Finding track entry/exit events in {args.video_path} "
          f"(using your tuned tracker config, same as main.py)...")
    events = find_track_events(args.video_path)
    print(f"Found {len(events)} track entry/exit events.")

    rows = extract_event_frames(args.video_path, events, clip_name)
    build_reid_event_table(rows, clip_name)

    if args.people:
        people_present = [p.strip() for p in args.people.split(",")]
        build_multiperson_clip_table(clip_name, people_present)

    print("\nNext steps:")
    print(f"1. Open data/ground_truth/events_{clip_name}/ and fill 'true_identity' where needed.")
    print(f"2. Open data/ground_truth/{clip_name}_reid_ground_truth.csv — fill true_identity, "
          f"pipeline_label_before/after (from your main.py run's console output or DB), and reid_correct (y/n).")
    if args.people:
        print(f"3. Open data/ground_truth/multiperson_ground_truth.csv — run main.py on this clip, "
              f"list every distinct name it reported, flag any not in true_people_present.")


if __name__ == "__main__":
    main()
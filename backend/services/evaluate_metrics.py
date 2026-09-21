"""
Evaluation metrics (Week 3 item 5 / Week 4).

Summarizes the numbers your report needs. Some require a small amount of
manual ground-truth labeling first — this script tells you exactly what
to fill in and where.

Run from repo root: python backend/services/evaluate_metrics.py
"""

import os
import csv
import glob
import sqlite3
import sys
import pandas as pd

ID_SWITCH_LOG = "data/track_outputs/id_switch_log.csv"
BENCHMARK_CSV = "detection/benchmark_results.csv"
DB_PATH = "backend/database/surveillance.db"
GROUND_TRUTH_TEMPLATE = "data/track_outputs/reid_ground_truth_template.csv"


def summarize_reid():
    """
    ID-switch + ReID success rate, comparing histogram vs deep methods.

    This script alone can't know whether a re-link was CORRECT (that
    requires a human watching the clip and confirming it was really the
    same person) — so it generates a template CSV for you to fill in a
    'correct' column (1/0), then re-run this to get the actual success rate.
    """
    if not os.path.exists(ID_SWITCH_LOG):
        print("No id_switch_log.csv yet — run backend/main.py on a leave/re-enter clip first.", flush=True)
        return

    df = pd.read_csv(ID_SWITCH_LOG)
    if df.empty:
        print("id_switch_log.csv is empty — no reappearance events logged yet.", flush=True)
        return

    print(f"\n{len(df)} track-reappearance events logged.", flush=True)
    print(f"  Histogram method proposed a match in {df['hist_match_id'].notna().sum()} cases.", flush=True)
    print(f"  Deep method proposed a match in {df['deep_match_id'].notna().sum()} cases.", flush=True)

    if not os.path.exists(GROUND_TRUTH_TEMPLATE):
        df_out = df.copy()
        df_out["was_it_really_the_same_person(1=yes,0=no)"] = ""
        os.makedirs(os.path.dirname(GROUND_TRUTH_TEMPLATE), exist_ok=True)
        df_out.to_csv(GROUND_TRUTH_TEMPLATE, index=False)
        print(f"\nTemplate written to {GROUND_TRUTH_TEMPLATE}", flush=True)
        print("  Watch your leave/re-enter clips, fill in the last column "
              "with 1 or 0 for each row, then re-run this script for the "
              "actual ReID success rate comparison.", flush=True)
        return

    labeled = pd.read_csv(GROUND_TRUTH_TEMPLATE)
    truth_col = "was_it_really_the_same_person(1=yes,0=no)"
    labeled = labeled[labeled[truth_col].isin([0, 1, "0", "1"])]
    if labeled.empty:
        print(f"\n{GROUND_TRUTH_TEMPLATE} exists but has no filled-in rows yet.", flush=True)
        return

    labeled[truth_col] = labeled[truth_col].astype(int)

    for method, match_col in [("Histogram", "hist_match_id"), ("Deep", "deep_match_id")]:
        proposed = labeled[labeled[match_col].notna()]
        if proposed.empty:
            print(f"\n{method}: no matches proposed to evaluate.", flush=True)
            continue
        correct = (proposed[truth_col] == 1).sum()
        total = len(proposed)
        print(f"\n{method} ReID success rate: {correct}/{total} ({100*correct/total:.1f}%)", flush=True)


def summarize_recognition_accuracy():
    """
    True positive rate on enrolled users, false positive rate on unknown.
    Reads directly from the events table — assumes you've run zone_and_trigger.py
    or main.py on both enrolled-person clips and your unknown_01/unknown_02 clips.
    """
    if not os.path.exists(DB_PATH):
        print("\nNo database found yet.", flush=True)
        return

    conn = sqlite3.connect(DB_PATH)
    events = pd.read_sql_query("""
        SELECT events.confidence, users.name
        FROM events
        LEFT JOIN users ON users.user_id = events.user_id
        WHERE events.event_type IN ('entry', 'exit')
    """, conn)
    conn.close()

    if events.empty:
        print("\nNo entry/exit events logged yet.", flush=True)
        return

    recognized = events[events["name"].notna()]
    unknown = events[events["name"].isna()]

    print(f"\nRecognition events: {len(recognized)} recognized, {len(unknown)} logged as Unknown.", flush=True)
    if not recognized.empty:
        print(f"  Average confidence on recognized matches: {recognized['confidence'].mean():.3f}", flush=True)
    print("  NOTE: whether each 'recognized' row and each 'Unknown' row is "
          "actually CORRECT still needs a manual check against who was really "
          "in that clip — cross-reference against your unknown_01/unknown_02 "
          "clips (should show 0 recognized rows) and your enrolled-person "
          "clips (should show 0 Unknown rows) to get true/false positive rates.", flush=True)


def summarize_fps():
    if not os.path.exists(BENCHMARK_CSV):
        print("\nNo benchmark_results.csv yet — run detection/benchmark.py first.", flush=True)
        return
    df = pd.read_csv(BENCHMARK_CSV)
    print("\nEnd-to-end FPS (from Step 3 benchmark):", flush=True)
    print(df.to_string(index=False), flush=True)


def summarize_detection_precision_recall():
    """
    Compares the model's own detections against hand-labeled ground truth.
    Automatically finds and combines EVERY file matching *ground_truth*.csv
    in data/ground_truth/ — you can label separate clips into separate
    files (e.g. multi_person_ground_truth.csv, single_person_ground_truth.csv)
    with label_boxes.py, and this reads all of them together, no manual
    merging needed. A predicted box counts as correct if it overlaps a
    ground-truth box by at least IOU_THRESHOLD.
    """
    GROUND_TRUTH_DIR = "data/ground_truth"
    IMAGES_DIR = "data/ground_truth/images"
    IOU_THRESHOLD = 0.5

    gt_csv_paths = glob.glob(os.path.join(GROUND_TRUTH_DIR, "*ground_truth*.csv"))
    if not gt_csv_paths:
        print(f"\nNo ground truth yet — run backend/utils/extract_frames_for_labeling.py "
              f"then backend/utils/label_boxes.py first. Looking for any file matching "
              f"*ground_truth*.csv in {GROUND_TRUTH_DIR}/", flush=True)
        return

    print(f"\nFound {len(gt_csv_paths)} ground-truth file(s): "
          f"{[os.path.basename(p) for p in gt_csv_paths]}", flush=True)

    import cv2
    from ultralytics import YOLO

    # Ensure parent module imports work
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from backend import config

    def iou(box_a, box_b):
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
        inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0

    gt_df = pd.concat([pd.read_csv(p) for p in gt_csv_paths], ignore_index=True)
    
    if not os.path.exists(config.MODEL_PATH):
        print(f"\nModel file not found at {config.MODEL_PATH}", flush=True)
        return

    model = YOLO(config.MODEL_PATH)

    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    for image_name, group in gt_df.groupby("image"):
        image_path = os.path.join(IMAGES_DIR, image_name)
        if not os.path.exists(image_path):
            continue

        gt_boxes = [
            (row.x1, row.y1, row.x2, row.y2)
            for row in group.itertuples()
            if pd.notna(row.x1)
        ]

        img = cv2.imread(image_path)
        if img is None:
            continue

        results = model.predict(
            source=img,
            classes=[0],
            device=config.DEVICE,
            conf=config.DETECTION_CONF_THRESHOLD,
            verbose=False
        )
        pred_boxes = [tuple(b) for b in results[0].boxes.xyxy.cpu().numpy()] if results else []

        # Count frames with no ground-truth persons and no predicted persons as True Negatives
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            total_tn += 1
            continue

        matched_gt = set()
        for pred_box in pred_boxes:
            best_iou, best_idx = 0, -1
            for idx, gt_box in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue
                score = iou(pred_box, gt_box)
                if score > best_iou:
                    best_iou, best_idx = score, idx
            if best_iou >= IOU_THRESHOLD:
                total_tp += 1
                matched_gt.add(best_idx)
            else:
                total_fp += 1

        total_fn += len(gt_boxes) - len(matched_gt)

    total_instances = total_tp + total_fp + total_fn + total_tn
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    accuracy = (total_tp + total_tn) / total_instances if total_instances > 0 else 0

    print(f"\nDetection metrics (IOU >= {IOU_THRESHOLD}, {gt_df['image'].nunique()} labeled images):", flush=True)
    print(f"  True positives:  {total_tp}", flush=True)
    print(f"  False positives: {total_fp}  (model detected a person that wasn't labeled)", flush=True)
    print(f"  False negatives: {total_fn}  (a labeled person the model missed)", flush=True)
    print(f"  True negatives:  {total_tn}  (no person labeled and none detected)", flush=True)
    print(f"  Precision: {precision:.3f}   Recall: {recall:.3f}   Accuracy: {accuracy:.3f}", flush=True)


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("EVALUATION METRICS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Working Directory: {os.getcwd()}", flush=True)
    
    summarize_fps()
    summarize_reid()
    summarize_recognition_accuracy()
    summarize_detection_precision_recall()
    
    print("=" * 60, flush=True)
    print("EVALUATION COMPLETE", flush=True)
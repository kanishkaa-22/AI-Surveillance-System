"""
Manual ground-truth bounding box labeler. Same click-based style as
get_zone_coords.py — no external tool or account needed.

For each image in data/ground_truth/images/:
  - Click TWO points per person: top-left corner, then bottom-right corner
    of a box around them. Draw a box for EVERY person, regardless of
    whether their face is visible — this tool labels "is a person here,"
    not "who is this" or "can I see their face."
  - Press 'n' to move to the next image once you've boxed everyone in this one.
  - Press 's' to skip an image with nobody in it (still gets recorded, as
    zero boxes — important for false-positive checking).
  - Press 'q' to quit early (saves everything labeled so far).

Run from repo root:
    python backend/utils/label_boxes.py
    python backend/utils/label_boxes.py multi_person_ground_truth.csv

The optional argument names the output CSV — use this to keep separate
clips' ground truth in separate files (e.g. label one clip's frames, save
as multi_person_ground_truth.csv, then clear/replace the images folder and
label the next clip into single_person_ground_truth.csv). If you skip the
argument, it saves to the default ground_truth.csv.

evaluate_metrics.py automatically finds and combines ALL files matching
*ground_truth*.csv in data/ground_truth/ — you don't need to merge them
yourself, and you don't need to name them anything special beyond having
"ground_truth" somewhere in the filename.
"""

import os
import sys
import csv
import cv2

IMAGES_DIR = "data/ground_truth/images"
DEFAULT_OUTPUT_CSV = "data/ground_truth/ground_truth.csv"


def label_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return [], False

    display = img.copy()
    points = []
    boxes = []

    def click_event(event, x, y, flags, param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(display, (x, y), 4, (0, 0, 255), -1)
            if len(points) == 2:
                cv2.rectangle(display, points[0], points[1], (0, 255, 0), 2)
                boxes.append((points[0][0], points[0][1], points[1][0], points[1][1]))
                points.clear()
            cv2.imshow(window_name, display)

    window_name = f"Label people: {os.path.basename(image_path)} " \
                  f"(click top-left then bottom-right per person, 'n'=next, 's'=skip/none, 'q'=quit)"
    cv2.imshow(window_name, display)
    cv2.setMouseCallback(window_name, click_event)

    quit_early = False
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):
            break
        elif key == ord('s'):
            boxes.clear()
            break
        elif key == ord('q'):
            quit_early = True
            break

    cv2.destroyAllWindows()
    return boxes, quit_early


def main():
    output_csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_CSV
    if not output_csv.startswith("data/ground_truth/") and "/" not in output_csv and "\\" not in output_csv:
        output_csv = os.path.join("data/ground_truth", output_csv)

    if not os.path.isdir(IMAGES_DIR):
        print(f"'{IMAGES_DIR}' not found. Run extract_frames_for_labeling.py first, "
              f"or make sure the images you want to label are placed there.")
        return

    images = sorted([f for f in os.listdir(IMAGES_DIR)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not images:
        print(f"No images found in {IMAGES_DIR}")
        return

    print(f"Labeling {len(images)} image(s) from {IMAGES_DIR}")
    print(f"Output will be saved to: {output_csv}\n")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    rows = []

    for image_file in images:
        image_path = os.path.join(IMAGES_DIR, image_file)
        boxes, quit_early = label_image(image_path)

        if boxes:
            for box in boxes:
                rows.append([image_file] + list(box))
        else:
            rows.append([image_file, "", "", "", ""])  # explicitly "0 people" image

        print(f"{image_file}: {len(boxes)} box(es) labeled")

        if quit_early:
            print("Stopped early — saving what's labeled so far.")
            break

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} labeled rows to {output_csv}")
    print("This is what evaluate_metrics.py will compare against the model's own detections.")


if __name__ == "__main__":
    main()
"""
Threshold calibration

Instead of guessing MATCH_THRESHOLD, this scores a known-identity clip
against the enrolled database and reports the actual score distribution —
so you can see where genuine matches and non-matches naturally separate,
rather than picking a number blind.
Usage:
    python backend/services/calibrate_threshold.py \
        data/detection_tracking_clips/single_person_01.mp4 <name>

The second argument is the TRUE identity of the person in that clip (must
match a name already enrolled via extract_embeddings.py).
"""

import os
import sys
import statistics
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "recognition"))
from enrollment_and_matching import load_all_enrolled_embeddings, cosine_similarity

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config

import torch  # noqa: F401 — see recognition/face_test.py for why this must
              # come before the insightface import
from insightface.app import FaceAnalysis


def main():
    if len(sys.argv) < 3:
        print("Usage: python calibrate_threshold.py <video_path> <true_identity_name>")
        return

    video_path, true_name = sys.argv[1], sys.argv[2]
    enrolled = load_all_enrolled_embeddings()

    true_user_vectors = None
    for user_id, data in enrolled.items():
        if data["name"] == true_name:
            true_user_vectors = data["vectors"]
    if true_user_vectors is None:
        print(f"'{true_name}' not found in enrolled users. Check the name "
              f"matches exactly what you enrolled.")
        return

    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=0, det_size=config.DET_SIZE)

    cap = cv2.VideoCapture(video_path)
    genuine_scores = []   # this person's face vs THEIR OWN enrolled vectors
    impostor_scores = []  # this person's face vs EVERYONE ELSE's enrolled vectors

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = face_app.get(frame)
        if len(faces) == 1:
            emb = faces[0].normed_embedding
            best_genuine = max(cosine_similarity(emb, v) for v in true_user_vectors)
            genuine_scores.append(best_genuine)

            for user_id, data in enrolled.items():
                if data["name"] != true_name:
                    best_impostor = max(cosine_similarity(emb, v) for v in data["vectors"])
                    impostor_scores.append(best_impostor)
        frame_idx += 1

    cap.release()

    if not genuine_scores:
        print("No usable face frames found in this clip.")
        return

    print(f"\nProcessed {frame_idx} frames, {len(genuine_scores)} had a clean face.\n")

    print(f"GENUINE match scores ({true_name} vs their own enrolled embeddings):")
    print(f"  min={min(genuine_scores):.3f}  max={max(genuine_scores):.3f}  "
          f"mean={statistics.mean(genuine_scores):.3f}  "
          f"median={statistics.median(genuine_scores):.3f}")

    if impostor_scores:
        print(f"\nIMPOSTOR scores ({true_name}'s face vs everyone else enrolled):")
        print(f"  min={min(impostor_scores):.3f}  max={max(impostor_scores):.3f}  "
              f"mean={statistics.mean(impostor_scores):.3f}  "
              f"median={statistics.median(impostor_scores):.3f}")

        naive_suggestion = (min(genuine_scores) + max(impostor_scores)) / 2
        print(f"\nNaive suggestion (midpoint of absolute min/max): {naive_suggestion:.3f}")
        print(f"  WARNING: this is sensitive to a single outlier frame in "
              f"either direction (one bad-angle genuine frame, or one unusual "
              f"impostor frame) — don't apply it directly without checking "
              f"the robust suggestion below.")

        genuine_sorted = sorted(genuine_scores)
        impostor_sorted = sorted(impostor_scores)
        genuine_p5 = genuine_sorted[max(int(len(genuine_sorted) * 0.05), 0)]
        impostor_p95 = impostor_sorted[min(int(len(impostor_sorted) * 0.95), len(impostor_sorted) - 1)]
        robust_suggestion = (genuine_p5 + impostor_p95) / 2
        print(f"\nRobust suggestion (midpoint of 5th-percentile genuine and "
              f"95th-percentile impostor, ignoring extreme outliers): "
              f"{robust_suggestion:.3f}")
        print(f"  genuine 5th percentile: {genuine_p5:.3f}   "
              f"impostor 95th percentile: {impostor_p95:.3f}")

        print(f"\nCurrent config.MATCH_THRESHOLD is {config.MATCH_THRESHOLD} — "
              f"compare against the ROBUST suggestion, not the naive one. Run "
              f"this on a few different people's clips before settling on "
              f"one final threshold for everyone.")
    else:
        print("\nOnly one person enrolled — can't compute impostor scores yet. "
              "Enroll at least one more person to get a meaningful threshold "
              "suggestion.")


if __name__ == "__main__":
    main()
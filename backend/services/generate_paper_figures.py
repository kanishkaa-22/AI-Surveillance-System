"""
Generates the three visual figures needed for the paper, as clean
standalone PNG images (these are meant to be print-quality and inserted directly into the paper document).

Run from repo root: python backend/services/generate_paper_figures.py

Outputs (in data/paper_figures/):
    fig2_threshold_calibration.png
    fig3_reid_comparison.png
    fig4_movement_heatmap.png
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "recognition"))
from enrollment_and_matching import load_all_enrolled_embeddings, cosine_similarity

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config

import torch  # noqa: F401 — must import before insightface, see recognition/face_test.py
from insightface.app import FaceAnalysis
import cv2

OUTPUT_DIR = os.path.normpath("data/paper_figures_1")
DB_PATH = "backend/database/surveillance.db"
ID_SWITCH_LOG = "data/track_outputs/id_switch_log.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def fig2_threshold_calibration(clip_name_pairs):
    """
    clip_name_pairs: list of (video_path, true_identity_name) tuples —
    one per enrolled person you want included, e.g.:
        [("data/detection_tracking_clips/single_person_01.mp4", "kanishkaa"),
         ("data/detection_tracking_clips/single_person_02.mp4", "evana"), ...]
    """
    enrolled = load_all_enrolled_embeddings()
    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=0, det_size=config.DET_SIZE)

    all_genuine, all_impostor, labels = [], [], []

    for video_path, true_name in clip_name_pairs:
        true_vectors = None
        for uid, data in enrolled.items():
            if data["name"] == true_name:
                true_vectors = data["vectors"]
        if true_vectors is None:
            print(f"Skipping {true_name} — not found in enrolled users.")
            continue

        cap = cv2.VideoCapture(video_path)
        genuine, impostor = [], []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            faces = face_app.get(frame)
            if len(faces) == 1:
                emb = faces[0].normed_embedding
                genuine.append(max(cosine_similarity(emb, v) for v in true_vectors))
                for uid, data in enrolled.items():
                    if data["name"] != true_name:
                        impostor.append(max(cosine_similarity(emb, v) for v in data["vectors"]))
        cap.release()

        all_genuine.append(genuine)
        all_impostor.append(impostor)
        labels.append(true_name)
        print(f"{true_name}: {len(genuine)} genuine samples, {len(impostor)} impostor samples")

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(labels)) * 2
    bp1 = ax.boxplot(all_genuine, positions=positions - 0.35, widths=0.6,
                      patch_artist=True, boxprops=dict(facecolor="#2E9E6B"))
    bp2 = ax.boxplot(all_impostor, positions=positions + 0.35, widths=0.6,
                      patch_artist=True, boxprops=dict(facecolor="#C0392B"))
    ax.axhline(y=config.MATCH_THRESHOLD, color="#1F3864", linestyle="--",
               label=f"MATCH_THRESHOLD = {config.MATCH_THRESHOLD}")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cosine similarity score")
    ax.set_title("Genuine vs. Impostor Score Distribution by Enrolled Identity")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0], ax.lines[-1]],
              ["Genuine", "Impostor", f"Threshold ({config.MATCH_THRESHOLD})"], loc="upper right")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig2_threshold_calibration.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def fig3_reid_comparison():
    if not os.path.exists(ID_SWITCH_LOG):
        print("No id_switch_log.csv found — run main.py on your reid clips first.")
        return
    df = pd.read_csv(ID_SWITCH_LOG)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: match rate by method (histogram vs deep)
    hist_matches = df["hist_match_id"].notna().sum()
    deep_matches = df["deep_match_id"].notna().sum()
    total = len(df)
    axes[0].bar(["Histogram", "Deep Embedding"], [hist_matches / total * 100, deep_matches / total * 100],
                color=["#2E75B6", "#2E9E6B"])
    axes[0].set_ylabel("Match rate (%)")
    axes[0].set_title(f"Re-ID Method Comparison (n={total} events)")
    axes[0].set_ylim(0, 100)

    # Right: ResNet18 vs OSNet score distribution, if deep_backend column exists
    if "deep_backend" in df.columns:
        backends = df["deep_backend"].dropna().unique()
        data_by_backend = [df[df["deep_backend"] == b]["deep_score"].replace(-1, np.nan).dropna() for b in backends]
        axes[1].boxplot(data_by_backend, tick_labels=list(backends), patch_artist=True,
                         boxprops=dict(facecolor="#E8A33D"))
        axes[1].axhline(y=config.REID_DEEP_THRESHOLD, color="#1F3864", linestyle="--",
                         label=f"Threshold ({config.REID_DEEP_THRESHOLD})")
        axes[1].set_ylabel("Deep embedding score")
        axes[1].set_title("Deep Backbone Comparison")
        axes[1].legend()
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No deep_backend column\n(older log format)",
                      ha="center", va="center")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig3_reid_comparison.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def fig4_movement_heatmap():
    if not os.path.exists(DB_PATH):
        print("Database not found.")
        return
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT cx, cy FROM position_log", conn)
    conn.close()

    df["cx"] = pd.to_numeric(df["cx"], errors="coerce")
    df["cy"] = pd.to_numeric(df["cy"], errors="coerce")
    df = df.dropna()

    if df.empty or df["cx"].nunique() <= 1:
        print("position_log has no valid spread of data yet — re-run main.py after "
              "confirming the cx/cy float-cast fix, then try again.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    h = ax.hist2d(df["cx"], df["cy"], bins=40, cmap="YlOrRd")
    fig.colorbar(h[3], ax=ax, label="Dwell density (position samples)")
    ax.invert_yaxis()  # image coordinates: y grows downward
    ax.set_xlabel("Horizontal position (pixels)")
    ax.set_ylabel("Vertical position (pixels)")
    ax.set_title("Dwell-Time and Movement Heatmap")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig4_movement_heatmap.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Generating paper figures...")
    print("=" * 60)

    # EDIT THIS LIST — one (clip, enrolled_name) pair per person you want in Fig 2
    calibration_clips = [
        #("data/detection_tracking_clips/single_person_01.mp4", "satya"),
        ("data/detection_tracking_clips/single_person_02.mp4", "kanishkaa"),
        # add more enrolled people here
    ]

    try:
        fig2_threshold_calibration(calibration_clips)
    except Exception as e:
        print(f"\nFig 2 FAILED, continuing to Fig 3/4 anyway: {e}")

    try:
        fig3_reid_comparison()
    except Exception as e:
        print(f"\nFig 3 FAILED, continuing to Fig 4 anyway: {e}")
        print("Common cause on Windows: the previous fig3_reid_comparison.png "
              "is open in an image viewer or Explorer preview pane, locking "
              "the file. Close any program showing it and re-run.")

    try:
        fig4_movement_heatmap()
    except Exception as e:
        print(f"\nFig 4 FAILED: {e}")

    print(f"\nAll figures saved to {OUTPUT_DIR}/")
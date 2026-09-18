"""
Enrollment flow (load .npy embeddings into DB) + matching logic.

Prerequisite: recognition/extract_embeddings.py has already been run, producing
recognition/embeddings/<person_name>.npy files (one array per person, shape
(num_kept_frames, 512)).
"""

import os
import sqlite3
import numpy as np

DB_PATH = "backend/database/surveillance.db"
EMBEDDINGS_DIR = "recognition/embeddings"
MATCH_THRESHOLD = 0.45   # cosine similarity threshold — tune this against your own test data


def load_embeddings_to_db():
    """One-time enrollment step: read each person's .npy file, insert into users + embeddings."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    npy_files = [f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".npy")]
    if not npy_files:
        print(f"No .npy files found in {EMBEDDINGS_DIR} — run extract_embeddings.py first.")
        return

    for npy_file in npy_files:
        person_name = os.path.splitext(npy_file)[0]
        embeddings = np.load(os.path.join(EMBEDDINGS_DIR, npy_file))

        cur.execute(
            "INSERT OR IGNORE INTO users (name) VALUES (?)", (person_name,)
        )
        cur.execute("SELECT user_id FROM users WHERE name = ?", (person_name,))
        user_id = cur.fetchone()[0]

        # Remove any previously loaded embeddings for this user (avoids duplicates on re-run)
        cur.execute("DELETE FROM embeddings WHERE user_id = ?", (user_id,))

        for i, emb in enumerate(embeddings):
            cur.execute(
                "INSERT INTO embeddings (user_id, vector, source_frame) VALUES (?, ?, ?)",
                (user_id, emb.astype(np.float32).tobytes(), f"frame_{i}"),
            )

        print(f"Loaded {len(embeddings)} embeddings for '{person_name}' (user_id={user_id})")

    conn.commit()
    conn.close()


def load_all_enrolled_embeddings():
    """Fetch every enrolled embedding back out as {user_id: name, vectors: [...]}."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT users.user_id, users.name, embeddings.vector
        FROM embeddings
        JOIN users ON users.user_id = embeddings.user_id
    """)
    rows = cur.fetchall()
    conn.close()

    enrolled = {}  # user_id -> {"name": str, "vectors": [np.array, ...]}
    for user_id, name, vector_blob in rows:
        vec = np.frombuffer(vector_blob, dtype=np.float32)
        enrolled.setdefault(user_id, {"name": name, "vectors": []})
        enrolled[user_id]["vectors"].append(vec)
    return enrolled


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def match_face(query_embedding, enrolled=None, threshold=MATCH_THRESHOLD):
    """
    Compare a single query embedding (512,) against all enrolled people.
    Returns (name, confidence) if matched above threshold, else ("Unknown", best_score).
    Each enrolled person has multiple stored embeddings (different angles) —
    we take the BEST match across all of a person's stored vectors.
    """
    if enrolled is None:
        enrolled = load_all_enrolled_embeddings()

    best_name = None
    best_score = -1.0

    for user_id, data in enrolled.items():
        for vec in data["vectors"]:
            score = cosine_similarity(query_embedding, vec)
            if score > best_score:
                best_score = score
                best_name = data["name"]

    if best_score >= threshold:
        return best_name, best_score
    return "Unknown", best_score


if __name__ == "__main__":
    print("Loading embeddings into database ...")
    load_embeddings_to_db()

    print("\nRunning a quick self-test: matching each enrolled person's own "
          "first embedding against the full database (should match themselves).")
    enrolled = load_all_enrolled_embeddings()
    for user_id, data in enrolled.items():
        test_vec = data["vectors"][0]
        name, score = match_face(test_vec, enrolled)
        status = "OK" if name == data["name"] else "MISMATCH"
        print(f"  {data['name']}: matched as '{name}' (score={score:.3f}) [{status}]")
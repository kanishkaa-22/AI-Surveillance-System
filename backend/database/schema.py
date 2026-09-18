"""
Database schema.

Safe to re-run — uses CREATE TABLE IF NOT EXISTS, so running this again
after a schema tweak won't wipe existing data unless you drop tables
yourself first.
"""

import sqlite3
import os

DB_PATH = "backend/database/surveillance.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    enrolled_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS embeddings (
    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL,
    vector       BLOB NOT NULL,        -- 512-float embedding, stored as bytes
    source_frame TEXT,                 -- which enrollment frame this came from
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS zones (
    zone_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT NOT NULL,           -- e.g. "main_entry", "restricted_lab"
    zone_type TEXT NOT NULL,           -- "line" or "polygon"
    coords    TEXT NOT NULL            -- JSON-encoded coordinates
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    ended_at   TEXT,
    video_source TEXT                  -- clip filename or "live_webcam"
);

CREATE TABLE IF NOT EXISTS track_history (
    track_history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL,
    track_id    INTEGER NOT NULL,      -- ByteTrack's ID, unique only within a session
    user_id     INTEGER,               -- filled in once/if recognized; NULL = unidentified
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL,
    track_id    INTEGER NOT NULL,
    user_id     INTEGER,               -- NULL if unrecognized ("Unknown")
    zone_id     INTEGER,
    event_type  TEXT NOT NULL,         -- "entry", "exit", "restricted_access", "unknown_alert", "reid_relink", "periodic_check"
    confidence  REAL,
    timestamp   TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (zone_id) REFERENCES zones(zone_id)
);

-- NEW: position_log powers the dwell-time heatmap / movement analytics
-- (Novelty feature — see dashboard Analytics tab). Written periodically by
-- main.py, NOT every single frame, to keep the table size reasonable.
CREATE TABLE IF NOT EXISTS position_log (
    position_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL,
    track_id    INTEGER NOT NULL,
    user_id     INTEGER,               -- NULL if not yet recognized at this moment
    cx          REAL NOT NULL,         -- centroid x (pixels, source frame coordinates)
    cy          REAL NOT NULL,         -- centroid y
    timestamp   TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
"""


def main():
    os.makedirs("backend", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.commit()

    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    conn.close()

    print(f"Database ready at {DB_PATH}")
    print("Tables:", [t[0] for t in tables])


if __name__ == "__main__":
    main()
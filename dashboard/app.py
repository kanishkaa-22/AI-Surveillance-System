import os
import time
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

DB_PATH = "backend/database/surveillance.db"
LATEST_FRAME_PATH = "dashboard/latest_frame.jpg"
ID_SWITCH_LOG_PATH = "data/track_outputs/id_switch_log.csv"
OCCUPANCY_WINDOW_SECONDS = 10

st.set_page_config(page_title="Surveillance Analytics", layout="wide", page_icon="🎥")

st.markdown("""
<style>
    .main { background-color: #F7F9FC; }
    div[data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #E4E9F2; border-radius: 12px;
        padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetricLabel"] { color: #5B6472; font-size: 0.85rem; }
    div[data-testid="stMetricValue"] { color: #1F3864; font-weight: 700; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #1F3864; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    div[data-testid="stDataFrame"] { border: 1px solid #E4E9F2; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("AI Surveillance & Identity Analytics")
st.caption("Live monitoring, recognition, re-identification, and movement analytics — real-time, GPU-accelerated.")


@st.cache_data(ttl=2)
def load_data():
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    events_df = pd.read_sql_query("""
        SELECT events.event_id, events.timestamp, events.event_type,
               events.confidence, COALESCE(users.name, 'Unknown') AS identity,
               events.track_id, zones.name AS zone
        FROM events
        LEFT JOIN users ON users.user_id = events.user_id
        LEFT JOIN zones ON zones.zone_id = events.zone_id
        ORDER BY events.timestamp DESC
    """, conn)
    track_df = pd.read_sql_query("""
        SELECT track_history.track_id, COALESCE(users.name, 'Unknown') AS identity,
               track_history.first_seen, track_history.last_seen
        FROM track_history
        LEFT JOIN users ON users.user_id = track_history.user_id
        ORDER BY track_history.last_seen DESC
    """, conn)
    position_df = pd.read_sql_query("""
        SELECT position_log.cx, position_log.cy, position_log.timestamp,
               COALESCE(users.name, 'Unknown') AS identity
        FROM position_log
        LEFT JOIN users ON users.user_id = position_log.user_id
    """, conn)
    users_df = pd.read_sql_query("SELECT name FROM users", conn)
    conn.close()
    return {"events": events_df, "tracks": track_df, "positions": position_df, "users": users_df}


def load_id_switch_log():
    if not os.path.exists(ID_SWITCH_LOG_PATH) or os.path.getsize(ID_SWITCH_LOG_PATH) == 0:
        return None
    try:
        return pd.read_csv(ID_SWITCH_LOG_PATH, on_bad_lines='skip')
    except pd.errors.EmptyDataError:
        return None


data = load_data()
if data is None:
    st.warning("Database not found yet — run backend/database/schema.py and backend/main.py first.")
    st.stop()

events_df, track_df, position_df, users_df = data["events"], data["tracks"], data["positions"], data["users"]
id_switch_df = load_id_switch_log()

col_a, col_b, col_c = st.columns([1, 1, 4])
with col_a:
    if st.button("Refresh now"):
        st.cache_data.clear()
        st.rerun()
with col_b:
    live_auto = st.toggle("Auto-refresh Live tab (3s)", value=False,
                           help="Only the Live Monitor tab refreshes automatically — Analytics/People/Log stay still so charts don't lag. Use 'Refresh now' for those.")

# ---- KPI row (computed once per full load, not per fragment refresh) ----
if track_df is not None and not track_df.empty:
    track_df["last_seen_dt"] = pd.to_datetime(track_df["last_seen"])
    recent_cutoff = pd.Timestamp.now() - pd.Timedelta(seconds=OCCUPANCY_WINDOW_SECONDS)
    currently_present = track_df[track_df["last_seen_dt"] >= recent_cutoff]
    known_people = currently_present[currently_present["identity"] != "Unknown"]["identity"].nunique()
    unknown_tracks = currently_present[currently_present["identity"] == "Unknown"]["track_id"].nunique()
    occupancy_now = known_people + unknown_tracks
else:
    currently_present = pd.DataFrame()
    occupancy_now = 0

total_events = len(events_df) if events_df is not None else 0
enrolled_count = len(users_df) if users_df is not None else 0
avg_conf = events_df["confidence"].dropna().mean() if events_df is not None and not events_df.empty else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("People In View Now", occupancy_now)
k2.metric("Enrolled Users", enrolled_count)
k3.metric("Total Events Logged", total_events)
k4.metric("Avg. Recognition Confidence", f"{avg_conf:.2f}" if avg_conf else "—")

st.divider()

tab_live, tab_analytics, tab_people, tab_log = st.tabs(
    ["📹 Live Monitor", "Analytics", "People", "Event Log"]
)

# ---------------------------------------------------------------------------
# LIVE MONITOR — the only tab that auto-refreshes, and only itself


@st.fragment(run_every="3s" if live_auto else None)
def live_monitor_fragment():
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Live Feed")
        if os.path.exists(LATEST_FRAME_PATH) and os.path.getsize(LATEST_FRAME_PATH) > 0:
            try:
                with Image.open(LATEST_FRAME_PATH) as img:
                    img.verify()
                st.image(LATEST_FRAME_PATH, use_container_width=True)
            except Exception:
                st.info("Syncing frame feed...")
        else:
            st.info("No live frame yet — start backend/main.py to see the feed.")
    with right:
        st.subheader("Currently Present")
        if not currently_present.empty:
            st.dataframe(currently_present[["track_id", "identity", "last_seen"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("No one currently in view.")

        st.subheader("Recent Events")
        if events_df is not None and not events_df.empty:
            st.dataframe(events_df.head(8)[["timestamp", "identity", "event_type", "zone"]],
                         use_container_width=True, hide_index=True)


with tab_live:
    live_monitor_fragment()

with tab_analytics:
    st.subheader("Occupancy Over Time")
    if not events_df.empty:
        occ_df = events_df.copy()
        occ_df["timestamp"] = pd.to_datetime(occ_df["timestamp"])
        occ_df = occ_df.sort_values("timestamp")
        occ_df["delta"] = occ_df["event_type"].map({"entry": 1, "exit": -1}).fillna(0)
        occ_df["running_occupancy"] = occ_df["delta"].cumsum().clip(lower=0)
        fig = px.line(occ_df, x="timestamp", y="running_occupancy", markers=True)
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                           plot_bgcolor="white", yaxis_title="People in space")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No events yet to chart.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Event Type Breakdown")
        if not events_df.empty:
            counts = events_df["event_type"].value_counts().reset_index()
            counts.columns = ["event_type", "count"]
            fig2 = px.bar(counts, x="event_type", y="count", color="event_type",
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig2.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Recognition Confidence Distribution")
        conf_data = events_df["confidence"].dropna()
        if not conf_data.empty:
            fig3 = px.histogram(conf_data, nbins=20, color_discrete_sequence=["#2E75B6"])
            fig3.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                                xaxis_title="Confidence score", yaxis_title="Count")
            st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Dwell-Time & Movement Heatmap")
    if position_df is not None and not position_df.empty:
        heat_data = position_df.copy()
        heat_data["cx"] = pd.to_numeric(heat_data["cx"], errors="coerce")
        heat_data["cy"] = pd.to_numeric(heat_data["cy"], errors="coerce")
        heat_data = heat_data.dropna(subset=["cx", "cy"])

        if heat_data.empty or heat_data["cx"].nunique() <= 1:
            st.warning(
                "Position data exists but looks invalid or has no spread (all one value). "
                "This means the rows currently in position_log predate the cx/cy float-cast "
                "fix in backend/main.py. Clear the table and re-run main.py on a fresh clip: "
                "`DELETE FROM position_log` then re-run, before this chart will show real data."
            )
        else:
            fig_heat = px.density_heatmap(
                heat_data, x="cx", y="cy", nbinsx=30, nbinsy=30,
                color_continuous_scale="Viridis",
                labels={"cx": "Horizontal Position (X)", "cy": "Vertical Position (Y)"},
            )
            fig_heat.update_yaxes(autorange="reversed")
            fig_heat.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No position data logged yet — this populates as backend/main.py runs.")

    st.subheader("Re-Identification: Histogram vs. Deep Embedding")
    if id_switch_df is not None and not id_switch_df.empty:
        hist_matches = id_switch_df["hist_match_id"].notna().sum()
        deep_matches = id_switch_df["deep_match_id"].notna().sum()
        total = len(id_switch_df)
        comp_df = pd.DataFrame({
            "Method": ["Histogram", "Deep Embedding"],
            "Matches Found": [hist_matches, deep_matches],
            "No Match": [total - hist_matches, total - deep_matches],
        })
        fig5 = px.bar(comp_df, x="Method", y=["Matches Found", "No Match"], barmode="group",
                      color_discrete_sequence=["#2E9E6B", "#C7D0DC"])
        fig5.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig5, use_container_width=True)
        st.caption(f"Across {total} track reappearance events.")
    else:
        st.info("No ReID comparison data yet.")

with tab_people:
    st.subheader("Registered Track History")
    if track_df is not None and not track_df.empty:
        st.dataframe(track_df[["track_id", "identity", "first_seen", "last_seen"]],
                     use_container_width=True, hide_index=True)
    else:
        st.info("No recorded tracking histories found.")

with tab_log:
    st.subheader("Full Event Log")
    if events_df is not None and not events_df.empty:
        event_types = ["All"] + sorted(events_df["event_type"].unique().tolist())
        selected_type = st.selectbox("Filter by event type", event_types)
        filtered = events_df if selected_type == "All" else events_df[events_df["event_type"] == selected_type]
        st.dataframe(filtered, use_container_width=True, hide_index=True)
    else:
        st.info("No events logged yet.")
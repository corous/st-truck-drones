# st-dashboard.py
"""Streamlit planning console for truck + 4‑drone delivery teams.
   ‑ real‑time WebSocket telemetry (vehicles)
   ‑ editable stop list
   ‑ Mapbox/Deck.GL rendering
   Requires: streamlit>=1.33, pydeck>=0.8, websockets>=12.0
"""

from __future__ import annotations
import asyncio, json, os, threading
import requests
import sseclient

import streamlit as st
import pydeck as pdk

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
MAPBOX_TOKEN = (
    st.secrets.get("mapbox_token")            # Streamlit secrets
    or os.getenv("MAPBOX_TOKEN")               # or env‑var for local dev / CI
)

if not MAPBOX_TOKEN:
    st.error(
        "Missing Mapbox token. Add it to .streamlit/secrets.toml (mapbox_token) "
        "or set the MAPBOX_TOKEN environment variable."
    )
    st.stop()

pdk.settings.mapbox_api_key = MAPBOX_TOKEN

SSE_URL = os.getenv("SSE_URL", "https://render-vehicles.onrender.com/stream")  # Render deployment

ICON_URLS = {
    "truck": "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas/cars.png",
    "drone": "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas/rocket.png",
}

INITIAL_STOPS = [
    {"id": "A", "address": "Warehouse, 123 Main St", "lat": 33.69321, "lon": -117.83345},
]

# ---------------------------------------------------------------------------
# 2. SESSION STATE BOOTSTRAP
# ---------------------------------------------------------------------------
if "vehicles" not in st.session_state:
    st.session_state["vehicles"] = {}
if "ws_task" not in st.session_state:
    st.session_state["ws_task"] = None

# ---------------------------------------------------------------------------
# 3. SSE LISTENER
# ---------------------------------------------------------------------------
def listener() -> None:
    """Connects to SSE_URL, parses each JSON line into st.session_state['vehicles']."""
    client = sseclient.SSEClient(SSE_URL)
    for event in client.events():
        try:
            vehicle: dict = json.loads(event.data)
            st.session_state["vehicles"][vehicle["id"]] = vehicle
            # flip a bool so Streamlit detects a state change and reruns
            st.session_state["_ping"] = not st.session_state.get("_ping", False)
        except json.JSONDecodeError:
            continue

def ensure_listener_task() -> None:
    """Guarantees a background thread is running."""
    if st.session_state["ws_task"]:
        return  # already spawned

    t = threading.Thread(
        target=listener,
        daemon=True,
        name="sse-listener",
    )
    t.start()
    st.session_state["ws_task"] = t

ensure_listener_task()

# ---------------------------------------------------------------------------
# 4. UI LAYOUT
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Delivery Planner", layout="wide")

st.title("🚚 Truck + 🚁 Drone Delivery Planner")

left, right = st.columns([3, 1])

# ---- 4a. Map pane ----------------------------------------------------------
with left:
    stops_layer = pdk.Layer(
        "ScatterplotLayer",
        data=INITIAL_STOPS,
        get_position="[lon, lat]",
        get_radius=60,
        get_fill_color=[255, 0, 0],
    )

    veh_layer = pdk.Layer(
        "IconLayer",
        data=list(st.session_state["vehicles"].values()),
        get_position="[lon, lat]",
        get_icon="type",
        get_size=4,
        size_scale=15,
        get_rotation="[heading]",
        icon_mapping={
            k: {"url": url, "width": 128, "height": 128, "anchorY": 128}
            for k, url in ICON_URLS.items()
        },
    )

    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/streets-v12",
        initial_view_state=pdk.ViewState(
            latitude=33.6846, longitude=-117.8265, zoom=12,
        ),
        layers=[stops_layer, veh_layer],
    )

    st.pydeck_chart(deck, use_container_width=True)

# ---- 4b. Stop list + controls ---------------------------------------------
with right:
    st.subheader("Stops (drag to edit)")

    # Convert to & from editor‑friendly dicts
    if "stops" not in st.session_state:
        st.session_state["stops"] = INITIAL_STOPS.copy()

    st.session_state["stops"] = st.data_editor(
        st.session_state["stops"],
        num_rows="dynamic",
        use_container_width=True,
        key="stops_editor",
    )

    col1, col2, col3 = st.columns(3)

    # -- Import
    uploaded_file = st.file_uploader("Choose CSV", type=["csv"], key="csv_uploader")
    if uploaded_file is not None:
        try:
            import pandas as pd
            df = pd.read_csv(uploaded_file)
            # Convert numeric columns to float
            df['lat'] = df['lat'].astype(float)
            df['lon'] = df['lon'].astype(float)
            # Convert to list of dicts and update session state
            st.session_state["stops"] = df.to_dict("records")
            st.success(f"Successfully imported {len(df)} stops")
            # Force a rerun to update the display
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error importing CSV: {str(e)}")

    # -- Commit (stub)
    if col2.button("Commit"):
        st.success("Route committed (placeholder – call your FastAPI here)")

    # -- Reset vehicles
    if col3.button("Reset Vehicles"):
        st.session_state["vehicles"].clear()
        st.session_state["_ping"] = not st.session_state.get("_ping", False)
        st.info("Vehicle positions cleared.")

# ---------------------------------------------------------------------------
# End of file
# ---------------------------------------------------------------------------

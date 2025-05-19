# st-dashboard.py
"""Streamlit planning console for truck + 4‑drone delivery teams.
   ‑ real‑time WebSocket telemetry (vehicles)
   ‑ editable stop list
   ‑ Mapbox/Deck.GL rendering
   Requires: streamlit>=1.33, pydeck>=0.8, websockets>=12.0
"""

from __future__ import annotations
import asyncio, json, os, threading

import streamlit as st
import pydeck as pdk

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
MAPBOX_TOKEN = (
    st.secrets.get("mapbox_token")            # Streamlit secrets
    or os.getenv("MAPBOX_TOKEN")               # or env‑var for local dev / CI
)

if not MAPBOX_TOKEN:
    st.error(
        "Missing Mapbox token. Add it to .streamlit/secrets.toml (mapbox_token) "
        "or set the MAPBOX_TOKEN environment variable."
    )
    st.stop()

pdk.settings.mapbox_api_key = MAPBOX_TOKEN

WS_URL = os.getenv("WS_URL", "wss://backend.example.com/telemetry")  # TODO: change

ICON_URLS = {
    "truck": "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas/cars.png",
    "drone": "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas/rocket.png",
}

INITIAL_STOPS = [
    {"id": "A", "address": "123 Main St", "lat": 33.6846, "lon": -117.8265},
    {"id": "B", "address": "456 Oak Ave", "lat": 33.6890, "lon": -117.8000},
]

# ---------------------------------------------------------------------------
# 2. SESSION STATE BOOTSTRAP
# ---------------------------------------------------------------------------
if "vehicles" not in st.session_state:
    st.session_state["vehicles"] = {}
if "ws_task" not in st.session_state:
    st.session_state["ws_task"] = None

# ---------------------------------------------------------------------------
# 3. ASYNC WEBSOCKET LISTENER
# ---------------------------------------------------------------------------
async def listener() -> None:
    """Connects to WS_URL, parses each JSON line into st.session_state['vehicles']."""
    import websockets  # local import so requirements are clear

    async with websockets.connect(WS_URL) as ws:
        async for message in ws:
            try:
                vehicle: dict = json.loads(message)
                st.session_state["vehicles"][vehicle["id"]] = vehicle
                # flip a bool so Streamlit detects a state change and reruns
                st.session_state["_ping"] = not st.session_state.get("_ping", False)
            except json.JSONDecodeError:
                continue


def ensure_ws_task() -> None:
    """Guarantees a background thread with its own event‑loop is running."""
    if st.session_state["ws_task"]:
        return  # already spawned

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    t = threading.Thread(
        target=loop.run_until_complete,
        args=(listener(),),
        daemon=True,
        name="ws-listener",
    )
    t.start()
    st.session_state["ws_task"] = t


ensure_ws_task()

# ---------------------------------------------------------------------------
# 4. UI LAYOUT
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Delivery Planner", layout="wide")

st.title("🚚 Truck + 🚁 Drone Delivery Planner")

left, right = st.columns([3, 1])

# ---- 4a. Map pane ----------------------------------------------------------
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

# ---- 4b. Stop list + controls ---------------------------------------------
with right:
    st.subheader("Stops (drag / edit)")

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
    if col1.button("Import CSV"):
        up = st.file_uploader("Choose CSV", type=["csv"])
        if up:
            import pandas as pd
            df = pd.read_csv(up)
            st.session_state["stops"] = df.to_dict("records")
            st.success("Imported {} stops".format(len(df)))

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

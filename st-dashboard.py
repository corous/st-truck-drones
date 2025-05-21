# st-dashboard.py
"""Streamlit delivery‑planner dashboard
   • Real‑time vehicle telemetry via SSE
   • CSV‑driven stop list with drag‑and‑drop
   • On *Commit*:
       1. markers for stops are added/updated
       2. a naïve route is planned for one truck (blue line) and four drones (purple lines)
       3. truck/drone routes rendered with PathLayer

   Requires: streamlit>=1.33, pydeck>=0.8, sseclient‑py>=0.6
"""

from __future__ import annotations
import json, os, threading, itertools, random

import streamlit as st
import pydeck as pdk
import pandas as pd
import requests, sseclient  # telemetry

# ---------------------------------------------------------------------------
# 1. CONFIGURATION & GLOBALS
# ---------------------------------------------------------------------------
MAPBOX_TOKEN = (
    st.secrets.get("mapbox_token") or os.getenv("MAPBOX_TOKEN")
)
if not MAPBOX_TOKEN:
    st.error("Missing Mapbox token – set mapbox_token in secrets or env var.")
    st.stop()

pdk.settings.mapbox_api_key = MAPBOX_TOKEN

SSE_URL = os.getenv("SSE_URL", "https://render-vehicles.onrender.com/stream")

ICON_URLS = {
    "truck": "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas/cars.png",
    "drone": "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas/rocket.png",
}

# Default single warehouse placeholder – will be overridden by CSV / editor
INITIAL_STOPS = [
    {"id": "WH", "address": "Warehouse", "lat": 33.69321, "lon": -117.83345},
]

# ---------------------------------------------------------------------------
# 2. SESSION STATE INITIALISATION
# ---------------------------------------------------------------------------
state_defaults = {
    "vehicles": {},     # live telemetry
    "stops": INITIAL_STOPS.copy(),
    "routes": None,     # {'truck': [...], 'drones': [[...], ...]}
    "_listener": None,  # background thread reference
}
for k, v in state_defaults.items():
    st.session_state.setdefault(k, v)

# ---------------------------------------------------------------------------
# 3.  TELEMETRY LISTENER (SSE → session_state['vehicles'])
# ---------------------------------------------------------------------------

def _sse_listener(url: str) -> None:
    """Background thread: consume Server‑Sent Events and update vehicle state.
    Uses a manual `requests` stream + sseclient parser so we can force UTF‑8
    decoding (avoids the bytes/str concatenation error on Python 3.13).
    Re‑connects automatically on any network or parsing error.
    """
    import time, requests, sseclient

    while True:  # auto‑reconnect loop
        try:
            with requests.get(url, stream=True, timeout=30) as resp:
                resp.raise_for_status()
                resp.encoding = "utf-8"          # make iter_lines yield str
                client = sseclient.SSEClient(resp)

                for event in client.events():
                    if not event.data:
                        continue
                    try:
                        data = json.loads(event.data)
                        st.session_state["vehicles"][data["id"]] = data
                        # trigger Streamlit rerun
                        st.session_state["_ping"] = not st.session_state.get("_ping", False)
                    except (json.JSONDecodeError, KeyError):
                        continue  # skip malformed payloads
        except Exception as err:
            # Log then back‑off before retrying
            print("[SSE] listener error:", err)
            time.sleep(5)
            continue

# Spawn the listener exactly once per session
if st.session_state["_listener"] is None:
    t = threading.Thread(target=_sse_listener, args=(SSE_URL,), daemon=True)
    t.start()
    st.session_state["_listener"] = t
    t = threading.Thread(target=_sse_listener, args=(SSE_URL,), daemon=True)
    t.start()
    st.session_state["_listener"] = t

# ---------------------------------------------------------------------------
# 4.  PAGE LAYOUT
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Delivery Planner", layout="wide")
st.title("🚚 Truck + 🚁 Drone Delivery Planner")
left, right = st.columns([3, 1])

# ---------------------------- 4A  MAP PANEL ------------------------------- #
with left:
    # Scatterplot for stops (red)
    stops_layer = pdk.Layer(
        "ScatterplotLayer",
        data=st.session_state["stops"],
        get_position="[lon, lat]",
        get_radius=70,
        get_fill_color=[255, 0, 0],
    )

    # Icon layer for live truck/drone positions
    veh_layer = pdk.Layer(
        "IconLayer",
        data=list(st.session_state["vehicles"].values()),
        get_position="[lon, lat]",
        get_icon="type",
        icon_mapping={
            k: {"url": url, "width": 128, "height": 128, "anchorY": 128}
            for k, url in ICON_URLS.items()
        },
        get_size=4,
        size_scale=15,
        get_rotation="[heading]",
    )

    layers = [stops_layer, veh_layer]

    # Path layers if routes have been planned
    if st.session_state["routes"]:
        truck_path = pdk.Layer(
            "PathLayer",
            data=[{"path": st.session_state["routes"]["truck"]}],
            get_path="path",
            get_color=[0, 0, 255],         # blue
            width_scale=20,
            width_min_pixels=3,
        )
        drone_paths = pdk.Layer(
            "PathLayer",
            data=[{"path": p} for p in st.session_state["routes"]["drones"]],
            get_path="path",
            get_color=[128, 0, 128],       # purple
            width_scale=12,
            width_min_pixels=2,
        )
        layers.extend([truck_path, drone_paths])

    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/streets-v12",
        initial_view_state=pdk.ViewState(
            latitude=st.session_state["stops"][0]["lat"],
            longitude=st.session_state["stops"][0]["lon"],
            zoom=12,
        ),
        layers=layers,
    )
    st.pydeck_chart(deck, use_container_width=True)

# ------------------------ 4B  STOP LIST + CONTROLS ------------------------ #
with right:
    st.subheader("Stops (drag to edit or CSV import)")

    # Editable grid
    st.session_state["stops"] = st.data_editor(
        st.session_state["stops"],
        num_rows="dynamic",
        use_container_width=True,
        key="stops_editor",
    )

    # File uploader
    uploaded = st.file_uploader("Import CSV", type=["csv"], key="csv_uploader")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            df["lat"] = df["lat"].astype(float)
            df["lon"] = df["lon"].astype(float)
            st.session_state["stops"] = df.to_dict("records")
            st.success(f"Imported {len(df)} stops from CSV")
            st.rerun()
        except Exception as exc:
            st.error(f"CSV import failed: {exc}")

    col1, col2, col3 = st.columns(3)

    # ---------------- Commit button -----------------
    def plan_routes():
        stops = st.session_state["stops"]
        if len(stops) < 2:
            st.warning("Need at least two stops (warehouse + 1 delivery) to plan routes.")
            return

        # Warehouse is first entry
        origin = stops[0]
        warehouse_coord = [origin["lon"], origin["lat"]]

        # ---- Truck path: warehouse -> all stops in order -> warehouse ----
        truck_path = [warehouse_coord] + [[s["lon"], s["lat"]] for s in stops[1:]] + [warehouse_coord]

        # ---- Drone paths: round‑robin assignment excluding warehouse ----
        deliveries = stops[1:]
        drone_bins = [[] for _ in range(4)]
        for idx, stop in enumerate(deliveries):
            drone_bins[idx % 4].append(stop)

        drone_paths: list[list[list[float]]] = []
        for bin_ in drone_bins:
            if not bin_:
                continue  # fewer stops than drones
            path = [warehouse_coord] + [[s["lon"], s["lat"]] for s in bin_] + [warehouse_coord]
            drone_paths.append(path)

        st.session_state["routes"] = {"truck": truck_path, "drones": drone_paths}
        st.success("✅ Routes planned and drawn on the map.")
        st.rerun()

    if col2.button("Commit / Plan Route"):
        plan_routes()

    # ---------------- Reset vehicles (telemetry) ----------------
    if col3.button("Reset Vehicles"):
        st.session_state["vehicles"].clear()
        st.session_state["_ping"] = not st.session_state.get("_ping", False)
        st.info("Vehicle positions cleared.")

# ---------------------------------------------------------------------------
# End of file
# ---------------------------------------------------------------------------

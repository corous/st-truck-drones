# st-dashboard.py – full file (2025‑06‑27)
# --------------------------------------------------------------
# Streamlit dashboard for planning + live‑tracking
#  • Mapbox basemap & real‑time vehicle icons via SSE
#  • Truck route now follows actual roads (Mapbox Directions API)
# --------------------------------------------------------------

from __future__ import annotations
import json, os, threading, time, math, requests, polyline
from typing import List

import streamlit as st
import pydeck as pdk
import pandas as pd
import sseclient

# ----------------------- mapbox + telemetry config ----------------------- #
MAPBOX_TOKEN = st.secrets.get("mapbox_token") or os.getenv("MAPBOX_TOKEN")
if not MAPBOX_TOKEN:
    st.error("Missing Mapbox token – set mapbox_token in secrets or env var.")
    st.stop()

MB_DIRECTIONS = "https://api.mapbox.com/directions/v5/mapbox/driving"
SSE_URL = os.getenv("SSE_URL", "https://render-vehicles.onrender.com/stream")
START_DELAY_SEC = 30  # simulator warm‑up delay

pdk.settings.mapbox_api_key = MAPBOX_TOKEN  # fallback for older pydeck

ICON_URLS = {
    "truck": "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas/cars.png",
    "drone": "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas/rocket.png",
}

INITIAL_STOPS = [
    {"id": "WH", "address": "Warehouse", "lat": 33.69321, "lon": -117.83345},
]

# ----------------------- geometry helpers ------------------------------- #
EARTH_MI = 3958.8

def haversine_miles(lat1, lon1, lat2, lon2):
    phi1, phi2 = map(math.radians, (lat1, lat2))
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * EARTH_MI * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ----------------------- road‑aligned path via Mapbox -------------------- #

def road_path(lon1, lat1, lon2, lat2) -> List[List[float]]:
    url = f"{MB_DIRECTIONS}/{lon1},{lat1};{lon2},{lat2}"
    params = {
        "geometries": "polyline6",
        "overview": "full",
        "access_token": MAPBOX_TOKEN,
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    coords = polyline.decode(r.json()["routes"][0]["geometry"], precision=6)
    return [[lon, lat] for lat, lon in coords]

# ----------------------- session‑state defaults -------------------------- #
def_state = dict(
    vehicles={},
    stops=INITIAL_STOPS.copy(),
    routes=None,
    listener=None,
    sse_status="🔄 connecting…",
)
for k, v in def_state.items():
    st.session_state.setdefault(k, v)

# ----------------------- SSE background thread --------------------------- #

def _sse_listener():
    time.sleep(START_DELAY_SEC)  # wait for simulator
    client = sseclient.SSEClient(SSE_URL)
    for evt in client.events():
        try:
            data = json.loads(evt.data)
            st.session_state["vehicles"][data["id"]] = data
            st.session_state["_ping"] = not st.session_state.get("_ping", False)
            st.session_state["sse_status"] = "🟢 live"
        except Exception:
            continue

if st.session_state["listener"] is None:
    t = threading.Thread(target=_sse_listener, daemon=True)
    t.start()
    st.session_state["listener"] = t

# ----------------------- UI layout --------------------------------------- #
st.set_page_config(page_title="Delivery Planner", layout="wide")
st.title("🚚 Truck + 🚁 Drone Delivery Planner")
st.caption(f"SSE feed status: {st.session_state['sse_status']}")
left, right = st.columns([3, 1], gap="small")

# ----------------------- Map panel --------------------------------------- #
with left:
    depot_layer = pdk.Layer(
        "ScatterplotLayer",
        data=[st.session_state["stops"][0]],
        get_position="[lon, lat]",
        get_radius=90,
        get_fill_color=[0, 180, 0],
    )
    stop_layer = pdk.Layer(
        "ScatterplotLayer",
        data=st.session_state["stops"][1:],
        get_position="[lon, lat]",
        get_radius=70,
        get_fill_color=[255, 0, 0],
    )
    veh_layer = pdk.Layer(
        "IconLayer",
        data=list(st.session_state["vehicles"].values()),
        get_position="[lon, lat]",
        get_icon="type",
        icon_mapping={k: {"url": u, "width": 128, "height": 128, "anchorY": 128} for k, u in ICON_URLS.items()},
        get_size=4,
        size_scale=15,
        get_rotation="[heading]",
    )
    layers: List[pdk.Layer] = [depot_layer, stop_layer, veh_layer]
    if st.session_state["routes"]:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=[{"path": st.session_state["routes"]["truck"]}],
                get_path="path",
                get_color=[0, 0, 255],
                get_width=5,
            )
        )
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=[{"path": p} for p in st.session_state["routes"]["drones"]],
                get_path="path",
                get_color=[128, 0, 128],
                get_width=4,
            )
        )
    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/streets-v12",
        initial_view_state=pdk.ViewState(
            latitude=st.session_state["stops"][0]["lat"],
            longitude=st.session_state["stops"][0]["lon"],
            zoom=12,
        ),
        layers=layers,
        api_keys={"mapbox": MAPBOX_TOKEN},  # ensure token reaches browser
    )
    st.pydeck_chart(deck, use_container_width=True)

# ----------------------- Control panel ------------------------------------ #
with right:
    st.subheader("Stops (edit or import)")
    st.session_state["stops"] = st.data_editor(
        st.session_state["stops"],
        num_rows="dynamic",
        use_container_width=True,
        key="stops_editor",
    )
    up = st.file_uploader("Import CSV", type=["csv"])
    if up:
        try:
            df = pd.read_csv(up)
            df["lat"], df["lon"] = df["lat"].astype(float), df["lon"].astype(float)
            st.session_state["stops"] = df.to_dict("records")
            st.success(f"Imported {len(df)} stops – click Commit.")
        except Exception as e:
            st.error(str(e))

    col1, col2 = st.columns(2)

    # ---------------- route planner (truck uses Mapbox road path) ---------- #
    def plan_routes():
        stops = st.session_state["stops"]
        if len(stops) < 2:
            st.warning("Need at least depot and one customer.")
            return
        RANGE = 3.0
        truck_set = {0, len(stops) - 1}
        prev_cache, next_cache = {}, {}

        def prev_truck(i):
            if i not in prev_cache:
                prev_cache[i] = max(j for j in truck_set if j < i)
            return prev_cache[i]

        def next_truck(i):
            if i not in next_cache:
                next_cache[i] = min(j for j in truck_set if j > i)
            return next_cache[i]

        drone_cand = []
        for idx in range(1, len(stops) - 1):
            l, r = prev_truck(idx), next_truck(idx)
            d = (
                haversine_miles(stops[l]["lat"], stops[l]["lon"], stops[idx]["lat"], stops[idx]["lon"]) +
                haversine_miles(stops[idx]["lat"], stops[idx]["lon"], stops[r]["lat"], stops[r]["lon"])
            )
            if d <= RANGE:
                drone_cand.append((idx, l, r))
            else:
                truck_set.add(idx)

        # truck road path via Directions API
        truck_seq = sorted(truck_set)
        road_poly = []
        for a, b in zip(truck_seq, truck_seq[1:]):
            road_poly += road_path(stops[a]["lon"], stops[a]["lat"], stops[b]["lon"], stops[b]["lat"])[:-1]
        road_poly.append([stops[truck_seq[-1]]["lon"], stops[truck_seq[-1]]["lat"]])

        # drones
        bins = [[] for _ in range(4)]
        for n, triple in enumerate(drone_cand):
            bins[n % 4].append(triple)
        drone_polys = []
        for b in bins:
            poly: List[List[float]] = []
            for cid, l, r in b:
                pts = [
                    [stops[l]["lon"], stops[l]["lat"]],
                    [stops[cid]["lon"], stops[cid]["lat"]],
                    [stops[r]["lon"], stops[r]["lat"]],
                ]
                if poly and poly[-1] == pts[0]:
                    poly.extend(pts[1:])
                else:
                    poly.extend(pts)
            if poly:
                drone_polys.append(poly)

        st.session_state["routes"] = {"truck": road_poly, "drones": drone_polys}
        st.success("✅ Planned with Mapbox Directions (truck road path)")
        st.rerun()

    if col1.button("Commit / Plan"):
        plan_routes()

    if col2.button("Reset Telemetry"):
        st.session_state["vehicles"].clear()
        st.info("Cleared icons")

# ---------------------------------------------------------------------------
# end of file
# ---------------------------------------------------------------------------

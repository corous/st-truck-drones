# st-dashboard.py  – 2025‑06‑27
"""Streamlit planner + live tracking.

Key upgrade in this revision
---------------------------
* Truck route (blue line) is now **road‑accurate**. For every
  consecutive pair of truck stops we query the **Mapbox Directions** API
  (driving profile), decode the returned polyline, and stitch those
  segments together. The drones (purple lines) continue to fly straight.

Prerequisites
-------------
* Your Mapbox token must include the scope **directions:read**.
* Add `requests` and `polyline` to `requirements.txt` if not present.
"""

from __future__ import annotations
import json, math, os, threading, time, requests, polyline
from typing import List, Dict

import streamlit as st
import pydeck as pdk
import pandas as pd
import sseclient

# --------------------------------------------------
# 1. Page & Mapbox configuration
# --------------------------------------------------
st.set_page_config(page_title="Delivery Planner", layout="wide")

MAPBOX_TOKEN: str | None = (
    st.secrets.get("mapbox_token") or os.getenv("MAPBOX_TOKEN")
)
if not MAPBOX_TOKEN:
    st.error("Missing Mapbox token – add to secrets or env var.")
    st.stop()

MAPBOX_DIRECTIONS_URL = "https://api.mapbox.com/directions/v5/mapbox/driving"
SSE_URL = os.getenv("SSE_URL", "https://render-vehicles.onrender.com/stream")

pdk.settings.mapbox_api_key = MAPBOX_TOKEN  # backward compatibility

# --------------------------------------------------
# 2.  Constants & helpers
# --------------------------------------------------
ICON_URLS = {
    "truck": "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas/cars.png",
    "drone": "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas/rocket.png",
}

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


def road_segment(lon1: float, lat1: float, lon2: float, lat2: float) -> List[List[float]]:
    """Return a list[[lon, lat], …] following the road between two points."""
    params = {
        "geometries": "polyline6",
        "overview": "full",
        "access_token": MAPBOX_TOKEN,
    }
    url = f"{MAPBOX_DIRECTIONS_URL}/{lon1},{lat1};{lon2},{lat2}"
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    geometry = r.json()["routes"][0]["geometry"]
    coords = polyline.decode(geometry, precision=6)  # returns (lat, lon)
    # Convert to [lon, lat] for deck.gl
    return [[lon, lat] for lat, lon in coords]

# --------------------------------------------------
# 3.  Session‑state initialisation
# --------------------------------------------------
def_state = dict(
    vehicles={},
    stops=[{"id": "WH", "address": "Warehouse", "lat": 33.69321, "lon": -117.83345}],
    routes=None,
    listener=None,
    sse_status="🔄 connecting…",
)
for k, v in def_state.items():
    st.session_state.setdefault(k, v)

# --------------------------------------------------
# 4.  SSE background listener
# --------------------------------------------------

def _sse_listener():
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
    threading.Thread(target=_sse_listener, daemon=True).start()
    st.session_state["listener"] = True

# --------------------------------------------------
# 5.  UI layout – map left, controls right
# --------------------------------------------------
st.title("🚚 Truck + 🚁 Drone Delivery Planner")
st.caption(f"SSE feed status: {st.session_state['sse_status']}")
left, right = st.columns([3, 1])

# ----------------- 5A. Map pane ------------------- #
with left:
    depot_layer = pdk.Layer(
        "ScatterplotLayer", data=[st.session_state["stops"][0]],
        get_position="[lon, lat]", get_radius=90, get_fill_color=[0, 180, 0],
    )
    stops_layer = pdk.Layer(
        "ScatterplotLayer", data=st.session_state["stops"][1:],
        get_position="[lon, lat]", get_radius=70, get_fill_color=[255, 0, 0],
    )
    veh_layer = pdk.Layer(
        "IconLayer", data=list(st.session_state["vehicles"].values()),
        get_position="[lon, lat]", get_icon="type",
        icon_mapping={k: {"url": u, "width": 128, "height": 128, "anchorY": 128} for k, u in ICON_URLS.items()},
        get_size=4, size_scale=15, get_rotation="[heading]",
    )
    layers: List[pdk.Layer] = [depot_layer, stops_layer, veh_layer]

    if st.session_state["routes"]:
        layers.append(pdk.Layer("PathLayer", data=[{"path": st.session_state["routes"]["truck"]}],
                                 get_path="path", get_color=[0,0,255], get_width=5))
        layers.append(pdk.Layer("PathLayer", data=[{"path": p} for p in st.session_state["routes"]["drones"]],
                                 get_path="path", get_color=[128,0,128], get_width=4))

    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/streets-v12",
        initial_view_state=pdk.ViewState(
            latitude=st.session_state["stops"][0]["lat"],
            longitude=st.session_state["stops"][0]["lon"],
            zoom=12,
        ),
        layers=layers,
        api_keys={"mapbox": MAPBOX_TOKEN},
    )
    st.pydeck_chart(deck, use_container_width=True)

# ----------------- 5B. Control pane --------------- #
with right:
    st.subheader("Stops (edit or import)")
    st.session_state["stops"] = st.data_editor(
        st.session_state["stops"], num_rows="dynamic", use_container_width=True)

    up = st.file_uploader("Import CSV", type=["csv"])
    if up:
        try:
            df = pd.read_csv(up)
            df["lat"], df["lon"] = df["lat"].astype(float), df["lon"].astype(float)
            st.session_state["stops"] = df.to_dict("records")
            st.success("Imported – click Commit to plan")
        except Exception as exc:
            st.error(str(exc))

    col1, col2 = st.columns(2)

    def plan_routes():
        stops = st.session_state["stops"]
        if len(stops) < 2:
            st.warning("Need at least depot and one stop")
            return
        RANGE = 3.0
        truck_set = {0, len(stops)-1}
        prev_cache, next_cache = {}, {}
        def prev_t(i):
            if i not in prev_cache:
                prev_cache[i] = max(j for j in truck_set if j < i)
            return prev_cache[i]
        def next_t(i):
            if i not in next_cache:
                next_cache[i] = min(j for j in truck_set if j > i)
            return next_cache[i]

        cand = []
        for i in range(1, len(stops)-1):
            l, r = prev_t(i), next_t(i)
            dist = (
                haversine_miles(stops[l]["lat"], stops[l]["lon"], stops[i]["lat"], stops[i]["lon"]) +
                haversine_miles(stops[i]["lat"], stops[i]["lon"], stops[r]["lat"], stops[r]["lon"]))
            if dist <= RANGE:
                cand.append((i,l,r))
            else:
                truck_set.add(i)

        # road-aligned truck polyline
        seq = sorted(truck_set)
        road_poly: List[List[float]] = []
        for a, b in zip(seq, seq[1:]):
            road_poly += road_segment(stops[a]["lon"], stops[a]["lat"], stops[b]["lon"], stops[b]["lat"])[:-1]
        road_poly.append([stops[seq[-1]]["lon"], stops[seq[-1]]["lat"]])

        # drones straight legs
        bins = [[] for _ in range(4)]
        for n, tri in enumerate(cand):
            bins[n % 4].append(tri)
        drone_polys: List[List[List[float]]] = []
        for bucket in bins:
            if not bucket:
                continue
            poly: List[List[float]] = []
            for cid, l, r in bucket:
                pts = [
                    [stops[l]["lon"], stops[l]["lat"]],
                    [stops[cid]["lon"], stops[cid]["lat"]],
                    [stops[r]["lon"], stops[r

# st-dashboard.py
"""Streamlit delivery‑planner dashboard (hot‑fix 2025‑05‑22)
   • Real‑time vehicle telemetry via SSE
   • CSV/GUI stop editing and simple route planning
   • Renders truck (blue) + drone (purple) paths on Mapbox map

   Hot‑fix notes
   -----------------------------
   1. *add_script_run_ctx* import now optional. If running on a Streamlit
      build that doesn’t expose the internal API the dashboard still loads
      (context attachment is skipped, warnings may appear but are harmless).
   2. Ensured `st.set_page_config(...)` is the very first Streamlit call
      to avoid silent layout failures on some Streamlit Cloud images.
"""

from __future__ import annotations
import json, os, threading, time, math
from typing import List

import streamlit as st
import pydeck as pdk
import pandas as pd
import requests, sseclient

# ---------------------------------------------------------------------------
# 1.  PAGE CONFIG (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Delivery Planner", layout="wide")

# ---------------------------------------------------------------------------
# 2.  CONFIG & CONSTANTS
# ---------------------------------------------------------------------------
MAPBOX_TOKEN = st.secrets.get("mapbox_token") or os.getenv("MAPBOX_TOKEN")
if not MAPBOX_TOKEN:
    st.error("Missing Mapbox token – add mapbox_token to secrets or env var.")
    st.stop()

pdk.settings.mapbox_api_key = MAPBOX_TOKEN

SSE_URL = os.getenv("SSE_URL", "https://render-vehicles.onrender.com/stream")

ICON_URLS = {
    "truck": "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas/cars.png",
    "drone": "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas/rocket.png",
}

INITIAL_STOPS = [
    {"id": "WH", "address": "Warehouse", "lat": 33.69321, "lon": -117.83345},
]

# ---------------------------------------------------------------------------
# 3.  SESSION STATE DEFAULTS
# ---------------------------------------------------------------------------
state_defaults = dict(
    vehicles={},
    stops=INITIAL_STOPS.copy(),
    routes=None,
    listener=None,
    sse_status="🔄 connecting…",
)
for k, v in state_defaults.items():
    st.session_state.setdefault(k, v)

# ---------------------------------------------------------------------------
# 4.  Optional import of add_script_run_ctx
# ---------------------------------------------------------------------------
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx  # type: ignore
    _ATTACH_CTX = True
except Exception:
    _ATTACH_CTX = False

# ---------------------------------------------------------------------------
# 5.  SSE LISTENER (runs in daemon thread)
# ---------------------------------------------------------------------------

def _sse_listener(url: str):
    if _ATTACH_CTX:
        try:
            add_script_run_ctx(threading.current_thread())
        except Exception:
            pass

    while True:
        try:
            with requests.get(url, stream=True, timeout=30) as resp:
                resp.raise_for_status()
                resp.encoding = "utf-8"
                client = sseclient.SSEClient(resp)
                st.session_state["sse_status"] = "🟢 live"

                for event in client.events():
                    if not event.data:
                        continue
                    try:
                        data = json.loads(event.data)
                        st.session_state["vehicles"][data["id"]] = data
                        st.session_state["_ping"] = not st.session_state.get("_ping", False)
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception as err:
            st.session_state["sse_status"] = "🔴 disconnected – retrying…"
            print("[SSE]", err)
            time.sleep(5)
            continue

if st.session_state["listener"] is None:
    t = threading.Thread(target=_sse_listener, args=(SSE_URL,), daemon=True, name="sse")
    t.start()
    st.session_state["listener"] = t

# ---------------------------------------------------------------------------
# 6.  UI LAYOUT
# ---------------------------------------------------------------------------
st.title("🚚 Truck + 🚁 Drone Delivery Planner")
st.caption(f"SSE feed status: {st.session_state['sse_status']}")
left, right = st.columns([3, 1], gap="small")

# ----------------------- MAP PANE ----------------------- #
with left:
        # Green dot for warehouse (first stop)
    depot_layer = pdk.Layer(
        "ScatterplotLayer",
        data=[st.session_state["stops"][0]],
        get_position="[lon, lat]",
        get_radius=90,
        get_fill_color=[0, 180, 0],
    )

    # Red dots for all other stops
    stops_layer = pdk.Layer(
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
        icon_mapping={k: {"url": v, "width": 128, "height": 128, "anchorY": 128} for k, v in ICON_URLS.items()},
        get_size=4,
        size_scale=15,
        get_rotation="[heading]",
    )

        layers: List[pdk.Layer] = [depot_layer, stops_layer, veh_layer]

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
    )
    st.pydeck_chart(deck, use_container_width=True)

# ----------------------- CONTROL PANE ------------------- #
with right:
    st.subheader("Stops (edit or import)")
    st.session_state["stops"] = st.data_editor(
        st.session_state["stops"],
        num_rows="dynamic",
        use_container_width=True,
        key="stops_editor",
    )

    uploaded = st.file_uploader("Import CSV", type=["csv"], key="uploader")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            df["lat"] = df["lat"].astype(float)
            df["lon"] = df["lon"].astype(float)
            st.session_state["stops"] = df.to_dict("records")
            st.success(f"Imported {len(df)} stops from CSV – click Commit to plan.")
        except Exception as e:
            st.error(f"CSV import failed: {e}")

    col1, col2 = st.columns(2)

        def haversine_miles(lat1, lon1, lat2, lon2):
        R = 3958.8  # Earth radius in miles
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

    def plan_routes():
        """Multi‑agent planner with 3‑mile drone range and depot green marker."""
        stops = st.session_state["stops"]
        if len(stops) < 2:
            st.warning("Need at least depot and one destination/customer.")
            return

        RANGE_MILES = 3.0

        # Build initial truck list (all non‑drone or out‑of‑range customers will land here)
        truck_indices = {0, len(stops)-1}  # depot & final destination always truck

        drone_candidates = []
        for idx, s in enumerate(stops[1:-1], start=1):
            flag = str(s.get("drone", 0)).lower()
            if flag in ("1", "true", "yes"):
                drone_candidates.append(idx)
            else:
                truck_indices.add(idx)

        # insert customers violating range back to truck later
        launch_cache = {}
        recover_cache = {}

        def nearest_prev_truck(i):
            if i in launch_cache:
                return launch_cache[i]
            prev = max([idx for idx in sorted(truck_indices) if idx < i], default=0)
            launch_cache[i] = prev
            return prev

        def nearest_next_truck(i):
            if i in recover_cache:
                return recover_cache[i]
            nxt = min([idx for idx in sorted(truck_indices) if idx > i], default=len(stops)-1)
            recover_cache[i] = nxt
            return nxt

        feasible_drone_idxs = []
        for cust_idx in drone_candidates:
            launch_idx = nearest_prev_truck(cust_idx)
            recov_idx  = nearest_next_truck(cust_idx)
            dist_miles = (
                haversine_miles(stops[launch_idx]["lat"], stops[launch_idx]["lon"], stops[cust_idx]["lat"], stops[cust_idx]["lon"]) +
                haversine_miles(stops[cust_idx]["lat"], stops[cust_idx]["lon"], stops[recov_idx]["lat"], stops[recov_idx]["lon"])
            )
            if dist_miles <= RANGE_MILES:
                feasible_drone_idxs.append((cust_idx, launch_idx, recov_idx))
            else:
                truck_indices.add(cust_idx)

        truck_path = [[stops[i]["lon"], stops[i]["lat"]] for i in sorted(truck_indices)]

        NUM_DRONES = 4
        drone_bins: list[list[tuple]] = [[] for _ in range(NUM_DRONES)]
        for n, triple in enumerate(feasible_drone_idxs):
            drone_bins[n % NUM_DRONES].append(triple)

        drone_polylines: list[list[list[float]]] = []
        for bin_ in drone_bins:
            if not bin_:
                continue
            poly: list[list[float]] = []
            for cust_idx, launch_idx, recov_idx in bin_:
                launch = [stops[launch_idx]["lon"], stops[launch_idx]["lat"]]
                cust   = [stops[cust_idx]["lon"], stops[cust_idx]["lat"]]
                recov  = [stops[recov_idx]["lon"], stops[recov_idx]["lat"]]
                if poly and poly[-1] == launch:
                    poly.extend([cust, recov])
                else:
                    poly.extend([launch, cust, recov])
            drone_polylines.append(poly)

        st.session_state["routes"] = {"truck": truck_path, "drones": drone_polylines}
        st.success("✅ Routes replanned with 3‑mile drone limit – map updated.")
        st.rerun()

    if col1.button("Commit / Plan Route"):
        plan_routes()

    if col2.button("Reset Telemetry"):
        st.session_state["vehicles"].clear()
        st.session_state["_ping"] = not st.session_state.get("_ping", False)
        st.info("Vehicle positions cleared.")

# ---------------------------------------------------------------------------
# End of file
# ---------------------------------------------------------------------------

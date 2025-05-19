# app.py
import asyncio, json, websockets, streamlit as st, pydeck as pdk

MAPBOX_TOKEN = st.secrets["mapbox_token"]
WS_URL       = "wss://backend.example.com/telemetry"

ICON_URLS = {"truck": "https://…/truck.png", "drone": "https://…/drone.png"}
INITIAL_STOPS = [
    {"id": "A", "address": "123 Main", "lat": 33.6846, "lon": -117.8265},
    # … more …
]

# ---------- session-state bootstrap ----------
if "vehicles" not in st.session_state:
    st.session_state["vehicles"] = {}
if "ws_ready" not in st.session_state:
    st.session_state["ws_ready"] = False

# ---------- async websocket listener ----------
async def listener():
    async for msg in websockets.connect(WS_URL):
        v = json.loads(msg)
        st.session_state["vehicles"][v["id"]] = v
        st.session_state["ws_ready"] = True
        st.experimental_rerun()      # < 1 ms trigger

def ensure_ws_task():
    if "ws_task" not in st.session_state:
        loop = asyncio.get_event_loop()
        st.session_state["ws_task"] = loop.create_task(listener())

ensure_ws_task()

# ---------- UI layout ----------
st.title("Delivery-Planner Console")

left, right = st.columns([3, 1])

# 1  Map with moving markers
with left:
    stops_layer = pdk.Layer(
        "ScatterplotLayer",
        INITIAL_STOPS,
        get_position="[lon, lat]",
        get_fill_color=[200, 30, 0],
        get_radius=30,
    )
    veh_layer = pdk.Layer(
        "IconLayer",
        list(st.session_state["vehicles"].values()),
        get_position="[lon, lat]",
        get_icon="type",
        get_size=4,
        icon_mapping={k: {"url": v, "width": 128, "height": 128, "anchorY": 128} for k, v in ICON_URLS.items()},
        size_scale=15,
    )
    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/streets-v12",
            layers=[stops_layer, veh_layer],
            initial_view_state=pdk.ViewState(latitude=33.68, longitude=-117.82, zoom=12),
            mapbox_key=MAPBOX_TOKEN,
        ),
        use_container_width=True,
    )

# 2  Stop list and controls
with right:
    st.subheader("Stops")
    edited = st.data_editor(INITIAL_STOPS, num_rows="dynamic", use_container_width=True)
    col1, col2, col3 = st.columns(3)
    if col1.button("Import CSV"): ...
    if col2.button("Commit"): ...
    if col3.button("Reset"): st.session_state["vehicles"].clear()

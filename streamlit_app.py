import io
import time
import json
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

try:
    from model_utils import load_model, predict, draw_prediction_overlay
except ImportError as e:
    st.error(f"Cannot import model_utils.py: {e}")
    st.stop()


#  PAGE CONFIG

st.set_page_config(
    page_title = "Weld Defect AI",
    page_icon  = "🔬",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# Custom CSS 
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0f1117; }
  .defect-badge   { background:#dc2626; color:#fff; padding:6px 16px;
                    border-radius:6px; font-size:1.2rem; font-weight:700; }
  .ok-badge       { background:#16a34a; color:#fff; padding:6px 16px;
                    border-radius:6px; font-size:1.2rem; font-weight:700; }
  .metric-card    { background:#1e2130; border-radius:10px; padding:16px;
                    border:1px solid #2d3149; }
  .stTabs [data-baseweb="tab-list"] { gap:8px; }
  .stTabs [data-baseweb="tab"]      { font-weight:600; font-size:1rem; }
</style>
""", unsafe_allow_html=True)


#  SESSION STATE INITIALISATION

def init_state() -> None:
    if "model"       not in st.session_state: st.session_state.model        = None
    if "class_names" not in st.session_state: st.session_state.class_names  = []
    if "img_size"    not in st.session_state: st.session_state.img_size     = 224
    if "device"      not in st.session_state: st.session_state.device       = None
    if "history"     not in st.session_state: st.session_state.history      = []

init_state()


#  MODEL LOADER 

@st.cache_resource(show_spinner="Loading ConvNeXt model …")
def _load(_w, _c):
    return load_model(weights_path=_w, config_path=_c)


def ensure_model_loaded() -> bool:
    weights = Path("model_weights/best_model.pth")
    config  = Path("model_weights/model_config.json")

    if not weights.exists():
        return False

    if st.session_state.model is None:
        m, cls, sz, dev = _load(str(weights), str(config))
        st.session_state.model        = m
        st.session_state.class_names  = cls
        st.session_state.img_size     = sz
        st.session_state.device       = dev

    return True


#  HELPER: RUN PREDICTION + LOG TO HISTORY

def run_prediction(image: Image.Image, source: str = "upload") -> Optional[dict]:
    result = predict(
        st.session_state.model,
        image,
        st.session_state.class_names,
        st.session_state.img_size,
        st.session_state.device,
    )
    result["source"] = source
    result["time"]   = time.strftime("%H:%M:%S")
    st.session_state.history.append(result)
    return result


#  HELPER: RENDER RESULT CARD

def render_result(result: dict, col=None) -> None:
    ctx = col or st
    is_def = result["is_defect"]
    badge  = (
        '<span class="defect-badge">⚠ DEFECT DETECTED</span>'
        if is_def else
        '<span class="ok-badge">✔ NO DEFECT</span>'
    )
    ctx.markdown(badge, unsafe_allow_html=True)
    ctx.metric("Confidence", f"{result['confidence']*100:.1f}%")
    ctx.metric("Latency",    f"{result['latency_ms']:.1f} ms")

    ctx.markdown("**Class Probabilities**")
    for cls, prob in result["probabilities"].items():
        ctx.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")


#  SIDEBAR

with st.sidebar:
    st.title("🔬 Weld Defect AI")
    st.caption("ConvNeXt · Binary Classification")
    st.divider()

    if ensure_model_loaded():
        st.success("✔ Model loaded", icon="🤖")
        cfg_path = Path("model_weights/model_config.json")
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            st.json(cfg, expanded=False)
    else:
        st.error(
            "Model weights not found.\n\n"
            "Place `best_model.pth` and `model_config.json` "
            "inside a `model_weights/` folder next to this app."
        )

    st.divider()

    # Confidence threshold slider
    threshold = st.slider(
        "Defect confidence threshold",
        min_value=0.5, max_value=0.99,
        value=0.70, step=0.01,
        help="Predictions below this threshold show as UNCERTAIN.",
    )

    st.divider()

    # Prediction history
    if st.session_state.history:
        st.markdown("### Recent Predictions")
        for item in reversed(st.session_state.history[-8:]):
            col_a, col_b = st.columns([3, 2])
            col_a.write(f"**{item['label'].upper()}**")
            col_b.caption(f"{item['confidence']*100:.0f}%  {item['time']}")
        if st.button("🗑 Clear history"):
            st.session_state.history = []
            st.rerun()


#  MAIN AREA — TABS
st.title("Weld Defect Detection")
st.caption("AI-powered visual inspection system · ConvNeXt fine-tuned on weld imagery")
st.divider()

tab_upload, tab_camera, tab_live = st.tabs(
    ["Upload Image", "Camera Snapshot", "Live Stream"]
)



#  TAB 1 — Upload Image

with tab_upload:
    st.subheader("Upload a weld image for inspection")

    uploaded = st.file_uploader(
        "Drag & drop or click to browse",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Accepts JPEG, PNG, BMP, WEBP",
    )

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        c1, c2 = st.columns([1, 1])

        with c1:
            st.image(img, caption="Uploaded image", use_container_width=True)

        with c2:
            if not ensure_model_loaded():
                st.error("Model not loaded — check sidebar.")
            else:
                with st.spinner("Analysing …"):
                    result = run_prediction(img, source="upload")

                if result["confidence"] < threshold and not result["is_defect"]:
                    st.warning(
                        f"⚠ Low confidence ({result['confidence']*100:.1f}%) — "
                        "result may be uncertain."
                    )

                render_result(result)

        # Download result JSON
        st.download_button(
            label   = "⬇ Download result JSON",
            data    = json.dumps(result, indent=2),
            file_name= f"result_{uploaded.name}.json",
            mime    = "application/json",
        )


#  TAB 2 — Camera Snapshot

with tab_camera:
    st.subheader("Capture a photo from your camera")
    st.info(
        "Your browser will ask for camera permission. "
        "Point camera at the weld and click 'Take photo'.",
        icon="ℹ️",
    )

    snap = st.camera_input("Take photo")

    if snap is not None:
        img = Image.open(snap).convert("RGB")
        c1, c2 = st.columns([1, 1])

        with c1:
            st.image(img, caption="Camera capture", use_container_width=True)

        with c2:
            if not ensure_model_loaded():
                st.error("Model not loaded — check sidebar.")
            else:
                with st.spinner("Analysing …"):
                    result = run_prediction(img, source="camera_snapshot")
                render_result(result)


#  TAB 3 — Live Stream (streamlit-webrtc)

with tab_live:
    st.subheader("Live camera inference (WebRTC)")

    if not WEBRTC_AVAILABLE:
        st.warning(
            "**streamlit-webrtc** is not installed.\n\n"
            "```bash\n"
            "pip install streamlit-webrtc aiortc av\n"
            "```\n\n"
            "After installing, restart the Streamlit app.",
            icon="⚠️",
        )
    elif not ensure_model_loaded():
        st.error("Load the model first (check sidebar).")
    else:
        # ── Share model state across WebRTC threads
        _model_ref       = st.session_state.model
        _class_names_ref = st.session_state.class_names
        _img_size_ref    = st.session_state.img_size
        _device_ref      = st.session_state.device

        class WeldVideoProcessor(VideoProcessorBase):
            def __init__(self) -> None:
                self._result     = {}
                self._lock       = threading.Lock()
                self._prev_time  = time.perf_counter()
                self._fps        = 0.0
                # Run inference every N frames to save CPU/GPU
                self._frame_skip = 3
                self._frame_cnt  = 0

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                bgr = frame.to_ndarray(format="bgr24")

                self._frame_cnt += 1
                if self._frame_cnt % self._frame_skip == 0:
                    res = predict(
                        _model_ref, bgr,
                        _class_names_ref, _img_size_ref, _device_ref,
                    )
                    with self._lock:
                        self._result = res

                now = time.perf_counter()
                self._fps = 1.0 / (now - self._prev_time + 1e-9)
                self._prev_time = now

                with self._lock:
                    r = self._result

                if r:
                    bgr = draw_prediction_overlay(bgr, r, self._fps)

                return av.VideoFrame.from_ndarray(bgr, format="bgr24")

        RTC_CFG = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        ctx = webrtc_streamer(
            key               = "weld_live",
            video_processor_factory = WeldVideoProcessor,
            rtc_configuration = RTC_CFG,
            media_stream_constraints = {"video": True, "audio": False},
            async_processing  = True,
        )

        if ctx.video_processor:
            with ctx.video_processor._lock:
                live_result = ctx.video_processor._result

            if live_result:
                st.markdown("### Live Result")
                render_result(live_result)

        st.caption(
            "Note: For localhost testing, WebRTC works fine. "
            "For Streamlit Cloud, you need a TURN server."
        )
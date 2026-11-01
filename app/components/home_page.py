"""Home dashboard view displaying system architecture, hardware status, and quick launch links."""

import streamlit as st
from pathlib import Path
from utils.system_info import get_system_metrics
from config.settings import MODELS_DIR, OUTPUTS_DIR

def render_home_page():
    st.title("Edge Video Analytics Platform")
    st.caption("High-performance object detection, multi-target tracking, and telemetry on Apple Silicon")

    # Hardware Status Cards
    st.subheader("System Hardware & Runtime Environment")
    metrics = get_system_metrics()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Platform", value="Apple M2", delta=f"{metrics['architecture']}")
    with col2:
        mps_status = "Available" if metrics["mps_available"] else "Unavailable"
        st.metric(label="Metal (MPS)", value=mps_status, delta="Hardware Accel" if metrics["mps_available"] else "CPU Only")
    with col3:
        st.metric(
            label="System RAM",
            value=f"{metrics['available_ram_gb']} GB Free",
            delta=f"{metrics['ram_usage_percent']}% Used",
            delta_color="normal" if metrics['ram_usage_percent'] < 85 else "inverse"
        )
    with col4:
        st.metric(label="PyTorch Version", value=metrics["torch_version"], delta="Ultralytics 8.4+")

    st.divider()

    # Core Capabilities
    st.subheader("Platform Capabilities")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### Live Stream & Camera")
        st.write("Real-time inference using local FaceTime HD camera or external USB video device with Heads-Up Display (HUD) overlay and frame-level tracking.")
        if st.button("Launch Camera Detection", use_container_width=True, type="primary"):
            st.session_state["nav_selection"] = "Live Camera"
            st.rerun()

    with c2:
        st.markdown("#### Video File Analytics")
        st.write("Batch or live-rate analysis of pre-recorded videos. Outputs MP4 files annotated with unique tracking IDs, class labels, and bounding boxes.")
        if st.button("Analyze Video File", use_container_width=True):
            st.session_state["nav_selection"] = "Video Analysis"
            st.rerun()

    with c3:
        st.markdown("#### Telemetry Dashboard")
        st.write("Inspect time-series telemetry datasets, object frequency distributions, frame latency statistics, and tracking stability reports.")
        if st.button("Open Telemetry View", use_container_width=True):
            st.session_state["nav_selection"] = "Analytics Dashboard"
            st.rerun()

    st.divider()

    # Architecture Overview
    st.subheader("Processing Pipeline Architecture")
    st.markdown("""
```
+-----------------------------------------------------------------------------------------+
|                                    INPUT SOURCES                                        |
|  [ Built-in FaceTime HD Camera ]   |   [ Recorded Video (MP4/AVI) ]   |   [ Static Image ]   |
+--------------------------------------------+--------------------------------------------+
                                             | Frame Stream (BGR)
                                             v
+-----------------------------------------------------------------------------------------+
|                            INFERENCE & TRACKING PIPELINE                                |
|                                                                                         |
|  1. Image Preprocessing (Letterbox resize to 640/320 px, FP32 Normalization)            |
|  2. YOLOv8n Feed-Forward Pass (Apple M2 CPU or Metal Performance Shaders / MPS)        |
|  3. Non-Maximum Suppression (NMS filter by confidence >= ConfThresh, IoU >= IoUThresh)  |
|  4. ByteTrack Association (Kalman Filter state estimation + Hungarian cost assignment)  |
+--------------------------------------------+--------------------------------------------+
                                             | Detected Objects + Persistent Track IDs
                                             v
+-----------------------------------------------------------------------------------------+
|                                TELEMETRY & PERSISTENCE                                  |
|                                                                                         |
|  * Real-Time Analytics Engine (FPS, Inference Latency, Cumulative Object Register)      |
|  * Overlay Renderer (Anti-aliased bounding boxes, unique ID tags, Top-deck HUD banner)   |
|  * Video Exporter (OpenCV VideoWriter MP4 encoder)                                      |
|  * Telemetry Log (Exportable CSV time-series for post-run audit)                        |
+-----------------------------------------------------------------------------------------+
```
    """)

    # Available Assets
    st.subheader("Local Storage Status")
    models = list(MODELS_DIR.glob("*.pt"))
    outputs = list(OUTPUTS_DIR.glob("*.*"))

    st.write(f"- **Local Models**: {len(models)} found ({', '.join([m.name for m in models])})")
    st.write(f"- **Exported Outputs**: {len(outputs)} files stored in `data/outputs/`")

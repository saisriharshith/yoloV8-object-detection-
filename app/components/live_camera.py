"""Live Camera Detection and Tracking component for Streamlit."""

import cv2
import time
import streamlit as st
from pathlib import Path
from core.detector import ObjectDetector
from core.tracker import ObjectTracker
from core.video_stream import VideoStream, get_available_cameras
from core.analytics import AnalyticsEngine
from utils.visualization import draw_detections, draw_hud
from config.settings import OUTPUTS_DIR

def render_live_camera_page():
    st.title("Live Camera Analytics")
    st.caption("Live video stream object detection and tracking using local camera devices")

    # Sidebar controls are pulled from global session state settings
    model_name = st.session_state.get("model_name", "models/yolov8n.pt")
    device = st.session_state.get("device", "cpu")
    conf_thresh = st.session_state.get("conf_thresh", 0.35)
    iou_thresh = st.session_state.get("iou_thresh", 0.45)
    img_size = st.session_state.get("img_size", 640)
    enabled_classes = st.session_state.get("enabled_classes", None)

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])
    with col_ctrl1:
        camera_idx = st.number_input("Camera Index", min_value=0, max_value=4, value=0, step=1)
    with col_ctrl2:
        mode = st.radio("Pipeline Mode", ["Tracking (ByteTrack)", "Pure Detection"], horizontal=True)
    with col_ctrl3:
        downscale = st.checkbox("Streamlit Display Downscale (Speedup UI)", value=True)

    # State management
    if "camera_running" not in st.session_state:
        st.session_state["camera_running"] = False

    b_col1, b_col2, b_col3 = st.columns([1, 1, 3])
    with b_col1:
        start_btn = st.button("Start Camera", type="primary", disabled=st.session_state["camera_running"])
    with b_col2:
        stop_btn = st.button("Stop Camera", disabled=not st.session_state["camera_running"])

    if start_btn:
        st.session_state["camera_running"] = True
        st.rerun()

    if stop_btn:
        st.session_state["camera_running"] = False
        st.rerun()

    # Metric placeholders
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    metric_fps = m_col1.empty()
    metric_lat = m_col2.empty()
    metric_cur = m_col3.empty()
    metric_uniq = m_col4.empty()

    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    if st.session_state["camera_running"]:
        status_placeholder.info(f"Opening camera device {camera_idx} on device '{device.upper()}'...")

        try:
            # Initialize vision engine
            if mode == "Tracking (ByteTrack)":
                engine = ObjectTracker(
                    model_name=model_name,
                    device=device,
                    conf_threshold=conf_thresh,
                    iou_threshold=iou_thresh,
                    img_size=img_size,
                    enabled_classes=enabled_classes
                )
            else:
                engine = ObjectDetector(
                    model_name=model_name,
                    device=device,
                    conf_threshold=conf_thresh,
                    iou_threshold=iou_thresh,
                    img_size=img_size,
                    enabled_classes=enabled_classes
                )

            analytics = AnalyticsEngine(fps_window_size=20)
            stream = VideoStream(source=camera_idx)
            status_placeholder.success(f"Camera active ({stream.width}x{stream.height} @ ~{stream.fps:.0f} FPS). Click 'Stop Camera' to end.")

            while st.session_state.get("camera_running", False):
                t_frame_start = analytics.start_frame()
                ret, frame = stream.read()
                if not ret:
                    st.error("Failed to acquire frame from camera. Device may have been disconnected.")
                    break

                # Inference
                if mode == "Tracking (ByteTrack)":
                    results, lat_ms = engine.track(frame)
                else:
                    results, lat_ms = engine.detect(frame)

                # Update analytics
                metrics = analytics.update(t_frame_start, lat_ms, results)

                # Render annotations
                annotated = draw_detections(
                    frame,
                    results,
                    show_boxes=True,
                    show_labels=True,
                    show_confidence=True,
                    show_track_id=(mode == "Tracking (ByteTrack)")
                )
                annotated = draw_hud(
                    annotated,
                    fps=metrics["fps"],
                    latency_ms=metrics["latency_ms"],
                    current_count=metrics["current_count"],
                    cumulative_unique_count=metrics["cumulative_unique_total"],
                    device=device,
                    class_counts=metrics["current_classes"]
                )

                # Update metrics
                metric_fps.metric("Processing FPS", f"{metrics['fps']:.1f}")
                metric_lat.metric("Model Latency", f"{metrics['latency_ms']:.1f} ms")
                metric_cur.metric("Current In-Frame", f"{metrics['current_count']}")
                metric_uniq.metric("Total Unique Tracked", f"{metrics['cumulative_unique_total']}")

                # Convert BGR to RGB for Streamlit
                display_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                if downscale and display_frame.shape[1] > 960:
                    display_frame = cv2.resize(display_frame, (960, int(display_frame.shape[0] * (960 / display_frame.shape[1]))))

                frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)

            stream.release()

        except Exception as e:
            st.session_state["camera_running"] = False
            status_placeholder.error(f"Camera Error: {e}")
    else:
        frame_placeholder.info("Camera stream is currently inactive. Select your camera index and click 'Start Camera'.")
        metric_fps.metric("Processing FPS", "0.0")
        metric_lat.metric("Model Latency", "0.0 ms")
        metric_cur.metric("Current In-Frame", "0")
        metric_uniq.metric("Total Unique Tracked", "0")

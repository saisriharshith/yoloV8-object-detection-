"""Recorded Video Upload, Processing, Tracking, and Export component."""

import cv2
import time
import tempfile
import streamlit as st
import pandas as pd
from pathlib import Path

from core.detector import ObjectDetector
from core.tracker import ObjectTracker
from core.video_stream import VideoStream
from core.analytics import AnalyticsEngine
from utils.visualization import draw_detections, draw_hud
from utils.video_exporter import VideoExporter
from config.settings import SAMPLES_DIR, INPUTS_DIR, OUTPUTS_DIR

def render_video_page():
    st.title("Recorded Video Analytics & Export")
    st.caption("Upload or select recorded video files for frame-by-frame object detection, tracking, and MP4 export")

    # Global inference settings
    model_name = st.session_state.get("model_name", "models/yolov8n.pt")
    device = st.session_state.get("device", "cpu")
    conf_thresh = st.session_state.get("conf_thresh", 0.35)
    iou_thresh = st.session_state.get("iou_thresh", 0.45)
    img_size = st.session_state.get("img_size", 640)
    enabled_classes = st.session_state.get("enabled_classes", None)

    # Source selection
    source_type = st.radio("Select Video Source", ["Sample Videos", "Upload Video File"], horizontal=True)
    video_path = None

    if source_type == "Sample Videos":
        samples = list(SAMPLES_DIR.glob("*.mp4")) + list(SAMPLES_DIR.glob("*.avi"))
        if samples:
            sample_names = [s.name for s in samples]
            selected_name = st.selectbox("Choose sample video", sample_names)
            video_path = SAMPLES_DIR / selected_name
        else:
            st.warning("No sample videos found in `data/samples/`.")
    else:
        uploaded_file = st.file_uploader("Upload video file (MP4, AVI, MOV)", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            # Save uploaded video to data/inputs/
            dest_path = INPUTS_DIR / uploaded_file.name
            with open(dest_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_path = dest_path
            st.success(f"Uploaded file saved: {uploaded_file.name}")

    if video_path and video_path.exists():
        st.subheader(f"Target: {video_path.name}")

        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            mode = st.selectbox("Processing Mode", ["Tracking (ByteTrack)", "Pure Detection"])
        with col_opt2:
            process_limit = st.number_input("Max Frames to Process (0 = Entire Video)", min_value=0, max_value=5000, value=150, step=30)
        with col_opt3:
            export_output = st.checkbox("Export Processed MP4 Video", value=True)

        # Action button
        if "video_processing" not in st.session_state:
            st.session_state["video_processing"] = False

        c_btn1, c_btn2 = st.columns([1, 4])
        with c_btn1:
            run_btn = st.button("Start Analysis", type="primary", disabled=st.session_state["video_processing"])

        if run_btn:
            st.session_state["video_processing"] = True
            
            # Setup output paths
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            out_stem = f"{video_path.stem}_processed_{timestamp_str}"
            out_video_path = OUTPUTS_DIR / f"{out_stem}.mp4"
            out_csv_path = OUTPUTS_DIR / f"{out_stem}_telemetry.csv"

            # Progress UI elements
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            metric_fps = m_col1.empty()
            metric_lat = m_col2.empty()
            metric_cur = m_col3.empty()
            metric_uniq = m_col4.empty()

            preview_col1, preview_col2 = st.columns(2)
            with preview_col1:
                st.caption("Input Frame")
                raw_preview = st.empty()
            with preview_col2:
                st.caption("Annotated Frame & Analytics")
                annotated_preview = st.empty()

            try:
                # Open stream
                stream = VideoStream(video_path)
                total_frames = stream.total_frames
                target_frames = min(total_frames, process_limit) if process_limit > 0 else total_frames

                # Initialize models
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

                analytics = AnalyticsEngine(fps_window_size=30)
                exporter = (
                    VideoExporter(
                        output_path=out_video_path,
                        fps=stream.fps,
                        frame_size=(stream.width, stream.height),
                        codec="mp4v"
                    )
                    if export_output
                    else None
                )

                frame_idx = 0
                start_clock = time.time()

                while frame_idx < target_frames:
                    t_frame_start = analytics.start_frame()
                    ret, frame = stream.read()
                    if not ret:
                        break

                    # Inference
                    if mode == "Tracking (ByteTrack)":
                        results, lat_ms = engine.track(frame)
                    else:
                        results, lat_ms = engine.detect(frame)

                    metrics = analytics.update(t_frame_start, lat_ms, results)

                    # Draw annotations
                    annotated = draw_detections(
                        frame.copy(),
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

                    if exporter:
                        exporter.write_frame(annotated)

                    frame_idx += 1
                    prog = frame_idx / target_frames if target_frames > 0 else 1.0
                    progress_bar.progress(prog)
                    status_text.text(f"Processing frame {frame_idx}/{target_frames} ({prog*100:.1f}%)")

                    # Live metrics
                    metric_fps.metric("Processing FPS", f"{metrics['fps']:.1f}")
                    metric_lat.metric("Model Latency", f"{metrics['latency_ms']:.1f} ms")
                    metric_cur.metric("Current In-Frame", f"{metrics['current_count']}")
                    metric_uniq.metric("Total Unique Tracked", f"{metrics['cumulative_unique_total']}")

                    # Update preview every few frames to keep UI responsive
                    if frame_idx % 3 == 0 or frame_idx == target_frames:
                        raw_preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        annotated_preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

                stream.release()
                export_meta = exporter.release() if exporter else None
                total_duration = time.time() - start_clock
                summary = analytics.get_summary()

                # Save telemetry CSV
                df_telemetry = analytics.to_dataframe()
                df_telemetry.to_csv(out_csv_path, index=False)

                st.session_state["video_processing"] = False
                st.session_state["latest_telemetry_csv"] = str(out_csv_path)

                st.success(f"Processing Complete! {frame_idx} frames processed in {total_duration:.2f}s ({frame_idx/total_duration:.1f} FPS avg).")

                # Summary Section
                st.subheader("Processing Summary")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.write(f"- **Total Frames**: {frame_idx}")
                    st.write(f"- **Throughput**: {frame_idx/total_duration:.1f} FPS")
                    st.write(f"- **Average Latency**: {summary['average_latency_ms']:.1f} ms")
                    st.write(f"- **Total Unique Tracked**: {summary['total_unique_objects']}")
                with res_col2:
                    st.write("**Unique Object Breakdown:**")
                    st.json(summary["unique_by_class"])

                # Download Buttons
                d_col1, d_col2 = st.columns(2)
                if export_meta and out_video_path.exists():
                    with open(out_video_path, "rb") as vf:
                        v_bytes = vf.read()
                    with d_col1:
                        st.download_button(
                            label="Download Processed Video (MP4)",
                            data=v_bytes,
                            file_name=out_video_path.name,
                            mime="video/mp4",
                            type="primary",
                            use_container_width=True
                        )

                with open(out_csv_path, "rb") as cf:
                    c_bytes = cf.read()
                with d_col2:
                    st.download_button(
                        label="Download Telemetry Log (CSV)",
                        data=c_bytes,
                        file_name=out_csv_path.name,
                        mime="text/csv",
                        use_container_width=True
                    )

            except Exception as e:
                st.session_state["video_processing"] = False
                st.error(f"Error during video processing: {e}")
